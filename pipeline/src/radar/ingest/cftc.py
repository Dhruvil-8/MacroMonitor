"""Zero-key CFTC Commitments of Traders (Legacy Futures-Only) Socrata ingestion.

Endpoint: https://publicreporting.cftc.gov/resource/jun7-fc8e.json
"""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def parse_cftc_rows(raw_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group and sort raw CFTC Socrata rows by contract market code."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in raw_rows:
        code = str(row.get("cftc_contract_market_code", "")).strip()
        d_raw = str(row.get("report_date_as_yyyy_mm_dd", ""))[:10]
        if not code or len(d_raw) != 10:
            continue
        try:
            oi = float(row.get("open_interest_all", 0) or 0)
            nc_long = float(row.get("noncomm_positions_long_all", 0) or 0)
            nc_short = float(row.get("noncomm_positions_short_all", 0) or 0)
            c_long = float(row.get("comm_positions_long_all", 0) or 0)
            c_short = float(row.get("comm_positions_short_all", 0) or 0)
        except (TypeError, ValueError):
            continue
        if oi <= 0:
            continue
        grouped.setdefault(code, {})[d_raw] = {
            "date": d_raw,
            "open_interest": oi,
            "noncomm_long": nc_long,
            "noncomm_short": nc_short,
            "comm_long": c_long,
            "comm_short": c_short,
        }

    return {
        code: [date_map[d] for d in sorted(date_map.keys())]
        for code, date_map in grouped.items()
    }


def _fetch_cftc_socrata(codes: list[str], limit: int = 5000, retries: int = 3) -> list[dict[str, Any]]:
    where = " OR ".join(f"cftc_contract_market_code='{c}'" for c in codes)
    params = urllib.parse.urlencode(
        {
            "$where": where,
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": str(limit),
        }
    )
    url = f"https://publicreporting.cftc.gov/resource/jun7-fc8e.json?{params}"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "GlobalMarketRadar/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Failed to fetch CFTC COT feed: {last_err}")


def ingest_cftc_cot(
    contracts: list[dict[str, Any]],
    raw_root: Path,
    run_date: str | None = None,
) -> dict[str, Any]:
    """Ingest and cache CFTC weekly Commitments of Traders snapshot."""
    today = dt.datetime.now(tz=dt.timezone.utc).date()
    snap_date = run_date or today.isoformat()
    snap_dir = raw_root / snap_date / "cftc"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_file = snap_dir / "cot_legacy.json"

    if snap_file.exists():
        return json.loads(snap_file.read_text(encoding="utf-8"))

    codes = [str(c["cftc_code"]) for c in contracts]
    try:
        raw_rows = _fetch_cftc_socrata(codes)
        by_code = parse_cftc_rows(raw_rows)
        payload = {
            "status": "ok",
            "error": None,
            "fetched_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
            "contracts": by_code,
        }
        snap_file.write_text(json.dumps(payload), encoding="utf-8")
        return payload
    except Exception as exc:
        if raw_root.exists():
            for d in sorted([p for p in raw_root.iterdir() if p.is_dir()], reverse=True):
                cand = d / "cftc" / "cot_legacy.json"
                if cand.exists():
                    fb = json.loads(cand.read_text(encoding="utf-8"))
                    fb["status"] = "cached_fallback"
                    fb["error"] = str(exc)
                    return fb
        return {"status": "failed", "error": str(exc), "contracts": {}}
