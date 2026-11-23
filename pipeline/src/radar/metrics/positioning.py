"""CFTC Commitments of Traders Positioning analytics (no look-ahead bias).

Formula:
- Net Speculative Contracts = NonComm_Long - NonComm_Short
- Net % of Open Interest = (NonComm_Long - NonComm_Short) / Open_Interest
- Expanding percentile & Z-score computed strictly against prior weekly reports
  (min_expanding_weeks = 52).
"""

from __future__ import annotations

from typing import Any

from radar.metrics.core import (
    compute_effective_n,
    compute_expanding_percentile_series,
    compute_expanding_zscore_series,
    compute_lag1_autocorr,
)


def compute_cot_contract_metrics(
    weekly_rows: list[dict[str, Any]],
    min_obs: int = 52,
) -> list[dict[str, Any]]:
    """Compute Net Speculative, % of Open Interest, expanding percentile, and Z-score."""
    if not weekly_rows:
        return []

    net_pct_series: list[float | None] = []
    net_contracts: list[float] = []
    for r in weekly_rows:
        oi = float(r["open_interest"])
        net = float(r["noncomm_long"]) - float(r["noncomm_short"])
        net_contracts.append(net)
        net_pct_series.append((net / oi) if oi > 0 else None)

    pct_series = compute_expanding_percentile_series(net_pct_series, min_obs=min_obs)
    z_series = compute_expanding_zscore_series(net_pct_series, min_obs=min_obs)

    out: list[dict[str, Any]] = []
    for i, r in enumerate(weekly_rows):
        npct = net_pct_series[i]
        out.append(
            {
                "date": r["date"],
                "open_interest": int(round(r["open_interest"])),
                "noncomm_long": int(round(r["noncomm_long"])),
                "noncomm_short": int(round(r["noncomm_short"])),
                "net_speculative": int(round(net_contracts[i])),
                "net_pct_oi": round(npct, 4) if npct is not None else None,
                "percentile": round(pct_series[i], 1) if pct_series[i] is not None else None,
                "z_score": round(z_series[i], 2) if z_series[i] is not None else None,
            }
        )
    return out


def build_positioning_payload(
    cot_raw: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Assemble positioning.json from CFTC COT contract series."""
    import yaml
    from pathlib import Path

    pos_cfg = params.get("positioning", {})
    min_weeks = int(pos_cfg.get("min_expanding_weeks", 52))
    tail_weeks = int(pos_cfg.get("tail_weeks", 156))
    
    # Load contracts from cot_contracts.yaml if available, else params
    cot_yaml_path = Path(__file__).resolve().parents[3] / "config" / "cot_contracts.yaml"
    contracts_cfg = []
    if cot_yaml_path.exists():
        with open(cot_yaml_path, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}
            contracts_cfg = cfg_data.get("contracts", [])
    if not contracts_cfg:
        contracts_cfg = pos_cfg.get("contracts", [])

    raw_by_code = cot_raw.get("contracts", {})

    contracts_out: list[dict[str, Any]] = []
    for cfg in contracts_cfg:
        code = str(cfg.get("cftc_contract_code", cfg.get("cftc_code", "")))
        rows = raw_by_code.get(code, [])
        enriched = compute_cot_contract_metrics(rows, min_obs=min_weeks)
        npcts = [x["net_pct_oi"] for x in enriched]
        n_obs = sum(1 for x in npcts if x is not None)
        n_eff = compute_effective_n(n_obs, compute_lag1_autocorr(npcts)) if n_obs > 10 else n_obs
        
        latest = enriched[-1] if enriched else None
        prev_1w = enriched[-2] if len(enriched) >= 2 else None
        prev_4w = enriched[-5] if len(enriched) >= 5 else None
        prev_13w = enriched[-14] if len(enriched) >= 14 else None

        chg_1w = (
            round((latest["net_pct_oi"] - prev_1w["net_pct_oi"]), 4)
            if (latest and prev_1w and latest["net_pct_oi"] is not None and prev_1w["net_pct_oi"] is not None)
            else None
        )
        chg_4w = (
            round((latest["net_pct_oi"] - prev_4w["net_pct_oi"]), 4)
            if (latest and prev_4w and latest["net_pct_oi"] is not None and prev_4w["net_pct_oi"] is not None)
            else None
        )
        chg_13w = (
            round((latest["net_pct_oi"] - prev_13w["net_pct_oi"]), 4)
            if (latest and prev_13w and latest["net_pct_oi"] is not None and prev_13w["net_pct_oi"] is not None)
            else None
        )

        p_now = latest["percentile"] if latest else None
        p_prev = prev_1w["percentile"] if prev_1w else None
        is_extreme = (
            (p_now is not None and p_prev is not None)
            and ((p_now >= 95.0 and p_prev >= 95.0) or (p_now <= 5.0 and p_prev <= 5.0))
        )
        extreme_type = (
            "crowded_long" if (p_now is not None and p_now >= 95.0)
            else ("crowded_short" if (p_now is not None and p_now <= 5.0) else "neutral")
        )

        contracts_out.append(
            {
                "id": cfg["id"],
                "name": cfg["name"],
                "cftc_code": code,
                "asset_class": cfg.get("asset_class", "general"),
                "category": cfg.get("category", "leveraged_funds"),
                "n": n_obs,
                "n_eff": n_eff,
                "latest_date": latest["date"] if latest else None,
                "open_interest": latest["open_interest"] if latest else None,
                "net_speculative": latest["net_speculative"] if latest else None,
                "net_pct_oi": latest["net_pct_oi"] if latest else None,
                "chg_1w_pct_oi": chg_1w,
                "chg_4w_pct_oi": chg_4w,
                "chg_13w_pct_oi": chg_13w,
                "percentile": p_now,
                "percentile_4w_ago": prev_4w["percentile"] if prev_4w else None,
                "percentile_13w_ago": prev_13w["percentile"] if prev_13w else None,
                "z_score": latest["z_score"] if latest else None,
                "is_extreme": is_extreme,
                "extreme_type": extreme_type,
                "lag_label": "Positions as of Tuesday, published Friday",
                "series": enriched[-tail_weeks:],
            }
        )

    return {
        "status": cot_raw.get("status", "ok"),
        "error": cot_raw.get("error"),
        "contracts": contracts_out,
    }
