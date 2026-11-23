"""Batch runner to generate historical analogues JSON payloads.

Computes Ledoit-Wolf shrinkage Mahalanobis distance, 20D declustered historical matches,
walk-forward skill scores, and quantile fan distributions for all presets in analogues.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

from radar.metrics.core import (
    compute_drawdown,
    compute_period_return,
    compute_realized_vol_series,
    compute_returns,
    compute_trend_tstat_series,
)
from radar.metrics.analogues import build_analogues_payloads

REPO_ROOT = Path(__file__).resolve().parents[4]
SERIES_DIR = REPO_ROOT / "data" / "processed" / "series"
CONFIG_FILE = REPO_ROOT / "pipeline" / "config" / "analogues.yaml"
OUT_DIRS = [
    REPO_ROOT / "web" / "public" / "data" / "v1" / "analogues",
    REPO_ROOT / "web" / "out" / "data" / "v1" / "analogues",
]

RATE_SERIES = {
    "us2y", "us5y", "us10y", "us30y", "us2s10s", "us10y_real", "us10y_breakeven", "sofr", "vix", "indiavix", "hy_spread", "ig_spread"
}


def load_series_metrics() -> dict[str, dict[str, Any]]:
    asset_metrics: dict[str, dict[str, Any]] = {}
    
    # Load all parquet files
    parquet_files = list(SERIES_DIR.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No series found in {SERIES_DIR}")

    # Determine common calendar dates from SPX
    spx_df = pd.read_parquet(SERIES_DIR / "spx.parquet")
    common_dates = [d.strftime("%Y-%m-%d") for d in spx_df.index]

    for p in parquet_files:
        asset_id = p.stem.lower()
        df = pd.read_parquet(p)
        df_dates = [d.strftime("%Y-%m-%d") for d in df.index]
        
        # Align to common dates
        s = df["value"].reindex(spx_df.index).ffill().bfill()
        vals: list[float | None] = [float(v) if pd.notna(v) else None for v in s]
        is_rate = asset_id in RATE_SERIES

        # Calculate metrics
        rets = compute_returns(vals, is_rate=is_rate)
        ret_20d = compute_period_return(vals, window=20, is_rate=is_rate)
        vol20 = compute_realized_vol_series(rets, window=20)
        dd = compute_drawdown(vals)
        trend = compute_trend_tstat_series(vals, window=60, is_rate=is_rate)

        asset_data = {
            "dates": common_dates,
            "values": vals,
            "is_rate": is_rate,
            "ret_20d": ret_20d,
            "vol20": vol20,
            "drawdown": dd,
            "trend_tstat": trend,
        }
        # Support both lower and upper case keys
        asset_metrics[asset_id] = asset_data
        asset_metrics[asset_id.upper()] = asset_data

    # Map aliases if needed (e.g., TLT, HYG, WTI)
    if "brent" in asset_metrics and "WTI" not in asset_metrics:
        asset_metrics["WTI"] = asset_metrics["brent"]
    if "hy_spread" in asset_metrics and "HYG" not in asset_metrics:
        asset_metrics["HYG"] = asset_metrics["hy_spread"]
    if "us30y" in asset_metrics and "TLT" not in asset_metrics:
        asset_metrics["TLT"] = asset_metrics["us30y"]

    return asset_metrics


def run() -> None:
    print(f"Loading config from {CONFIG_FILE}...")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("Computing core asset metrics...")
    asset_metrics = load_series_metrics()

    # In our dataset, min_obs can be adjusted to 252 (1 trading year) if total sample is 754
    cfg["engine"]["min_expanding_obs"] = 250

    print("Running Ledoit-Wolf Mahalanobis Historical Analogue Engine...")
    payloads = build_analogues_payloads(cfg, asset_metrics)

    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write presets index
        presets_index = {
            "schema_version": "1.0.0",
            "as_of": "2026-11-20",
            "code_version": "v1.0.0",
            "status": "ok",
            "presets": payloads["presets_index"]["presets"],
        }
        with open(out_dir / "presets.json", "w", encoding="utf-8") as f:
            json.dump(presets_index, f, indent=2)

        # Write evaluation
        eval_payload = {
            "schema_version": "1.0.0",
            "as_of": "2026-11-20",
            "code_version": "v1.0.0",
            "status": "ok",
            **payloads["evaluation"],
        }
        with open(out_dir / "evaluation.json", "w", encoding="utf-8") as f:
            json.dump(eval_payload, f, indent=2)

        # Write each preset
        for pid, preset_detail in payloads["by_preset"].items():
            wrapped = {
                "schema_version": "1.0.0",
                "as_of": "2026-11-20",
                "code_version": "v1.0.0",
                "status": "ok",
                **preset_detail,
            }
            with open(out_dir / f"{pid}.json", "w", encoding="utf-8") as f:
                json.dump(wrapped, f, indent=2)

    print("Successfully generated all historical analogue JSON files!")


if __name__ == "__main__":
    run()
