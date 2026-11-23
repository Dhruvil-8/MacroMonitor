# pipeline/src/radar/metrics/sector.py
"""Sector layer quantitative analytics (Section 4.2 of docs/MASTER_PLAN.md).

Implements:
1. Approximate sector contribution: contrib_s = w_s * r_s with w_s from published index factsheets.
2. Explicit residual: residual = r_index - sum_s(contrib_s), guaranteeing exact arithmetic reconciliation.
3. Sector rotation: 60-day relative strength ratio, 10-day momentum, 10-day history trail, and quadrant classification.
4. Multi-horizon sector heatmap: returns and anti-lookahead expanding percentiles across {1D, 5D, 1M, 3M, 1Y}.
5. Sector breadth: share of sectors beating the benchmark index.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd

from radar.metrics.rotation import (
    compute_relative_strength_ratio,
    compute_ratio_momentum,
    classify_rotation_phase,
)
from radar.metrics.stats import compute_expanding_percentile

LOGGER = logging.getLogger(__name__)

HORIZON_MAP = {
    "1D": 1,
    "5D": 5,
    "1M": 21,
    "3M": 63,
    "1Y": 252,
}

SECTOR_NAMES: dict[str, str] = {
    # US GICS Sectors
    "xlk": "Information Technology",
    "xlf": "Financials",
    "xlv": "Health Care",
    "xly": "Consumer Discretionary",
    "xlc": "Communication Services",
    "xli": "Industrials",
    "xlp": "Consumer Staples",
    "xle": "Energy",
    "xlu": "Utilities",
    "xlre": "Real Estate",
    "xlb": "Materials",
    # India Nifty Sectors
    "niftybank": "Financial Services",
    "niftyit": "Information Technology",
    "niftyfmcg": "Fast Moving Consumer Goods",
    "niftyauto": "Automobile",
    "niftypharma": "Healthcare & Pharma",
    "niftymetal": "Metals & Mining",
    "niftyrealty": "Real Estate",
}


def compute_cumulative_return(prices: pd.Series, window: int) -> float:
    """Compute cumulative return over window days: P_t / P_{t-w} - 1.0."""
    s = prices.dropna()
    if len(s) < window + 1:
        return 0.0
    val_now = float(s.iloc[-1])
    val_prior = float(s.iloc[-(window + 1)])
    if val_prior <= 0:
        return 0.0
    return float(val_now / val_prior - 1.0)


def compute_return_series_percentile(prices: pd.Series, window: int, min_obs: int = 120) -> float:
    """Compute current expanding percentile rank of rolling window return against history."""
    s = prices.dropna()
    if len(s) < window + min_obs:
        return 50.0  # Median neutral fallback if insufficient history
    # Rolling window returns
    rolling_returns = (s / s.shift(window) - 1.0).dropna()
    pct_series = compute_expanding_percentile(rolling_returns, min_obs=min_obs)
    valid_pct = pct_series.dropna()
    if valid_pct.empty:
        return 50.0
    return float(valid_pct.iloc[-1])


def compute_sector_breadth(sector_returns: dict[str, float], index_return: float) -> float:
    """Compute share of sectors beating the benchmark index return.

    Formula:
        breadth = sum(I(r_s > r_index)) / N_sectors
    Returns float in [0.0, 1.0].
    """
    if not sector_returns:
        return 0.0
    outperforming = sum(1 for r in sector_returns.values() if r > index_return)
    return float(outperforming / len(sector_returns))


def compute_approximate_contributions(
    sector_returns: dict[str, float],
    weights: dict[str, float],
    index_return: float,
) -> tuple[dict[str, float], float]:
    """Compute approximate contribution per sector and explicit residual.

    Formula:
        contrib_s = w_s * r_s
        residual = r_index - sum_s(contrib_s)

    Reconciliation invariant:
        sum_s(contrib_s) + residual == r_index (exact to 6 decimal places).
    """
    contributions: dict[str, float] = {}
    sum_contrib = 0.0
    for sid, r in sector_returns.items():
        w = float(weights.get(sid, 0.0))
        contrib = float(w * r)
        contributions[sid] = round(contrib, 6)
        sum_contrib += contrib

    residual = round(float(index_return - sum_contrib), 6)
    return contributions, residual


def compute_sector_rotation_trail(
    sector_price: pd.Series,
    index_price: pd.Series,
    window: int = 60,
    mom_window: int = 10,
    tail_days: int = 10,
) -> tuple[float, float, str, list[dict[str, Any]]]:
    """Compute relative strength ratio, momentum, quadrant phase, and trailing history."""
    aligned = pd.concat([sector_price, index_price], axis=1, keys=["sector", "index"]).dropna()
    if len(aligned) < window + mom_window:
        return 100.0, 100.0, "Leading", []

    rs_ratio = compute_relative_strength_ratio(aligned["sector"], aligned["index"], window=window)
    rs_mom = compute_ratio_momentum(rs_ratio, window=mom_window)

    valid_df = pd.concat([rs_ratio, rs_mom], axis=1, keys=["ratio", "mom"]).dropna()
    if valid_df.empty:
        return 100.0, 100.0, "Leading", []

    latest_ratio = float(valid_df["ratio"].iloc[-1])
    latest_mom = float(valid_df["mom"].iloc[-1])
    phase = classify_rotation_phase(latest_ratio, latest_mom)

    # Normalize phase name to editorial vocabulary:
    # "Fading" -> "Weakening", "Recovering" -> "Improving"
    phase_editorial = {
        "Leading": "Leading",
        "Fading": "Weakening",
        "Lagging": "Lagging",
        "Recovering": "Improving",
    }.get(phase, phase)

    trail: list[dict[str, Any]] = []
    tail_slice = valid_df.tail(tail_days)
    for dt, row in tail_slice.iterrows():
        date_str = str(dt.date()) if hasattr(dt, "date") else str(dt)[:10]
        trail.append({
            "date": date_str,
            "rs_ratio": round(float(row["ratio"]), 2),
            "rs_mom": round(float(row["mom"]), 2),
        })

    return round(latest_ratio, 2), round(latest_mom, 2), phase_editorial, trail


def analyze_region_sectors(
    region_id: str,
    region_cfg: dict[str, Any],
    prices_dict: dict[str, pd.Series],
    horizons: dict[str, int] | None = None,
    rotation_params: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Execute complete sector analytics for a region (US or India)."""
    if horizons is None:
        horizons = HORIZON_MAP
    if rotation_params is None:
        rotation_params = {"benchmark_window": 60, "momentum_window": 10, "tail_days": 10}

    index_id = region_cfg["index_id"]
    index_prices = prices_dict.get(index_id, pd.Series(dtype=float))
    weights: dict[str, float] = region_cfg.get("weights", {})

    # 1. Compute index returns for each horizon
    index_returns: dict[str, float] = {}
    for h_name, h_days in horizons.items():
        index_returns[h_name] = round(compute_cumulative_return(index_prices, h_days), 6)

    # 2. Compute sector-level metrics
    sector_items: list[dict[str, Any]] = []
    sector_returns_by_horizon: dict[str, dict[str, float]] = {h: {} for h in horizons}
    sector_contribs_by_horizon: dict[str, dict[str, float]] = {h: {} for h in horizons}

    for sid, weight in weights.items():
        s_prices = prices_dict.get(sid, pd.Series(dtype=float))
        s_name = SECTOR_NAMES.get(sid, sid.upper())

        s_returns: dict[str, float] = {}
        s_percentiles: dict[str, float] = {}
        for h_name, h_days in horizons.items():
            r = compute_cumulative_return(s_prices, h_days)
            pct = compute_return_series_percentile(s_prices, h_days)
            s_returns[h_name] = round(r, 6)
            s_percentiles[h_name] = round(pct, 1)
            sector_returns_by_horizon[h_name][sid] = round(r, 6)

        # Rotation & trail
        rs_ratio, rs_mom, phase, trail = compute_sector_rotation_trail(
            sector_price=s_prices,
            index_price=index_prices,
            window=rotation_params["benchmark_window"],
            mom_window=rotation_params["momentum_window"],
            tail_days=rotation_params["tail_days"],
        )

        sector_items.append({
            "id": sid,
            "name": s_name,
            "weight": round(weight, 4),
            "returns": s_returns,
            "percentiles": s_percentiles,
            "contributions": {},  # Populated after contributions calculation
            "rs_ratio": rs_ratio,
            "rs_mom": rs_mom,
            "phase": phase,
            "trail": trail,
        })

    # 3. Contributions, residuals, and breadth for each horizon
    residuals: dict[str, float] = {}
    sector_breadth: dict[str, float] = {}

    for h_name in horizons:
        contribs, res = compute_approximate_contributions(
            sector_returns=sector_returns_by_horizon[h_name],
            weights=weights,
            index_return=index_returns[h_name],
        )
        sector_contribs_by_horizon[h_name] = contribs
        residuals[h_name] = res
        sector_breadth[h_name] = round(
            compute_sector_breadth(sector_returns_by_horizon[h_name], index_returns[h_name]),
            4,
        )

    # Attach contributions to sector_items
    for item in sector_items:
        sid = item["id"]
        item["contributions"] = {
            h_name: sector_contribs_by_horizon[h_name].get(sid, 0.0)
            for h_name in horizons
        }

    return {
        "region_id": region_id,
        "country_name": region_cfg.get("country_name", region_id.upper()),
        "benchmark_name": region_cfg.get("benchmark_name", index_id.upper()),
        "index_id": index_id,
        "as_of": region_cfg.get("as_of", "2024-10-31"),
        "weights_source": region_cfg.get("source", "Published Index Factsheet"),
        "weights_source_url": region_cfg.get("source_url", ""),
        "index_returns": index_returns,
        "sector_breadth": sector_breadth,
        "residuals": residuals,
        "sectors": sector_items,
    }
