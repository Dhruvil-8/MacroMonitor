# pipeline/src/radar/metrics/country.py
"""Country layer quantitative metrics (Section 4.1 of docs/MASTER_PLAN.md).

Computes performance matrices across Local, USD, and FX lenses,
country-level breadth (% above 200D MA), average pairwise correlation,
and relative strength versus Rest-of-World.
"""

from __future__ import annotations

import logging
from typing import Any, Literal
import numpy as np
import pandas as pd

from radar.metrics.decomposition import compute_fx_identity

LOGGER = logging.getLogger(__name__)

LAUNCH_COUNTRIES = [
    {"code": "us", "name": "United States", "index_id": "spx", "fx_id": None, "quote_type": "base"},
    {"code": "ca", "name": "Canada", "index_id": "tsse", "fx_id": "usdcad", "quote_type": "usd_per_foreign_inv"},
    {"code": "br", "name": "Brazil", "index_id": "ibov", "fx_id": "usdbrl", "quote_type": "usd_per_foreign_inv"},
    {"code": "gb", "name": "United Kingdom", "index_id": "ftse", "fx_id": "gbpusd", "quote_type": "foreign_per_usd"},
    {"code": "de", "name": "Germany", "index_id": "dax", "fx_id": "eurusd", "quote_type": "foreign_per_usd"},
    {"code": "jp", "name": "Japan", "index_id": "n225", "fx_id": "usdjpy", "quote_type": "usd_per_foreign_inv"},
    {"code": "cn", "name": "China", "index_id": "ssec", "fx_id": "usdcny", "quote_type": "usd_per_foreign_inv"},
    {"code": "in", "name": "India", "index_id": "nifty50", "fx_id": "usdinr", "quote_type": "usd_per_foreign_inv"},
    {"code": "kr", "name": "South Korea", "index_id": "kospi", "fx_id": "usdkrw", "quote_type": "usd_per_foreign_inv"},
    {"code": "au", "name": "Australia", "index_id": "asx200", "fx_id": "audusd", "quote_type": "foreign_per_usd"},
    {"code": "tw", "name": "Taiwan", "index_id": "twii", "fx_id": "usdtwd", "quote_type": "usd_per_foreign_inv"},
    {"code": "ch", "name": "Switzerland", "index_id": "smi", "fx_id": "usdchf", "quote_type": "usd_per_foreign_inv"},
    {"code": "fr", "name": "France", "index_id": "cac40", "fx_id": "eurusd", "quote_type": "foreign_per_usd"},
    {"code": "mx", "name": "Mexico", "index_id": "ipc", "fx_id": "usdmxn", "quote_type": "usd_per_foreign_inv"},
    {"code": "id", "name": "Indonesia", "index_id": "jkse", "fx_id": "usdidr", "quote_type": "usd_per_foreign_inv"},
    {"code": "sa", "name": "Saudi Arabia", "index_id": "ksa", "fx_id": None, "quote_type": "base"},
]


def compute_cumulative_return(prices: pd.Series, window: int) -> float:
    """Compute cumulative return over window days."""
    s = prices.dropna()
    if len(s) < window + 1:
        return 0.0
    return float(s.iloc[-1] / s.iloc[-(window + 1)] - 1.0)


def compute_normalized_fx_return(fx_prices: pd.Series, window: int, quote_type: str) -> float:
    """Compute FX return normalized so positive return means foreign currency strengthened."""
    s = fx_prices.dropna()
    if len(s) < window + 1:
        return 0.0
    raw_return = float(s.iloc[-1] / s.iloc[-(window + 1)] - 1.0)
    if quote_type == "usd_per_foreign_inv":
        # Quote is USD/Foreign (e.g. USD/JPY 150 -> 155 means JPY weakened)
        # Value of 1 Foreign currency in USD is 1/S
        # Return = (1/S_t) / (1/S_{t-w}) - 1 = S_{t-w}/S_t - 1
        return float(s.iloc[-(window + 1)] / s.iloc[-1] - 1.0)
    # Direct quote (EUR/USD, GBP/USD, AUD/USD where up means local strengthened)
    return raw_return


def compute_country_breadth(country_prices_df: pd.DataFrame, window: int = 200) -> float:
    """Compute share of tracked countries trading above their 200-day moving average.

    Returns float in [0.0, 1.0].
    """
    if country_prices_df.empty:
        return 0.0
    above_count = 0
    valid_count = 0
    for col in country_prices_df.columns:
        s = country_prices_df[col].dropna()
        if len(s) >= window:
            ma200 = s.rolling(window).mean().iloc[-1]
            latest = s.iloc[-1]
            if latest > ma200:
                above_count += 1
            valid_count += 1
    if valid_count == 0:
        return 0.0
    return float(above_count / valid_count)


def compute_average_pairwise_correlation(
    country_returns_df: pd.DataFrame,
    window: int = 60,
) -> float:
    """Compute average pairwise correlation across countries over rolling window."""
    clean_rets = country_returns_df.dropna().tail(window)
    if len(clean_rets) < 20 or clean_rets.shape[1] < 2:
        return 0.0
    corr_matrix = clean_rets.corr().values
    n = corr_matrix.shape[0]
    # Extract strictly upper triangle
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    valid_tri = upper_tri[~np.isnan(upper_tri)]
    if len(valid_tri) == 0:
        return 0.0
    return float(np.mean(valid_tri))


def compute_country_metrics(
    prices_dict: dict[str, pd.Series],
    horizons: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute performance matrix for 10 launch countries satisfying FX identity.

    Horizons default: {'1D': 1, '5D': 5, '1M': 21, '3M': 63, '1Y': 252}
    """
    if horizons is None:
        horizons = {"1D": 1, "5D": 5, "1M": 21, "3M": 63, "1Y": 252}

    matrix_rows = []
    country_index_series = {}

    for c in LAUNCH_COUNTRIES:
        code = c["code"]
        name = c["name"]
        idx_id = c["index_id"]
        fx_id = c["fx_id"]
        q_type = c["quote_type"]

        p_index = prices_dict.get(idx_id)
        if p_index is None or p_index.empty:
            continue

        country_index_series[code] = p_index
        p_fx = prices_dict.get(fx_id) if fx_id else None

        row_data: dict[str, Any] = {
            "code": code,
            "name": name,
            "index_id": idx_id,
            "currency": "USD" if not fx_id else fx_id.upper()[-3:],
            "returns_local": {},
            "returns_usd": {},
            "returns_fx": {},
        }

        for h_label, h_days in horizons.items():
            r_loc = compute_cumulative_return(p_index, h_days)
            if fx_id and p_fx is not None and not p_fx.empty:
                r_fx = compute_normalized_fx_return(p_fx, h_days, q_type)
                _, r_loc, r_fx = compute_fx_identity(r_local=r_loc, r_fx=r_fx)
            else:
                r_fx = 0.0

            r_loc_clean = round(r_loc, 4)
            r_fx_clean = round(r_fx, 4)
            # Enforce exact identity on published numbers so (1 + r_usd) == (1 + r_loc)(1 + r_fx)
            r_usd_clean = round((1.0 + r_loc_clean) * (1.0 + r_fx_clean) - 1.0, 4)

            row_data["returns_local"][h_label] = r_loc_clean
            row_data["returns_usd"][h_label] = r_usd_clean
            row_data["returns_fx"][h_label] = r_fx_clean

        matrix_rows.append(row_data)

    # Compute breadth and average correlation
    all_indices_df = pd.DataFrame(country_index_series).dropna()
    breadth_200d = compute_country_breadth(all_indices_df, window=min(200, len(all_indices_df)))
    
    country_rets = np.log(all_indices_df / all_indices_df.shift(1)).dropna()
    avg_corr_60d = compute_average_pairwise_correlation(country_rets, window=min(60, len(country_rets)))
    avg_corr_250d = compute_average_pairwise_correlation(country_rets, window=min(250, len(country_rets)))

    return {
        "countries": matrix_rows,
        "breadth_200d": round(breadth_200d, 3),
        "avg_corr_60d": round(avg_corr_60d, 3),
        "avg_corr_250d": round(avg_corr_250d, 3),
        "tracked_count": len(matrix_rows),
    }
