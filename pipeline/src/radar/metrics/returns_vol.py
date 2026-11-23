# pipeline/src/radar/metrics/returns_vol.py
"""Returns and realized volatility metrics.

Implements docs/spec/s01-returns-vol.md and Section 8.1.
Pure functions: vectorized, zero I/O, no globals.
"""

import math
import numpy as np
import pandas as pd


def compute_returns(s: pd.Series, method: str = "log_return") -> pd.Series:
    """Compute returns or differences from a price/yield series.

    Parameters
    ----------
    s : pd.Series
        Price or yield series.
    method : str
        'log_return' -> ln(P_t / P_{t-1})
        'diff' -> (y_t - y_{t-1}) * 100 (in basis points)

    Returns
    -------
    pd.Series
    """
    if method == "log_return":
        # Safe log return
        ratio = s / s.shift(1)
        # Avoid log(0) or negative
        ratio = ratio.where(ratio > 0, np.nan)
        return np.log(ratio)
    elif method == "diff":
        # Rate/yield difference in basis points
        return (s - s.shift(1)) * 100.0
    else:
        raise ValueError(f"Unknown return method: {method}")


def compute_realized_volatility(
    returns: pd.Series,
    window: int = 20,
    annualization_factor: float = math.sqrt(252),
) -> pd.Series:
    """Compute rolling annualized realized volatility.

    Parameters
    ----------
    returns : pd.Series
        Stationary return series.
    window : int
        Lookback window in days (e.g. 10, 20, 60).
    annualization_factor : float
        Square root of business days per year, default sqrt(252).

    Returns
    -------
    pd.Series
    """
    rolling_std = returns.rolling(window=window, min_periods=window).std(ddof=1)
    return rolling_std * annualization_factor


def compute_vol_trend(vol20: pd.Series, vol60: pd.Series) -> pd.Series:
    """Compute volatility trend ratio: vol20 / vol60 - 1.

    Returns
    -------
    pd.Series
    """
    safe_vol60 = vol60.where(vol60 > 0, np.nan)
    return (vol20 / safe_vol60) - 1.0
