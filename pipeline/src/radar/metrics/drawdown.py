# pipeline/src/radar/metrics/drawdown.py
"""Drawdown metric.

Implements docs/spec/s02-drawdown.md and Section 8.1.
Pure function: vectorized, zero I/O, no globals.
"""

import numpy as np
import pandas as pd


def compute_drawdown(prices: pd.Series, window: int = 252) -> pd.Series:
    """Compute rolling peak-to-trough drawdown over lookback window.

    Formula:
        Peak_t = max(prices[t-w+1 ... t])
        DD_t = prices_t / Peak_t - 1

    Parameters
    ----------
    prices : pd.Series
        Price series.
    window : int
        Trailing rolling window in days (default 252).

    Returns
    -------
    pd.Series
        Drawdown decimal series <= 0.
    """
    rolling_peak = prices.rolling(window=window, min_periods=1).max()
    safe_peak = rolling_peak.where(rolling_peak > 0, np.nan)
    dd = (prices / safe_peak) - 1.0
    # Clip numerical epsilon noise above 0
    return dd.clip(upper=0.0)
