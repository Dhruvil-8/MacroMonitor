# audit/oracles/oracle_returns_vol_drawdown.py
"""Independent clean-room oracle for Returns, Volatility, and Drawdown (s01, s02)."""

import math
import numpy as np
import pandas as pd

def oracle_log_return(prices: list[float] | np.ndarray) -> np.ndarray:
    """r_t = ln(P_t / P_{t-1})"""
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 2:
        return np.array([])
    return np.log(arr[1:] / arr[:-1])

def oracle_diff_bps(rates: list[float] | np.ndarray) -> np.ndarray:
    """r_t = (y_t - y_{t-1}) * 100 in basis points."""
    arr = np.asarray(rates, dtype=float)
    if len(arr) < 2:
        return np.array([])
    return (arr[1:] - arr[:-1]) * 100.0

def oracle_realized_volatility(returns: list[float] | np.ndarray, window: int) -> np.ndarray:
    """sigma_w = std(r over w, ddof=1) * sqrt(252)."""
    arr = np.asarray(returns, dtype=float)
    n = len(arr)
    if n < window:
        return np.full(n, np.nan)
    
    out = np.full(n, np.nan)
    scale = math.sqrt(252.0)
    for i in range(window - 1, n):
        sub = arr[i - window + 1 : i + 1]
        mean = sum(sub) / window
        var = sum((x - mean) ** 2 for x in sub) / (window - 1)
        out[i] = math.sqrt(var) * scale
    return out

def oracle_vol_trend(vol20: np.ndarray, vol60: np.ndarray) -> np.ndarray:
    """trend = vol20 / vol60 - 1."""
    return np.where((vol60 > 0) & ~np.isnan(vol60) & ~np.isnan(vol20), vol20 / vol60 - 1.0, np.nan)

def oracle_rolling_drawdown(prices: list[float] | np.ndarray, window: int = 252) -> np.ndarray:
    """Drawdown = P_t / max(P over trailing window) - 1 <= 0."""
    arr = np.asarray(prices, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window + 1)
        peak = max(arr[start : i + 1])
        out[i] = (arr[i] / peak) - 1.0
    return out
