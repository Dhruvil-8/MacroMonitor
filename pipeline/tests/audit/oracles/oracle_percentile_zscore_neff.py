# audit/oracles/oracle_percentile_zscore_neff.py
"""Independent clean-room oracle for Percentile, Z-Score, and Effective N (s03)."""

import math
import numpy as np

def oracle_expanding_percentile(series: list[float] | np.ndarray, min_obs: int = 750) -> np.ndarray:
    """Percentile of x_t among strictly prior observations s in [0, t-1] (or empirical CDF).
    
    Formula: Percentile_t = (sum_{s=0}^t 1_{x_s <= x_t}) / (t + 1) * 100
    If observations < min_obs, return NaN.
    """
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    for t in range(n):
        if (t + 1) < min_obs:
            continue
        history = arr[: t + 1]
        valid = history[~np.isnan(history)]
        if len(valid) < min_obs:
            continue
        curr = arr[t]
        count_less_equal = np.sum(valid <= curr)
        out[t] = (count_less_equal / len(valid)) * 100.0
    return out

def oracle_expanding_zscore_anti_lookahead(series: list[float] | np.ndarray, min_obs: int = 60) -> np.ndarray:
    """Z-score strictly excluding current point t using sample mean and sample std of past [0, t-1].
    
    mu_{t-1} = mean(x_{0..t-1})
    sigma_{t-1} = std(x_{0..t-1}, ddof=1)
    z_t = (x_t - mu_{t-1}) / sigma_{t-1}
    """
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    for t in range(n):
        if t < min_obs:
            continue
        past = arr[:t]
        valid_past = past[~np.isnan(past)]
        if len(valid_past) < min_obs:
            continue
        m = float(np.mean(valid_past))
        s = float(np.std(valid_past, ddof=1))
        if s > 1e-12:
            out[t] = (arr[t] - m) / s
        else:
            out[t] = 0.0
    return out

def oracle_effective_sample_size(series: list[float] | np.ndarray) -> tuple[int, float]:
    """N_eff = N * (1 - rho1) / (1 + rho1), rho1 clipped to [0.0, 0.99]."""
    arr = np.asarray(series, dtype=float)
    valid = arr[~np.isnan(arr)]
    N = len(valid)
    if N < 3:
        return N, float(N)
    
    # Lag-1 autocorrelation
    x0 = valid[:-1]
    x1 = valid[1:]
    m0 = np.mean(x0)
    m1 = np.mean(x1)
    num = np.sum((x0 - m0) * (x1 - m1))
    den = math.sqrt(np.sum((x0 - m0) ** 2) * np.sum((x1 - m1) ** 2))
    rho1 = float(num / den) if den > 1e-12 else 0.0
    
    # Clip rho1 to [0.0, 0.99]
    rho1_clipped = max(0.0, min(0.99, rho1))
    neff = N * (1.0 - rho1_clipped) / (1.0 + rho1_clipped)
    return N, float(neff)
