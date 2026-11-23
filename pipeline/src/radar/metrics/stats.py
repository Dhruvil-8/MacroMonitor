# pipeline/src/radar/metrics/stats.py
"""Statistical context: effective sample size, expanding percentile, and expanding z-score.

Implements docs/spec/s03-percentile-zscore-neff.md and Section 8.2.
Pure functions: strictly anti-lookahead, vectorized, zero I/O.
"""

import bisect
import numpy as np
import pandas as pd


def compute_lag1_autocorr(s: pd.Series) -> float:
    """Compute lag-1 autocorrelation for a series."""
    clean = s.dropna()
    if len(clean) < 5:
        return 0.0
    corr = clean.autocorr(lag=1)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def compute_effective_n(n: int | float, rho1: float) -> float:
    """Compute effective sample size: N_eff = N * (1 - rho1) / (1 + rho1).

    Parameters
    ----------
    n : int or float
        Raw observation count.
    rho1 : float
        Lag-1 autocorrelation of the series, clipped to [0.0, 0.99].

    Returns
    -------
    float
    """
    clipped_rho = max(0.0, min(0.99, float(rho1)))
    return float(n * (1.0 - clipped_rho) / (1.0 + clipped_rho))


def compute_expanding_zscore(s: pd.Series, min_obs: int = 750) -> pd.Series:
    """Compute expanding z-score strictly excluding the current point.

    Formula:
        mu_{t-1} = mean(s[0 ... t-1])
        sigma_{t-1} = std(s[0 ... t-1], ddof=1)
        z_t = (s_t - mu_{t-1}) / sigma_{t-1}
    """
    # Shift series by 1 so expanding window uses strictly prior points [0 ... t-1]
    prior = s.shift(1)
    exp_mean = prior.expanding(min_periods=min_obs).mean()
    exp_std = prior.expanding(min_periods=min_obs).std(ddof=1)

    # Avoid divide-by-zero for constant series (std == 0)
    safe_std = exp_std.where(exp_std > 0, np.nan)
    z = (s - exp_mean) / safe_std

    # Where prior points had zero variance, define z as 0.0
    zero_variance_mask = (exp_std == 0) & exp_mean.notna()
    z = z.where(~zero_variance_mask, 0.0)
    return z


def compute_expanding_percentile(s: pd.Series, min_obs: int = 750) -> pd.Series:
    """Compute expanding percentile rank of s_t against past values in [0 ... t].

    Returns percentile in range [0.0, 100.0] or NaN if observation count < min_obs.
    """
    values = s.to_numpy()
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)

    valid_indices = np.where(~np.isnan(values))[0]
    if len(valid_indices) < min_obs:
        return pd.Series(out, index=s.index)

    history: list[float] = []

    for idx in valid_indices:
        val = float(values[idx])
        bisect.insort(history, val)
        count = len(history)
        if count >= min_obs:
            rank = bisect.bisect_right(history, val)
            out[idx] = (rank / count) * 100.0

    return pd.Series(out, index=s.index)
