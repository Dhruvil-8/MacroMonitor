# pipeline/src/radar/metrics/divergence.py
"""Divergence detection between pre-registered asset pairs.

Implements Section 8.5 and docs/spec/s06-insight-engine-divergence.md.
Pure functions: vectorized, zero I/O.
"""

from typing import Any

import numpy as np
import pandas as pd


def compute_divergence_series(z_a: pd.Series, z_b: pd.Series) -> pd.Series:
    """Compute divergence spread D_t = Z(mom20_A) - Z(mom20_B)."""
    return z_a - z_b


def detect_divergences(
    d_series: pd.Series, threshold: float = 2.0, consecutive_days: int = 3
) -> pd.Series:
    """Flag divergence when |D| exceeds threshold for consecutive_days.

    Returns boolean Series where True indicates active persistent divergence.
    """
    exceeded = d_series.abs() >= threshold
    # Count consecutive True days
    consecutive_count = (
        exceeded.astype(int)
        .groupby((~exceeded).cumsum())
        .cumsum()
    )
    return consecutive_count >= consecutive_days


def compute_cusum_series(
    series: pd.Series,
    slack_k: float = 0.5,
    threshold_h: float = 4.0,
) -> dict[str, Any]:
    """Compute two-sided CUSUM control chart (C24) on standardized innovations.

    S_t^+ = max(0, S_{t-1}^+ + z_t - k)
    S_t^- = max(0, S_{t-1}^- - z_t - k)
    """
    s = series.dropna()
    if len(s) < 10:
        return {"dates": [], "cusum_pos": [], "cusum_neg": [], "shifts": [], "threshold": threshold_h}

    mean_val = float(s.mean())
    std_val = float(s.std(ddof=1))
    if std_val <= 0:
        std_val = 1.0

    z = (s - mean_val) / std_val

    n = len(z)
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    shifts: list[dict[str, Any]] = []

    for t in range(1, n):
        s_pos[t] = max(0.0, s_pos[t - 1] + z.iloc[t] - slack_k)
        s_neg[t] = max(0.0, s_neg[t - 1] - z.iloc[t] - slack_k)

        if s_pos[t] >= threshold_h:
            shifts.append({
                "date": str(pd.to_datetime(z.index[t]).date()),
                "direction": "upward",
                "value": round(float(s_pos[t]), 2),
            })
            s_pos[t] = 0.0

        if s_neg[t] >= threshold_h:
            shifts.append({
                "date": str(pd.to_datetime(z.index[t]).date()),
                "direction": "downward",
                "value": round(float(s_neg[t]), 2),
            })
            s_neg[t] = 0.0

    dates = [str(pd.to_datetime(dt).date()) for dt in z.index]

    return {
        "dates": dates,
        "cusum_pos": [round(float(v), 2) for v in s_pos],
        "cusum_neg": [round(float(v), 2) for v in s_neg],
        "shifts": shifts,
        "threshold": threshold_h,
    }
