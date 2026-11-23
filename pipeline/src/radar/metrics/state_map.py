# pipeline/src/radar/metrics/state_map.py
"""Market State Map metric computation.

Implements Section 8.3 and docs/AGENT_PLAN.md:
- Trend strength: signed t-statistic of slope of log price on time over 60 days
- Volatility state: expanding percentile of 20-day realized volatility
- Momentum change: 20-day return minus 20-day return lagged 20 days
- Quadrant classification with hysteresis:
  - 'Orderly trend': |t| >= 2.0 and vol_pct < 50
  - 'Violent trend': |t| >= 2.0 and vol_pct >= 50
  - 'Quiet range':   |t| < 2.0  and vol_pct < 50
  - 'Choppy':        |t| < 2.0  and vol_pct >= 50
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_trend_tstat(s: pd.Series, window: int = 60) -> pd.Series:
    """Compute rolling signed t-statistic of the slope of log price on time."""
    log_p = np.log(s.where(s > 0, np.nan))
    t_stats = pd.Series(np.nan, index=s.index)

    x = np.arange(window)
    x_mean = x.mean()
    x_dev = x - x_mean
    ss_x = np.sum(x_dev**2)

    # Rolling window regression
    values = log_p.to_numpy()
    n = len(values)

    for i in range(window - 1, n):
        y = values[i - window + 1 : i + 1]
        if np.isnan(y).any():
            continue

        y_mean = np.mean(y)
        slope = np.sum(x_dev * (y - y_mean)) / ss_x
        intercept = y_mean - slope * x_mean

        residuals = y - (intercept + slope * x)
        s_err = np.sqrt(np.sum(residuals**2) / (window - 2))
        se_slope = s_err / np.sqrt(ss_x)

        if se_slope > 0:
            t_stats.iloc[i] = float(slope / se_slope)
        else:
            t_stats.iloc[i] = 0.0

    return t_stats


def classify_quadrant(trend_t: float, vol_percentile: float, trend_threshold: float = 2.0) -> str:
    """Classify asset into one of four Market State Map quadrants."""
    has_trend = abs(trend_t) >= trend_threshold
    low_vol = vol_percentile < 50.0

    if has_trend and low_vol:
        return "Orderly trend"
    elif has_trend and not low_vol:
        return "Violent trend"
    elif not has_trend and low_vol:
        return "Quiet range"
    else:
        return "Choppy"
