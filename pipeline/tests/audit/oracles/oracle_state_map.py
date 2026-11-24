# audit/oracles/oracle_state_map.py
"""Independent clean-room oracle for Market State Map metrics (s05)."""

import math
import numpy as np

def oracle_trend_t_stat(log_prices: list[float] | np.ndarray, window: int = 60) -> float:
    """Signed t-statistic of the OLS slope of log price on time over trailing window.
    
    Model: y_i = alpha + beta * i + epsilon_i for i = 0 .. window-1
    t = beta / se(beta)
    """
    arr = np.asarray(log_prices, dtype=float)
    if len(arr) < window:
        return 0.0
    y = arr[-window:]
    x = np.arange(window, dtype=float)
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    s_xx = np.sum((x - x_mean) ** 2)
    s_xy = np.sum((x - x_mean) * (y - y_mean))
    
    if s_xx < 1e-12:
        return 0.0
    
    beta = s_xy / s_xx
    alpha = y_mean - beta * x_mean
    
    residuals = y - (alpha + beta * x)
    dof = window - 2
    s_err2 = np.sum(residuals ** 2) / dof
    se_beta = math.sqrt(s_err2 / s_xx)
    
    if se_beta < 1e-12:
        return 0.0
    return float(beta / se_beta)

def oracle_state_quadrant(
    t_stat: float,
    vol_percentile: float,
    trend_threshold: float = 2.0,
    vol_midpoint: float = 50.0,
) -> str:
    """Classify regime into 4 quadrants:
    - |t| >= 2.0, Vol >= 50%: Violent trend
    - |t| >= 2.0, Vol < 50%: Orderly trend
    - |t| < 2.0, Vol >= 50%: Choppy
    - |t| < 2.0, Vol < 50%: Quiet range
    """
    is_trend = abs(t_stat) >= trend_threshold
    is_high_vol = vol_percentile >= vol_midpoint
    
    if is_trend and is_high_vol:
        return "Violent trend"
    elif is_trend and not is_high_vol:
        return "Orderly trend"
    elif not is_trend and is_high_vol:
        return "Choppy"
    else:
        return "Quiet range"
