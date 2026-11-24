# audit/oracles/oracle_correlation_divergence_rotation.py
"""Independent clean-room oracle for Correlation, Divergence, and Rotation (s06, s07)."""

import math
import numpy as np
import pandas as pd

def oracle_rolling_correlation(
    series_a: list[float] | np.ndarray,
    series_b: list[float] | np.ndarray,
    window: int = 60
) -> np.ndarray:
    """Pearson correlation over rolling window."""
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    n = min(len(a), len(b))
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        sub_a = a[i - window + 1 : i + 1]
        sub_b = b[i - window + 1 : i + 1]
        m_a = np.mean(sub_a)
        m_b = np.mean(sub_b)
        num = np.sum((sub_a - m_a) * (sub_b - m_b))
        den = math.sqrt(np.sum((sub_a - m_a) ** 2) * np.sum((sub_b - m_b) ** 2))
        out[i] = (num / den) if den > 1e-12 else 0.0
    return out

def oracle_divergence_spread(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    mom_window: int = 20,
    min_obs: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """mom_20 = P_t / P_{t-20} - 1
    D_t = Z(mom20_A) - Z(mom20_B) using expanding Z excluding t.
    """
    n = min(len(prices_a), len(prices_b))
    mom_a = np.full(n, np.nan)
    mom_b = np.full(n, np.nan)
    for i in range(mom_window, n):
        mom_a[i] = (prices_a[i] / prices_a[i - mom_window]) - 1.0
        mom_b[i] = (prices_b[i] / prices_b[i - mom_window]) - 1.0
        
    z_a = np.full(n, np.nan)
    z_b = np.full(n, np.nan)
    d = np.full(n, np.nan)
    for t in range(min_obs + mom_window, n):
        hist_a = mom_a[mom_window : t]
        hist_b = mom_b[mom_window : t]
        ma = np.mean(hist_a)
        sa = np.std(hist_a, ddof=1)
        mb = np.mean(hist_b)
        sb = np.std(hist_b, ddof=1)
        
        za = (mom_a[t] - ma) / sa if sa > 1e-12 else 0.0
        zb = (mom_b[t] - mb) / sb if sb > 1e-12 else 0.0
        z_a[t] = za
        z_b[t] = zb
        d[t] = za - zb
    return d, np.abs(d)

def oracle_rotation_coordinates(
    asset_prices: np.ndarray,
    benchmark_prices: np.ndarray,
    ma_window: int = 60,
    mom_window: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """RS_t = P_asset / P_bmk
    RS_Ratio_t = 100 * (1 + (RS_t - MA60(RS)) / MA60(RS))
    RS_Mom_t = 100 * (1 + (RS_Ratio_t - RS_Ratio_{t-10}) / RS_Ratio_{t-10})
    """
    n = min(len(asset_prices), len(benchmark_prices))
    rs = asset_prices[:n] / benchmark_prices[:n]
    
    rs_ratio = np.full(n, np.nan)
    for i in range(ma_window - 1, n):
        sub = rs[i - ma_window + 1 : i + 1]
        m = np.mean(sub)
        if m > 1e-12:
            rs_ratio[i] = 100.0 * (1.0 + (rs[i] - m) / m)
            
    rs_mom = np.full(n, np.nan)
    for i in range(ma_window - 1 + mom_window, n):
        prev = rs_ratio[i - mom_window]
        curr = rs_ratio[i]
        if not np.isnan(prev) and prev > 1e-12:
            rs_mom[i] = 100.0 * (1.0 + (curr - prev) / prev)
            
    return rs_ratio, rs_mom
