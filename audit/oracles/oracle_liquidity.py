# audit/oracles/oracle_liquidity.py
"""Independent clean-room oracle for Net Liquidity and Funding Dynamics (s09)."""

import numpy as np
import pandas as pd

def oracle_net_liquidity(walcl_mn: float, wtregen_mn: float, rrpontsyd_bn: float) -> float:
    """Net Liquidity = (WALCL / 1000) - (WTREGEN / 1000) - RRPONTSYD in Billions USD."""
    a = walcl_mn / 1000.0
    t = wtregen_mn / 1000.0
    r = rrpontsyd_bn
    return float(a - t - r)

def oracle_liquidity_changes(
    net_liq_series: np.ndarray,
    w4: int = 20,
    w13: int = 65
) -> tuple[np.ndarray, np.ndarray]:
    """Delta_4W = L_t - L_{t-20}, Delta_13W = L_t - L_{t-65}."""
    n = len(net_liq_series)
    d4 = np.full(n, np.nan)
    d13 = np.full(n, np.nan)
    for i in range(w4, n):
        d4[i] = net_liq_series[i] - net_liq_series[i - w4]
    for i in range(w13, n):
        d13[i] = net_liq_series[i] - net_liq_series[i - w13]
    return d4, d13
