# pipeline/src/radar/metrics/rotation.py
"""Relative Rotation Map metrics.

Implements Section 8.6 and docs/spec/s07-cross-asset-correlation-rotation.md.
Pure functions: vectorized, zero I/O.
"""

import numpy as np
import pandas as pd


def compute_relative_strength_ratio(
    asset_price: pd.Series, benchmark_price: pd.Series, window: int = 60
) -> pd.Series:
    """Compute relative strength ratio normalized to 100 against its 60-day moving average.

    Formula:
        RS = asset_price / benchmark_price
        RS_Ratio = 100 * (1 + (RS - MA_60(RS)) / MA_60(RS))
    """
    rs = asset_price / benchmark_price.where(benchmark_price > 0, np.nan)
    rs_ma = rs.rolling(window=window, min_periods=window).mean()
    safe_ma = rs_ma.where(rs_ma > 0, np.nan)
    return 100.0 * (1.0 + (rs - safe_ma) / safe_ma)


def compute_ratio_momentum(rs_ratio: pd.Series, window: int = 10) -> pd.Series:
    """Compute 10-day rate of change of relative strength ratio.

    Formula:
        RS_Mom = 100 * (1 + (RS_Ratio - RS_Ratio_{t-10}) / RS_Ratio_{t-10})
    """
    lagged = rs_ratio.shift(window)
    safe_lag = lagged.where(lagged > 0, np.nan)
    return 100.0 * (1.0 + (rs_ratio - safe_lag) / safe_lag)


def classify_rotation_phase(rs_ratio: float, rs_mom: float) -> str:
    """Classify into one of 4 rotation cycle phases.

    Phases:
        - 'Leading':    RS >= 100 and Mom >= 100
        - 'Fading':     RS >= 100 and Mom < 100
        - 'Lagging':    RS < 100  and Mom < 100
        - 'Recovering': RS < 100  and Mom >= 100
    """
    high_rs = rs_ratio >= 100.0
    high_mom = rs_mom >= 100.0

    if high_rs and high_mom:
        return "Leading"
    elif high_rs and not high_mom:
        return "Fading"
    elif not high_rs and not high_mom:
        return "Lagging"
    else:
        return "Recovering"
