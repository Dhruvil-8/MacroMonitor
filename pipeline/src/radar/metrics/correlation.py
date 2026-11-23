# pipeline/src/radar/metrics/correlation.py
"""Correlation metrics and heatmap generator.

Implements Section 8.4 and docs/spec/s07-cross-asset-correlation-rotation.md.
Pure functions: vectorized, zero I/O.
"""

import numpy as np
import pandas as pd


def compute_rolling_correlation(
    s_a: pd.Series, s_b: pd.Series, window: int = 60
) -> pd.Series:
    """Compute rolling Pearson correlation between two return series."""
    return s_a.rolling(window=window, min_periods=window).corr(s_b)


def compute_correlation_shift(rho_short: pd.Series, rho_long: pd.Series) -> pd.Series:
    """Compute correlation shift: Delta rho = rho_short - rho_long."""
    return rho_short - rho_long


def compute_heatmap_matrix(df_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Pearson correlation matrix for returns dataframe."""
    return df_returns.corr(method="pearson").round(4)
