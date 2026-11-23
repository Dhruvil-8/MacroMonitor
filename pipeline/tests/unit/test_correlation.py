# pipeline/tests/unit/test_correlation.py
import math
import numpy as np
import pandas as pd
import pytest

from radar.metrics.correlation import (
    compute_correlation_shift,
    compute_heatmap_matrix,
    compute_rolling_correlation,
)


def test_correlation_known_answer():
    # Two perfectly positively correlated series (r = +1.0)
    # Two perfectly negatively correlated series (r = -1.0)
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    s_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=dates)
    s_b = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0], index=dates)
    s_c = pd.Series([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0], index=dates)

    corr_pos = compute_rolling_correlation(s_a, s_b, window=5)
    corr_neg = compute_rolling_correlation(s_a, s_c, window=5)

    assert math.isclose(corr_pos.iloc[-1], 1.0, rel_tol=1e-6)
    assert math.isclose(corr_neg.iloc[-1], -1.0, rel_tol=1e-6)


def test_correlation_shift():
    # rho_60 = 0.8, rho_250 = 0.2 -> shift = 0.8 - 0.2 = +0.6
    rho_60 = pd.Series([0.8], index=pd.date_range("2026-01-01", periods=1))
    rho_250 = pd.Series([0.2], index=pd.date_range("2026-01-01", periods=1))

    shift = compute_correlation_shift(rho_60, rho_250)
    assert math.isclose(shift.iloc[0], 0.6, rel_tol=1e-6)


def test_heatmap_matrix_bounds():
    # Pearson matrix must be symmetric, diagonal = 1.0, all values in [-1, +1]
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "spx": np.random.randn(100),
            "us10y": np.random.randn(100),
            "gold": np.random.randn(100),
        },
        index=dates,
    )

    matrix = compute_heatmap_matrix(df)
    assert matrix.shape == (3, 3)
    # Diagonal must be 1.0
    for col in matrix.columns:
        assert math.isclose(matrix.loc[col, col], 1.0, abs_tol=1e-6)

    # All values in [-1, 1]
    assert (matrix.to_numpy() >= -1.0 - 1e-6).all()
    assert (matrix.to_numpy() <= 1.0 + 1e-6).all()
