# pipeline/tests/unit/test_drawdown.py
import math
import numpy as np
import pandas as pd
import pytest

from radar.metrics.drawdown import compute_drawdown


def test_drawdown_known_answer():
    # Hand calculation:
    # Prices: [100.0, 120.0, 90.0, 110.0, 80.0]
    # Rolling Peak (window 5):
    # t=0: Peak = 100 -> DD = 100/100 - 1 = 0.0
    # t=1: Peak = 120 -> DD = 120/120 - 1 = 0.0
    # t=2: Peak = 120 -> DD = 90/120 - 1 = -0.25 (-25%)
    # t=3: Peak = 120 -> DD = 110/120 - 1 = -0.08333333
    # t=4: Peak = 120 -> DD = 80/120 - 1 = -0.33333333
    prices = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0], index=pd.date_range("2026-01-01", periods=5, freq="B"))
    dd = compute_drawdown(prices, window=5)

    assert math.isclose(dd.iloc[0], 0.0, abs_tol=1e-6)
    assert math.isclose(dd.iloc[1], 0.0, abs_tol=1e-6)
    assert math.isclose(dd.iloc[2], -0.25, rel_tol=1e-6)
    assert math.isclose(dd.iloc[3], -1.0 / 12.0, rel_tol=1e-6)
    assert math.isclose(dd.iloc[4], -1.0 / 3.0, rel_tol=1e-6)


def test_drawdown_invariants():
    # Drawdown must always be <= 0 and >= -1.0
    np.random.seed(99)
    prices = pd.Series(np.random.uniform(50, 150, 200), index=pd.date_range("2025-01-01", periods=200, freq="B"))
    dd = compute_drawdown(prices, window=60)

    assert (dd <= 0.0).all()
    assert (dd >= -1.0).all()


def test_drawdown_lookahead_guarantee():
    # Future prices at t=50..100 must not alter drawdown at t < 50
    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    prices_orig = pd.Series(np.linspace(100, 200, 100), index=dates)

    dd_orig = compute_drawdown(prices_orig, window=20)

    # Crash price in future
    prices_corrupt = prices_orig.copy()
    prices_corrupt.iloc[50:] = 10.0
    dd_corrupt = compute_drawdown(prices_corrupt, window=20)

    pd.testing.assert_series_equal(dd_orig.iloc[:50], dd_corrupt.iloc[:50])
