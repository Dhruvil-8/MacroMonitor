# pipeline/tests/unit/test_returns_vol.py
import math
import numpy as np
import pandas as pd
import pytest

from radar.metrics.returns_vol import compute_returns, compute_realized_volatility, compute_vol_trend


def test_log_returns_known_answer():
    # Hand calculation:
    # P_0 = 100.0, P_1 = 110.0 -> r_1 = ln(110/100) = ln(1.1) = 0.0953101798
    # P_2 = 104.5 -> r_2 = ln(104.5/110) = ln(0.95) = -0.05129329438
    prices = pd.Series([100.0, 110.0, 104.5], index=pd.date_range("2026-01-01", periods=3, freq="B"))
    ret = compute_returns(prices, method="log_return")

    assert np.isnan(ret.iloc[0])
    assert math.isclose(ret.iloc[1], 0.0953101798, rel_tol=1e-6)
    assert math.isclose(ret.iloc[2], -0.05129329438, rel_tol=1e-6)


def test_diff_bps_known_answer():
    # Hand calculation:
    # y_0 = 4.25%, y_1 = 4.35% -> diff = +0.10% = +10.0 bps
    # y_2 = 4.30% -> diff = -0.05% = -5.0 bps
    yields = pd.Series([4.25, 4.35, 4.30], index=pd.date_range("2026-01-01", periods=3, freq="B"))
    diff_bps = compute_returns(yields, method="diff")

    assert np.isnan(diff_bps.iloc[0])
    assert math.isclose(diff_bps.iloc[1], 10.0, rel_tol=1e-6)
    assert math.isclose(diff_bps.iloc[2], -5.0, rel_tol=1e-6)


def test_invalid_return_method():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        compute_returns(s, method="invalid_method")


def test_realized_vol_known_answer():
    # Returns: [0.01, -0.01, 0.01, -0.01] over window 4
    # mean = 0
    # deviations: 0.01, -0.01, 0.01, -0.01
    # sum of squares = 4 * 0.0001 = 0.0004
    # variance (ddof=1) = 0.0004 / 3 = 0.00013333333333
    # std = sqrt(0.00013333333333) = 0.01154700538
    # annualized = std * sqrt(252) = 0.01154700538 * 15.874507866 = 0.1833030278
    returns = pd.Series([0.01, -0.01, 0.01, -0.01], index=pd.date_range("2026-01-01", periods=4, freq="B"))
    vol = compute_realized_volatility(returns, window=4, annualization_factor=math.sqrt(252))

    assert math.isclose(vol.iloc[3], 0.1833030278, rel_tol=1e-6)


def test_anti_lookahead_guarantee_volatility():
    # Verify that future price changes at t=100 do NOT alter volatility computed at t <= 50
    np.random.seed(123)
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    ret_history = pd.Series(np.random.normal(0, 0.01, 100), index=dates)

    vol_original = compute_realized_volatility(ret_history, window=20)

    # Corrupt future returns after t=50
    ret_corrupted = ret_history.copy()
    ret_corrupted.iloc[50:] = ret_corrupted.iloc[50:] * 10.0
    vol_corrupted = compute_realized_volatility(ret_corrupted, window=20)

    # Check that past values up to t=49 are IDENTICAL
    pd.testing.assert_series_equal(vol_original.iloc[:50], vol_corrupted.iloc[:50])


def test_vol_trend():
    vol20 = pd.Series([0.20, 0.25], index=pd.date_range("2026-01-01", periods=2, freq="B"))
    vol60 = pd.Series([0.20, 0.20], index=pd.date_range("2026-01-01", periods=2, freq="B"))
    trend = compute_vol_trend(vol20, vol60)

    # 0.20 / 0.20 - 1 = 0.0
    # 0.25 / 0.20 - 1 = +0.25 (+25%)
    assert math.isclose(trend.iloc[0], 0.0, rel_tol=1e-6)
    assert math.isclose(trend.iloc[1], 0.25, rel_tol=1e-6)
