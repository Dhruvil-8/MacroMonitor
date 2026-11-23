"""Unit tests for P5: Kalman Filter Time-Varying Beta Estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.metrics.kalman import KalmanBetaFilter, compare_kalman_vs_rolling_ols


def test_kalman_constant_beta():
    """Verify filter converges to constant true beta in simulation."""
    np.random.seed(42)
    T = 200
    true_beta = 1.75
    x = np.random.normal(0, 1, size=(T, 1))
    noise = np.random.normal(0, 0.2, size=T)
    y = (x[:, 0] * true_beta) + noise

    kf = KalmanBetaFilter(q_ratio=1e-5, init_obs=50)
    res = kf.fit_filter(y, x)

    final_b = res["final_beta"][0]
    ci_lower, ci_upper = res["final_intervals"][0]

    # Must be within 10% of true value
    assert abs(final_b - true_beta) < 0.15
    # True value should be inside confidence interval
    assert ci_lower <= true_beta <= ci_upper


def test_kalman_random_walk_beta():
    """Verify filter tracks a drifting true beta."""
    np.random.seed(123)
    T = 300
    beta_path = np.zeros(T)
    beta_path[0] = 1.0
    for t in range(1, T):
        beta_path[t] = beta_path[t - 1] + np.random.normal(0, 0.03)

    x = np.random.normal(0, 1, size=(T, 1))
    noise = np.random.normal(0, 0.15, size=T)
    y = (x[:, 0] * beta_path) + noise

    kf = KalmanBetaFilter(q_ratio=1e-3, init_obs=50)
    res = kf.fit_filter(y, x)

    filtered_betas = res["betas"][:, 0]
    # Correlation between estimated and true drifting path should be high (> 0.70)
    corr = np.corrcoef(filtered_betas[50:], beta_path[50:])[0, 1]
    assert corr > 0.70


def test_kalman_lookahead_discipline():
    """Strict test: estimate at t must be strictly identical when future data is altered."""
    np.random.seed(999)
    T = 150
    x = np.random.normal(0, 1, size=(T, 1))
    y = x[:, 0] * 1.2 + np.random.normal(0, 0.2, size=T)

    kf = KalmanBetaFilter(init_obs=40)
    res1 = kf.fit_filter(y[:100], x[:100])

    # Run on longer series with completely altered future data
    y_extended = y.copy()
    y_extended[100:] += 50.0  # Big jump in the future
    res2 = kf.fit_filter(y_extended, x)

    # Filtered estimate at index 99 must be mathematically identical
    np.testing.assert_allclose(res1["betas"][99], res2["betas"][99], rtol=1e-10)
    np.testing.assert_allclose(res1["se"][99], res2["se"][99], rtol=1e-10)


def test_compare_kalman_vs_rolling_ols():
    np.random.seed(777)
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    x = pd.DataFrame({"driver1": np.random.normal(0, 0.01, size=200)}, index=dates)
    y = pd.Series(x["driver1"] * 1.5 + np.random.normal(0, 0.005, size=200), index=dates)

    comp = compare_kalman_vs_rolling_ols(y, x, rolling_window=60, eval_horizon=80)
    assert comp["selected_estimator"] in ["kalman", "rolling_ols"]
    assert "kalman_r2_oos" in comp
    assert "rolling_ols_r2_oos" in comp
    assert comp["eval_observations"] == 80
