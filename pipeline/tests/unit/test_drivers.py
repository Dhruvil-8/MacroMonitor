# pipeline/tests/unit/test_drivers.py
"""Unit tests for Driver sensitivity layer metrics (Section 4.3 of docs/MASTER_PLAN.md).

Tests:
1. Planted-coefficient recovery under Newey-West HAC errors.
2. Noise fixture rejection: random unassociated factors yield non-significant betas.
3. Stability metric boundary and sign invariants.
4. Pre-registered economic priors matching and alignment evaluation.
5. Rolling beta anti-lookahead guarantee.
6. Bivariate scatter and confidence band properties.
"""

import numpy as np
import pandas as pd
import pytest

from radar.metrics.drivers import (
    fit_window_multi_factor,
    compute_driver_stability,
    get_economic_prior,
    evaluate_prior_alignment,
    compute_rolling_multi_factor_series,
    compute_bivariate_scatter_and_fit,
    analyze_target_drivers,
)


def test_planted_coefficient_recovery_hac():
    """Verify that multi-factor OLS with HAC recovers planted true coefficients."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    x1 = np.random.normal(0, 0.01, n)
    x2 = np.random.normal(0, 0.015, n)
    noise = np.random.normal(0, 0.003, n)

    true_b1 = 0.85
    true_b2 = -0.50
    y = true_b1 * x1 + true_b2 * x2 + noise

    y_series = pd.Series(y, index=dates)
    X_df = pd.DataFrame({"factor1": x1, "factor2": x2}, index=dates)

    fit = fit_window_multi_factor(y_series, X_df)

    assert abs(fit["betas"]["factor1"] - true_b1) < 0.05
    assert abs(fit["betas"]["factor2"] - true_b2) < 0.05

    ci1_low, ci1_high = fit["intervals"]["factor1"]
    assert ci1_low <= true_b1 <= ci1_high

    ci2_low, ci2_high = fit["intervals"]["factor2"]
    assert ci2_low <= true_b2 <= ci2_high

    assert fit["r2"] > 0.80


def test_noise_fixture_rejection():
    """Verify that random unassociated noise yields betas whose 95% CI covers zero."""
    np.random.seed(999)
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    y = pd.Series(np.random.normal(0, 0.01, n), index=dates)
    X = pd.DataFrame({
        "noise1": np.random.normal(0, 0.01, n),
        "noise2": np.random.normal(0, 0.01, n),
    }, index=dates)

    fit = fit_window_multi_factor(y, X)

    for col in ["noise1", "noise2"]:
        ci_low, ci_high = fit["intervals"][col]
        # 95% CI must span zero for pure noise
        assert ci_low <= 0.0 <= ci_high
        assert fit["p_values"][col] > 0.05

    assert fit["r2"] < 0.10


def test_stability_metric_invariants():
    """Verify stability metric is bounded in [0.0, 1.0] and measures sign consistency."""
    # 1. Perfectly stable positive series
    all_pos = pd.Series([0.5, 0.6, 0.4, 0.8, 0.7])
    assert compute_driver_stability(all_pos) == 1.0

    # 2. Perfectly stable negative series
    all_neg = pd.Series([-0.3, -0.2, -0.5, -0.1])
    assert compute_driver_stability(all_neg) == 1.0

    # 3. 50% stable series: latest is positive, 2 prior negative, 2 prior positive
    mixed = pd.Series([-0.2, 0.3, -0.1, 0.4])
    # Latest is 0.4 (pos). Positive values: 0.3, 0.4 (2/4 = 0.5)
    assert compute_driver_stability(mixed) == 0.5

    # 4. Empty series
    assert compute_driver_stability(pd.Series([], dtype=float)) == 0.0


def test_economic_prior_matching():
    """Verify pre-registered economic priors match documented economic logic."""
    # Oil importers (India, Japan, Germany) should have negative prior for Brent
    assert get_economic_prior("brent", "nifty50", "in") == "negative"
    assert get_economic_prior("brent", "n225", "jp") == "negative"
    assert get_economic_prior("brent", "dax", "de") == "negative"

    # Oil exporters (Canada, Brazil) should have positive prior for Brent
    assert get_economic_prior("brent", "tsse", "ca") == "positive"
    assert get_economic_prior("brent", "ibov", "br") == "positive"

    # ROW equity and Copper are universally positive
    assert get_economic_prior("row_equity", "spx", "us") == "positive"
    assert get_economic_prior("copper", "nifty50", "in") == "positive"

    # DXY tightening proxy
    assert get_economic_prior("dxy", "nifty50", "in") == "negative"
    assert get_economic_prior("dxy", "spx", "us") == "negative"

    # Alignment checking
    assert evaluate_prior_alignment(0.75, "positive") is True
    assert evaluate_prior_alignment(-0.25, "positive") is False
    assert evaluate_prior_alignment(-0.40, "negative") is True
    assert evaluate_prior_alignment(0.15, "negative") is False
    assert evaluate_prior_alignment(0.05, "neutral") is True


def test_rolling_beta_anti_lookahead():
    """Verify rolling beta estimates depend strictly on past data up to date t."""
    np.random.seed(123)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    x = pd.Series(np.random.normal(0, 0.01, n), index=dates, name="x")
    y = pd.Series(0.8 * x + np.random.normal(0, 0.002, n), index=dates, name="y")

    # Rolling multi-factor on full dataset
    rolling_full = compute_rolling_multi_factor_series(y, pd.DataFrame({"x": x}), window=100, step=10)

    # Sub-dataset truncated at day 200
    rolling_trunc = compute_rolling_multi_factor_series(y.iloc[:200], pd.DataFrame({"x": x.iloc[:200]}), window=100, step=10)

    df_full = rolling_full["x"]
    df_trunc = rolling_trunc["x"]

    # Beta values on overlapping dates must match identically
    common_dates = set(df_full["date"]).intersection(set(df_trunc["date"]))
    assert len(common_dates) > 0

    for d in common_dates:
        b_full = float(df_full[df_full["date"] == d]["beta"].iloc[0])
        b_trunc = float(df_trunc[df_trunc["date"] == d]["beta"].iloc[0])
        assert abs(b_full - b_trunc) < 1e-6, f"Lookahead detected at date {d}"


def test_bivariate_scatter_and_fit():
    """Verify bivariate scatter output contains valid points, line, band, and statistics."""
    np.random.seed(77)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    x = pd.Series(np.random.normal(0, 0.01, n), index=dates)
    y = pd.Series(1.2 * x + np.random.normal(0, 0.005, n), index=dates)

    scatter_res = compute_bivariate_scatter_and_fit(y, x, window=200)

    assert len(scatter_res["scatter"]) == 200
    assert len(scatter_res["line"]) == 25
    assert len(scatter_res["band_upper"]) == 25
    assert len(scatter_res["band_lower"]) == 25
    assert abs(scatter_res["beta"] - 1.2) < 0.15
    assert scatter_res["r2"] > 0.50
    assert scatter_res["t_stat"] > 5.0
