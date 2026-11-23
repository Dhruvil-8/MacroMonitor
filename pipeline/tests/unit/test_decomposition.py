# pipeline/tests/unit/test_decomposition.py
"""Tests proving Move Decomposer mathematical guarantees and point-in-time safety."""

import re
import numpy as np
import pandas as pd
import pytest

from radar.metrics.decomposition import (
    compute_fx_identity,
    compute_vif,
    fit_prior_window_betas,
    compute_decomposition,
    generate_decomposition_narrative,
)


def test_fx_identity_mathematical_precision():
    """Prove (1 + r_usd) = (1 + r_local) * (1 + r_fx) within numerical tolerance."""
    test_cases = [
        (0.05, 0.02),    # Local +5%, FX +2% -> USD +7.1%
        (-0.03, -0.01),  # Local -3%, FX -1% -> USD -3.97%
        (0.10, -0.08),   # Local +10%, FX -8% -> USD +1.2%
        (-0.15, 0.05),   # Local -15%, FX +5% -> USD -10.75%
        (0.0, 0.0),
    ]

    for r_local, r_fx in test_cases:
        r_usd, ret_local, ret_fx = compute_fx_identity(r_local=r_local, r_fx=r_fx)
        expected_usd = (1.0 + r_local) * (1.0 + r_fx) - 1.0
        assert abs(r_usd - expected_usd) < 1e-12
        assert abs(ret_local - r_local) < 1e-12
        assert abs(ret_fx - r_fx) < 1e-12

        # Invert: derive r_fx from r_usd and r_local
        derived_usd, _, derived_fx = compute_fx_identity(r_local=r_local, r_usd=r_usd)
        assert abs(derived_fx - r_fx) < 1e-12


def test_prior_window_no_lookahead_guarantee():
    """Prove modifying data inside the horizon does not change betas or contributions."""
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")

    # Generate synthetic target and factors
    f1 = pd.Series(np.random.normal(0, 0.01, size=n_days), index=dates, name="row_equity")
    f2 = pd.Series(np.random.normal(0, 0.015, size=n_days), index=dates, name="brent")
    factors_df = pd.concat([f1, f2], axis=1)

    target_returns = 0.7 * f1 - 0.4 * f2 + np.random.normal(0, 0.005, size=n_days)
    target_prices = pd.Series(100.0 * np.exp(np.cumsum(target_returns)), index=dates)

    # 1. Run baseline decomposition for 5-day horizon
    decomp_baseline = compute_decomposition(
        target_series=target_prices,
        factors_df=factors_df,
        horizon_days=5,
        lens="local",
        prior_window_days=250,
        target_name="TestTarget",
    )

    # 2. Alter target price INSIDE the horizon period (last 5 days)
    altered_target = target_prices.copy()
    altered_target.iloc[-2] *= 1.50  # Extreme 50% shock inside horizon!

    decomp_altered = compute_decomposition(
        target_series=altered_target,
        factors_df=factors_df,
        horizon_days=5,
        lens="local",
        prior_window_days=250,
        target_name="TestTarget",
    )

    # Betas MUST be identical because they were estimated on prior window ending before horizon
    for factor in ["row_equity", "brent"]:
        assert decomp_baseline["betas"][factor] == decomp_altered["betas"][factor]
        # Intervals must be identical
        assert decomp_baseline["intervals"][factor] == decomp_altered["intervals"][factor]


def test_planted_coefficients_recovery():
    """Prove OLS HAC recovers known planted beta coefficients within confidence intervals."""
    np.random.seed(123)
    n_days = 400
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")

    # True planted parameters
    true_beta_row = 0.85
    true_beta_oil = -0.40

    x_row = np.random.normal(0, 0.01, size=n_days)
    x_oil = np.random.normal(0, 0.02, size=n_days)
    epsilon = np.random.normal(0, 0.004, size=n_days)

    y_ret = true_beta_row * x_row + true_beta_oil * x_oil + epsilon

    y_prior = pd.Series(y_ret[:300], index=dates[:300])
    X_prior = pd.DataFrame({"row_equity": x_row[:300], "brent": x_oil[:300]}, index=dates[:300])

    fit = fit_prior_window_betas(y_prior, X_prior)

    recovered_row = fit["betas"]["row_equity"]
    recovered_oil = fit["betas"]["brent"]

    # Check that estimated betas are within 0.10 of true values
    assert abs(recovered_row - true_beta_row) < 0.10
    assert abs(recovered_oil - true_beta_oil) < 0.10

    # Check confidence intervals cover the true values
    ci_row = fit["intervals"]["row_equity"]
    ci_oil = fit["intervals"]["brent"]
    assert ci_row[0] <= true_beta_row <= ci_row[1]
    assert ci_oil[0] <= true_beta_oil <= ci_oil[1]


def test_vif_collinearity_triggers_ridge_fallback():
    """Prove collinear design matrix triggers Ridge regularization flag."""
    np.random.seed(99)
    n_obs = 250
    dates = pd.date_range("2023-01-01", periods=n_obs, freq="B")

    # Create two nearly identical factors (high collinearity)
    f1 = np.random.normal(0, 0.01, size=n_obs)
    f2 = f1 + np.random.normal(0, 0.0005, size=n_obs)  # Correlation > 0.999
    y = 0.5 * f1 + np.random.normal(0, 0.005, size=n_obs)

    y_series = pd.Series(y, index=dates)
    X_df = pd.DataFrame({"factor1": f1, "factor2": f2}, index=dates)

    fit = fit_prior_window_betas(y_series, X_df, vif_threshold=5.0)

    # Must be flagged as regularized
    assert fit["regularized"] is True
    assert fit["vifs"]["factor1"] > 5.0 or fit["vifs"]["factor2"] > 5.0


def test_mostly_unexplained_flag():
    """Prove target with pure noise returns mostly_unexplained = True."""
    np.random.seed(55)
    n_days = 350
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")

    # Factors
    f1 = pd.Series(np.random.normal(0, 0.01, size=n_days), index=dates, name="row")
    factors_df = pd.DataFrame({"row": f1})

    # Target is independent pure random noise
    target_prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0, 0.02, size=n_days))), index=dates)

    decomp = compute_decomposition(
        target_series=target_prices,
        factors_df=factors_df,
        horizon_days=5,
        lens="local",
        prior_window_days=250,
        unexplained_r2_floor=0.10,
        target_name="NoiseAsset",
    )

    assert decomp["mostly_unexplained"] is True
    assert decomp["r2_out_sample"] < 0.10


def test_time_zone_synthetic_lagged_pair_alignment():
    """Prove aligned decomposition attributes lagged overseas move correctly while naive fails.

    Scenario: Asian target (e.g. Nikkei / Nifty) responds to US equity move from previous night.
    Naive same-date correlation/regression fails to detect the relationship (R2 ~ 0).
    Time-aligned regression correctly identifies the factor beta and high R2.
    """
    from datetime import date
    from radar.metrics.decomposition import (
        determine_calendar_lag,
        align_factor_series,
    )

    calendars_cfg = {
        "n225": {"tz": "Asia/Tokyo", "close_time": "15:30"},
        "spx": {"tz": "America/New_York", "close_time": "16:00"},
    }

    # Verify lag determination
    ref_date = date(2024, 6, 15)  # Summer DST
    lag = determine_calendar_lag("n225", "spx", calendars_cfg, ref_date)
    assert lag == 1, f"Expected lag 1 for US factor relative to Tokyo target, got {lag}"

    # Verify inverse lag (US target relative to Tokyo factor)
    inv_lag = determine_calendar_lag("spx", "n225", calendars_cfg, ref_date)
    assert inv_lag == 0, f"Expected lag 0 for Tokyo factor relative to US target, got {inv_lag}"

    # Generate synthetic data
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")

    # True US factor return on day t
    us_returns = pd.Series(np.random.normal(0, 0.012, size=n_days), index=dates, name="spx")

    # True Asian market responds to PREVIOUS US close (day t-1) with beta = 0.80
    true_beta = 0.80
    raw_asian = true_beta * us_returns.shift(1).values + np.random.normal(0, 0.003, size=n_days)
    asian_returns = pd.Series(raw_asian, index=dates, name="n225")

    clean_df = pd.concat([asian_returns, us_returns], axis=1).dropna()
    y = clean_df["n225"]
    x_naive = clean_df[["spx"]]

    # 1. Naive same-date regression
    fit_naive = fit_prior_window_betas(y, x_naive)
    naive_beta = fit_naive["betas"]["spx"]
    naive_r2 = fit_naive["r2_in_sample"]

    # Naive version must FAIL to capture the relationship
    assert abs(naive_beta) < 0.15, f"Naive beta unexpectedly high: {naive_beta}"
    assert naive_r2 < 0.05, f"Naive R2 unexpectedly high: {naive_r2}"

    # 2. Time-aligned regression (factor shifted by determined lag = 1)
    x_aligned = pd.DataFrame(
        {"spx": align_factor_series(clean_df["spx"], lag=1)}, index=clean_df.index
    ).dropna()
    y_aligned = y.loc[x_aligned.index]

    fit_aligned = fit_prior_window_betas(y_aligned, x_aligned)
    aligned_beta = fit_aligned["betas"]["spx"]
    aligned_r2 = fit_aligned["r2_in_sample"]

    # Aligned version must SUCCEED in capturing the true relationship
    assert abs(aligned_beta - true_beta) < 0.05, f"Aligned beta {aligned_beta} didn't match {true_beta}"
    assert fit_aligned["intervals"]["spx"][0] <= true_beta <= fit_aligned["intervals"]["spx"][1]
    assert aligned_r2 > 0.80, f"Aligned R2 {aligned_r2} was not high"


def test_narrative_safety_and_forbidden_words():
    """Prove generated narrative sentences adhere strictly to associative language."""
    narrative = generate_decomposition_narrative(
        target_name="India",
        horizon_days=5,
        lens="usd",
        total_return=-0.014,
        contributions={"rest_of_world_equity": -0.006, "us_dollar": -0.003, "brent": -0.002, "fx_effect": -0.004},
        residual=0.001,
        r2_oos=0.41,
    )

    forbidden_patterns = [
        r"\bbecause\b",
        r"\bcaused by\b",
        r"\bwill\b",
        r"\bexpected\b",
        r"\bbuy\b",
        r"\bsell\b",
        r"\btarget\b",
        r"\bforecast\b",
    ]

    for pattern in forbidden_patterns:
        match = re.search(pattern, narrative, re.IGNORECASE)
        assert match is None, f"Forbidden term '{pattern}' found in narrative: '{narrative}'"

    # Confirm associative language is present
    assert "associated with" in narrative
    assert "out-of-sample R-squared 0.41" in narrative
