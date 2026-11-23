# pipeline/tests/unit/test_stats.py
import math
import numpy as np
import pandas as pd
import pytest

from radar.metrics.stats import compute_effective_n, compute_expanding_percentile, compute_expanding_zscore, compute_lag1_autocorr


def test_effective_n_known_answers():
    # Formula: N_eff = N * (1 - rho1) / (1 + rho1) with rho1 clipped to [0, 0.99]
    # N = 100, rho1 = 0.0 -> N_eff = 100 * (1 - 0) / (1 + 0) = 100
    # N = 100, rho1 = 0.5 -> N_eff = 100 * (0.5 / 1.5) = 100 / 3 = 33.333333
    # N = 100, rho1 = 0.8 -> N_eff = 100 * (0.2 / 1.8) = 100 / 9 = 11.111111
    # N = 100, rho1 = -0.5 -> rho1 clipped to 0 -> N_eff = 100
    # N = 100, rho1 = 1.0 -> rho1 clipped to 0.99 -> N_eff = 100 * (0.01 / 1.99) = 0.50251256

    assert math.isclose(compute_effective_n(100, 0.0), 100.0, rel_tol=1e-6)
    assert math.isclose(compute_effective_n(100, 0.5), 100.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(compute_effective_n(100, 0.8), 100.0 / 9.0, rel_tol=1e-6)
    assert math.isclose(compute_effective_n(100, -0.5), 100.0, rel_tol=1e-6)
    assert math.isclose(compute_effective_n(100, 1.0), 100.0 * 0.01 / 1.99, rel_tol=1e-5)


def test_lag1_autocorr():
    # Highly persistent AR(1) series
    np.random.seed(42)
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    corr = compute_lag1_autocorr(s)
    assert corr > 0.80

    # Short series
    short_s = pd.Series([1.0, 2.0])
    assert compute_lag1_autocorr(short_s) == 0.0


def test_expanding_zscore_known_answer_and_anti_lookahead():
    # Expanding z-score strictly excludes current point t:
    # History: [10.0, 20.0, 30.0]
    # At t=3: current point is 40.0
    # Prior mean = (10+20+30)/3 = 20.0
    # Prior var (ddof=1) = ((10-20)^2 + (20-20)^2 + (30-20)^2) / 2 = 200 / 2 = 100.0
    # Prior std = 10.0
    # z_3 = (40.0 - 20.0) / 10.0 = +2.0
    s = pd.Series([10.0, 20.0, 30.0, 40.0], index=pd.date_range("2026-01-01", periods=4, freq="B"))
    z = compute_expanding_zscore(s, min_obs=3)

    assert np.isnan(z.iloc[0])
    assert np.isnan(z.iloc[1])
    assert np.isnan(z.iloc[2])
    assert math.isclose(z.iloc[3], 2.0, rel_tol=1e-6)

    # Constant series (zero variance) yields 0.0
    const_s = pd.Series([5.0, 5.0, 5.0, 5.0], index=pd.date_range("2026-01-01", periods=4, freq="B"))
    const_z = compute_expanding_zscore(const_s, min_obs=3)
    assert const_z.iloc[3] == 0.0


def test_expanding_percentile_properties():
    # Returns values strictly in [0, 100]
    np.random.seed(42)
    s = pd.Series(np.random.randn(800), index=pd.date_range("2023-01-01", periods=800, freq="B"))
    pct = compute_expanding_percentile(s, min_obs=750)

    # First 749 must be NaN
    assert pct.iloc[:749].isna().all()

    # Valid percentiles must be in [0, 100]
    valid_pct = pct.dropna()
    assert (valid_pct >= 0.0).all()
    assert (valid_pct <= 100.0).all()

    # If length < min_obs, all should be NaN
    short_pct = compute_expanding_percentile(s.iloc[:100], min_obs=750)
    assert short_pct.isna().all()
