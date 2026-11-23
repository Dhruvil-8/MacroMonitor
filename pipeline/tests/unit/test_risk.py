# pipeline/tests/unit/test_risk.py
"""Unit tests for Risk Lab metrics (Section 4.4 of docs/MASTER_PLAN.md).

Tests:
1. Volatility cone monotonicity and bound invariants (min <= p10 <= ... <= max).
2. EWMA volatility responsiveness and non-negativity.
3. Distribution shape (skew, kurtosis, JB test, QQ quantiles).
4. Tail risk invariants (ES >= VaR, heavy-tail Tail Ratio >= 1.0).
5. Autocorrelation & volatility clustering (ACF of returns vs squared returns).
6. Underwater drawdown calculation and peak-trough-recovery tracking.
7. PCA factor structure with strict sign alignment (US loading > 0 invariant).
8. Downside correlation and stress correlation premium under tail co-movement.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from radar.metrics.risk import (
    compute_volatility_cone,
    compute_ewma_volatility,
    compute_distribution_shape_and_qq,
    compute_tail_risk,
    compute_autocorrelations,
    compute_underwater_drawdowns,
    compute_pca_factor_structure,
    compute_downside_correlation,
)


def test_volatility_cone_monotonicity_and_bounds():
    """Verify that volatility cone percentiles are strictly monotonic within each window."""
    np.random.seed(42)
    n = 400
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    # Geometric Brownian motion
    rets = np.random.normal(0.0004, 0.012, n)
    prices = pd.Series(100.0 * np.exp(np.cumsum(rets)), index=dates)

    windows = [10, 20, 60, 120]
    cone = compute_volatility_cone(prices, windows=windows)

    assert len(cone) == len(windows)
    for row in cone:
        assert row["window"] in windows
        assert row["min"] <= row["p10"] <= row["p25"] <= row["p50"] <= row["p75"] <= row["p90"] <= row["max"]
        assert 0.0 <= row["percentile"] <= 100.0
        assert row["current"] >= 0.0


def test_ewma_volatility_properties():
    """Verify EWMA volatility calculation and responsiveness."""
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    # Low vol for first 150 days, high vol for last 50 days
    rets1 = np.random.normal(0, 0.005, 150)
    rets2 = np.random.normal(0, 0.030, 50)
    rets = pd.Series(np.concatenate([rets1, rets2]), index=dates)

    ewma = compute_ewma_volatility(rets, decay=0.94)

    assert len(ewma) == n
    assert (ewma >= 0).all()
    # Volatility should be substantially higher at the end than at day 140
    assert float(ewma.iloc[-1]) > float(ewma.iloc[140]) * 2.0


def test_distribution_shape_and_qq():
    """Verify skewness, kurtosis, Jarque-Bera, and QQ plot quantiles."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Fat-tailed Student-t distribution (df=4)
    t_rets = pd.Series(sp_stats.t.rvs(df=4, scale=0.01, size=n), index=dates)
    res = compute_distribution_shape_and_qq(t_rets)

    # Student-t with df=4 has theoretical excess kurtosis = 6 / (4 - 4) -> infinite; sample kurtosis > 2.0
    assert res["excess_kurtosis"] > 1.5
    # Jarque-Bera should strongly reject normality (p-value < 0.01)
    assert res["jb_pvalue"] < 0.01
    assert len(res["qq_theoretical"]) == 50
    assert len(res["qq_empirical"]) == 50
    # QQ theoretical should be symmetric around 0
    assert abs(res["qq_theoretical"][0] + res["qq_theoretical"][-1]) < 0.05


def test_tail_risk_heavy_tail_properties():
    """Verify Historical vs Gaussian VaR/ES and Tail Ratio on fat-tailed returns."""
    np.random.seed(42)
    n = 2000
    dates = pd.date_range("2018-01-01", periods=n, freq="B")

    # Heavy-tailed Student-t (df=3.5)
    t_rets = pd.Series(sp_stats.t.rvs(df=3.5, loc=-0.0002, scale=0.01, size=n), index=dates)
    tail_risk = compute_tail_risk(t_rets, [0.95, 0.99])

    assert len(tail_risk) == 2
    for entry in tail_risk:
        # ES must exceed or equal VaR
        assert entry["hist_es"] >= entry["hist_var"]
        assert entry["gauss_es"] >= entry["gauss_var"]
        # In a fat-tailed distribution, historical ES is noticeably higher than Gaussian ES
        assert entry["tail_ratio"] >= 1.0


def test_autocorrelation_vol_clustering():
    """Verify that returns have near-zero autocorrelation while squared returns show persistence."""
    np.random.seed(42)
    n = 800
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Simulate GARCH-like volatility clustering: sigma_t depends on past shock
    r = np.zeros(n)
    sigma = np.zeros(n)
    sigma[0] = 0.01
    for t in range(1, n):
        sigma[t] = np.sqrt(0.00001 + 0.15 * (r[t-1]**2) + 0.80 * (sigma[t-1]**2))
        r[t] = np.random.normal(0, sigma[t])

    rets = pd.Series(r, index=dates)
    acf_res = compute_autocorrelations(rets, max_lags=10)

    assert len(acf_res["lags"]) == 10
    assert len(acf_res["acf_returns"]) == 10
    assert len(acf_res["acf_squared"]) == 10
    assert acf_res["band"] > 0
    assert acf_res["n_eff"] > 0

    # Lag 1 squared return autocorrelation should be positive and higher than raw return autocorrelation
    assert acf_res["acf_squared"][0] > acf_res["acf_returns"][0]
    assert acf_res["acf_squared"][0] > 0.10


def test_underwater_drawdowns_episodes():
    """Verify peak-to-trough drawdown series and episode extraction."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    # Prices: 100 -> 110 -> 88 (-20% from peak) -> 99 -> 110 (recovered) -> 100 -> 90 (-18.18%)
    prices = pd.Series([100.0, 110.0, 88.0, 99.0, 110.0, 115.0, 92.0, 92.0, 100.0, 115.0], index=dates)

    dd_series, episodes = compute_underwater_drawdowns(prices, top_k=3)

    assert len(dd_series) == 10
    assert (dd_series <= 0.0).all()
    assert float(dd_series.iloc[1]) == 0.0
    assert abs(float(dd_series.iloc[2]) - (-0.20)) < 1e-4

    assert len(episodes) >= 2
    # Deepest episode should be -20%
    deepest = episodes[0]
    assert abs(deepest["depth"] - (-0.20)) < 1e-3
    assert deepest["status"] == "Recovered"


def test_pca_factor_structure_sign_alignment():
    """Verify PCA explained variance and strict sign alignment invariant (US loading > 0)."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    # Global factor driving all countries positively
    global_factor = np.random.normal(0, 0.015, n)
    spx = global_factor + np.random.normal(0, 0.005, n)
    ftse = 0.8 * global_factor + np.random.normal(0, 0.006, n)
    dax = 0.9 * global_factor + np.random.normal(0, 0.007, n)
    n225 = 0.7 * global_factor + np.random.normal(0, 0.008, n)

    df = pd.DataFrame({"spx": spx, "ftse": ftse, "dax": dax, "n225": n225}, index=dates)

    pca = compute_pca_factor_structure(df, us_col="spx", window=250)

    assert len(pca["scree"]) >= 4
    # PC1 should capture dominant variance (> 50%)
    assert pca["scree"][0] > 50.0

    # Invariant: US loading on PC1 MUST be strictly positive
    assert pca["loadings_pc1"]["spx"] > 0.0
    # Other correlated markets should also have positive loadings
    assert pca["loadings_pc1"]["ftse"] > 0.0
    assert pca["loadings_pc1"]["dax"] > 0.0

    # Rolling history should exist
    assert len(pca["pc1_history"]) > 0


def test_downside_correlation_stress_premium():
    """Verify downside correlation and stress premium during tail co-movement."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    # Generate benchmark
    bench = np.random.normal(0, 0.01, n)
    # Target is weakly correlated on normal days, but strongly co-drops on bad benchmark days
    target = np.random.normal(0, 0.01, n)
    worst_bench_mask = bench < np.percentile(bench, 10.0)
    target[worst_bench_mask] = 1.2 * bench[worst_bench_mask] + np.random.normal(0, 0.002, np.sum(worst_bench_mask))

    s_target = pd.Series(target, index=dates)
    s_bench = pd.Series(bench, index=dates)

    res = compute_downside_correlation(s_target, s_bench, stress_quantile=0.10)

    assert res["downside_corr"] > res["all_day_corr"]
    assert res["stress_premium"] > 0.0
    assert res["n_stress"] == 50
