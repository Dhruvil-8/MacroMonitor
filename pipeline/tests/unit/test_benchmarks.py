# pipeline/tests/unit/test_benchmarks.py
import numpy as np
import pandas as pd
import pytest
from radar.metrics.benchmarks import (
    compute_benchmark_agreement,
    compute_cross_correlation_spectrum,
    compute_episode_summary,
)


def test_cross_correlation_spectrum_detects_lead():
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    base = pd.Series(np.random.normal(0, 1, 100), index=dates)

    # s1 leads s2 by 2 periods (s2 is s1 shifted by +2 periods)
    s1 = base
    s2 = base.shift(2)

    spectrum = compute_cross_correlation_spectrum(s1, s2, max_lag=4)
    assert len(spectrum) == 9
    best_entry = max(spectrum, key=lambda x: x["correlation"])
    assert best_entry["lag_weeks"] == 2
    assert best_entry["correlation"] > 0.95


def test_benchmark_agreement():
    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    s1 = pd.Series(np.linspace(10, 50, 100) + np.random.normal(0, 1, 100), index=dates)
    s2 = pd.Series(np.linspace(10, 50, 100) + np.random.normal(0, 1, 100), index=dates)

    res = compute_benchmark_agreement(s1, s2, "Test Benchmark")
    assert res["n_obs"] == 100
    assert res["effective_n"] > 0
    assert res["pearson_corr"] > 0.8
    assert res["spearman_rank_corr"] > 0.8


def test_episode_summary_calculation():
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    stress = pd.Series(20.0, index=dates)

    # Inject shock around March 2020 (Covid shock)
    covid_mask = (dates >= "2020-03-01") & (dates <= "2020-03-31")
    stress.loc[covid_mask] = 85.0

    episodes = [
        {
            "id": "covid_crash_2020",
            "name": "COVID-19 Global Pandemic Shock",
            "start_date": "2020-02-19",
            "end_date": "2020-03-23",
            "peak_date": "2020-03-23",
            "category": "exogenous_liquidity",
            "description": "Worldwide lockdowns",
        }
    ]

    summary = compute_episode_summary(stress, episodes)
    assert len(summary) == 1
    assert summary[0]["peak_stress_score"] == 85.0
    assert summary[0]["elevated_ratio"] > 1.0
