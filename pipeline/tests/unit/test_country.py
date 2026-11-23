# pipeline/tests/unit/test_country.py
"""Tests for country layer performance matrix and FX identity guarantees."""

import numpy as np
import pandas as pd
import pytest

from radar.metrics.country import (
    compute_country_metrics,
    compute_country_breadth,
    compute_average_pairwise_correlation,
    compute_normalized_fx_return,
)


def test_country_matrix_fx_identity():
    """Prove (1 + r_usd) = (1 + r_local) * (1 + r_fx) holds for every country and horizon."""
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")

    # Generate synthetic price series for test countries
    prices_dict = {}
    for code, idx_id, fx_id in [
        ("us", "spx", None),
        ("in", "nifty50", "usdinr"),
        ("de", "dax", "eurusd"),
        ("jp", "n225", "usdjpy"),
        ("gb", "ftse", "gbpusd"),
    ]:
        prices_dict[idx_id] = pd.Series(
            100.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, size=n_days))),
            index=dates,
        )
        if fx_id:
            prices_dict[fx_id] = pd.Series(
                80.0 * np.exp(np.cumsum(np.random.normal(0.0001, 0.005, size=n_days))),
                index=dates,
            )

    result = compute_country_metrics(prices_dict)

    for country in result["countries"]:
        for h in ["1D", "5D", "1M", "3M", "1Y"]:
            r_loc = country["returns_local"][h]
            r_usd = country["returns_usd"][h]
            r_fx = country["returns_fx"][h]

            expected_usd = (1.0 + r_loc) * (1.0 + r_fx) - 1.0
            assert abs(r_usd - expected_usd) < 1e-4, (
                f"FX identity failed for {country['name']} ({h}): "
                f"r_usd={r_usd}, (1+r_loc)(1+r_fx)-1={expected_usd}"
            )


def test_country_breadth_invariants():
    """Prove breadth is bounded in [0.0, 1.0]."""
    dates = pd.date_range("2023-01-01", periods=250, freq="B")
    df = pd.DataFrame(
        {
            "c1": np.linspace(100, 200, 250),  # strictly above MA
            "c2": np.linspace(200, 100, 250),  # strictly below MA
        },
        index=dates,
    )
    b = compute_country_breadth(df, window=50)
    assert 0.0 <= b <= 1.0
    assert b == 0.5  # Exactly 1 out of 2 above


def test_average_pairwise_correlation_invariants():
    """Prove average pairwise correlation is bounded in [-1.0, 1.0]."""
    np.random.seed(10)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rets = pd.DataFrame(
        {
            "c1": np.random.normal(0, 0.01, 100),
            "c2": np.random.normal(0, 0.01, 100),
            "c3": np.random.normal(0, 0.01, 100),
        },
        index=dates,
    )
    avg_corr = compute_average_pairwise_correlation(rets, window=60)
    assert -1.0 <= avg_corr <= 1.0


def test_fx_quote_inversion():
    """Prove USD/JPY rise implies negative foreign currency return."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    # USD/JPY rises from 150 to 155 (USD strengthened, JPY weakened)
    usdjpy = pd.Series([150.0] * 5 + [155.0] * 5, index=dates)
    r_fx = compute_normalized_fx_return(usdjpy, window=5, quote_type="usd_per_foreign_inv")
    assert r_fx < 0.0, f"Expected negative JPY return when USD/JPY rises, got {r_fx}"
