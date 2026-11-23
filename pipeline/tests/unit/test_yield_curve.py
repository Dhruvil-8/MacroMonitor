"""Unit tests for P4: Nelson-Siegel Yield Curve Lab and Rates Analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.metrics.yield_curve import (
    fit_nelson_siegel,
    evaluate_nelson_siegel,
    nelson_siegel_loadings,
    compute_yield_curve_payload,
    MATURITIES_MONTHS,
    MATURITY_LABELS,
)


def test_nelson_siegel_loadings_shape():
    loadings = nelson_siegel_loadings(MATURITIES_MONTHS)
    assert loadings.shape == (len(MATURITIES_MONTHS), 3)
    # First column (Level) must be all 1s
    np.testing.assert_allclose(loadings[:, 0], 1.0)
    # Slope factor starts near 1.0 for tau -> 0 and approaches 0 for large tau
    assert loadings[0, 1] > loadings[-1, 1]


def test_nelson_siegel_exact_recovery():
    """Plant exact L, S, C parameters and verify fit recovers them within tolerance."""
    planted_L = 5.25
    planted_S = -1.50
    planted_C = -2.20

    taus = MATURITIES_MONTHS
    exact_yields = evaluate_nelson_siegel(taus, planted_L, planted_S, planted_C)

    L, S, C, rmse = fit_nelson_siegel(taus, exact_yields)
    assert abs(L - planted_L) < 1e-4
    assert abs(S - planted_S) < 1e-4
    assert abs(C - planted_C) < 1e-4
    assert rmse < 1e-5


def test_yield_curve_payload_spreads_and_inversion():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    # Build inverted curve: 3M > 10Y and 2Y > 10Y
    df = pd.DataFrame(
        {
            "3M": np.full(100, 5.25),
            "2Y": np.full(100, 4.80),
            "5Y": np.full(100, 4.40),
            "10Y": np.full(100, 4.25),
            "30Y": np.full(100, 4.50),
        },
        index=dates,
    )

    real_10y = pd.Series(2.05, index=dates)
    be_10y = pd.Series(2.20, index=dates)

    payload = compute_yield_curve_payload(df, real_10y_series=real_10y, breakeven_10y_series=be_10y, min_obs=30)
    assert payload["factors"]["level"] > 4.0
    # In Diebold-Li convention, y(0) = L + S, so when short rate > long rate (inverted), S > 0
    assert payload["factors"]["slope"] > 0

    # Check spreads
    spread_10_2 = next(s for s in payload["spreads"] if "10Y - 2Y" in s["name"])
    assert spread_10_2["inverted"] is True
    assert spread_10_2["current"] < 0

    # Check anatomy
    assert len(payload["anatomy"]) > 0
    latest_a = payload["anatomy"][-1]
    assert latest_a["nominal_10y"] == 4.25
    assert latest_a["real_10y"] == 2.05
    assert latest_a["breakeven_10y"] == 2.20
    # Fisher identity holds: 2.05 + 2.20 == 4.25
    assert abs((latest_a["real_10y"] + latest_a["breakeven_10y"]) - latest_a["nominal_10y"]) < 0.01
