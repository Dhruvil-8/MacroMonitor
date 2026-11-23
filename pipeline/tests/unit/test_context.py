# pipeline/tests/unit/test_context.py
"""Unit tests for Historical Context layer (Section 4.5 & Section 5.1 of docs/MASTER_PLAN.md).

Tests:
1. Declustered analogs spacing and top-k selection.
2. Distribution histogram bins and total count conservation.
3. Gaussian Kernel Density Estimate (Silverman's rule) and normal curve integration.
4. QQ plot theoretical vs empirical quantile properties.
5. ECDF monotonicity and bound invariants [0, 1].
6. Calendar monthly heatmap aggregation.
7. Declustered analog forward outcome calculation (5D, 20D).
"""

import math
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from radar.metrics.context import (
    compute_declustered_analogs,
    compute_distribution_histogram,
    compute_kde_and_normal_overlay,
    compute_qq_quantiles,
    compute_ecdf,
    compute_calendar_heatmap,
)


def test_declustered_analogs_spacing():
    """Verify that declustering enforces minimum time gap between selected episodes."""
    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    values = np.linspace(50, 100, 100)
    values[50] = 149.9
    values[51] = 149.95
    values[52] = 149.92
    values[20] = 149.8
    values[99] = 150.0

    s = pd.Series(values, index=dates)
    analogs = compute_declustered_analogs(s, target_idx=99, top_k=3, min_gap_days=20)

    assert len(analogs) >= 2
    selected_dates = [a["date"] for a in analogs]
    for i in range(len(selected_dates)):
        for j in range(i + 1, len(selected_dates)):
            gap = abs((pd.to_datetime(selected_dates[i]) - pd.to_datetime(selected_dates[j])).days)
            assert gap >= 20, f"Dates {selected_dates[i]} and {selected_dates[j]} are only {gap} days apart!"


def test_distribution_histogram_bins():
    """Verify histogram binning and count conservation."""
    np.random.seed(42)
    s = pd.Series(np.random.normal(0, 1, 500))
    hist_data = compute_distribution_histogram(s, num_bins=20)

    assert len(hist_data["bins"]) == 20
    assert len(hist_data["counts"]) == 20
    assert sum(hist_data["counts"]) == 500


def test_kde_and_normal_overlay_integration():
    """Verify Silverman's KDE and normal overlay properties."""
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0.0005, 0.015, 800))
    res = compute_kde_and_normal_overlay(rets, grid_points=100)

    assert len(res["grid"]) == 100
    assert len(res["kde"]) == 100
    assert len(res["normal"]) == 100

    # Grid should be centered near 0
    assert abs(res["grid"][0] + res["grid"][-1]) < 0.02

    # Numerical integration using trapezoid rule should approximate 1.0 (within 3%)
    dx = res["grid"][1] - res["grid"][0]
    kde_area = float(np.sum(res["kde"]) * dx)
    norm_area = float(np.sum(res["normal"]) * dx)
    assert abs(kde_area - 1.0) < 0.05
    assert abs(norm_area - 1.0) < 0.05


def test_qq_quantiles_properties():
    """Verify QQ plot empirical vs theoretical quantiles."""
    np.random.seed(42)
    # Standard normal returns
    rets = pd.Series(np.random.normal(0.0, 0.02, 1000))
    qq = compute_qq_quantiles(rets, num_points=50)

    assert len(qq["theoretical"]) == 50
    assert len(qq["empirical"]) == 50

    # On Gaussian data, empirical and theoretical should match closely
    max_err = np.max(np.abs(np.array(qq["empirical"]) - np.array(qq["theoretical"])))
    assert max_err < 0.20


def test_ecdf_monotonicity_and_bounds():
    """Verify empirical CDF values are monotonic increasing in [0, 1]."""
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0, 0.01, 300))
    ecdf = compute_ecdf(rets, num_points=50)

    assert len(ecdf["x"]) == 50
    assert len(ecdf["y"]) == 50
    # X and Y must be strictly non-decreasing
    assert all(ecdf["x"][i] <= ecdf["x"][i+1] for i in range(len(ecdf["x"]) - 1))
    assert all(ecdf["y"][i] <= ecdf["y"][i+1] for i in range(len(ecdf["y"]) - 1))
    assert 0.0 < ecdf["y"][0] <= 0.05
    assert 0.95 <= ecdf["y"][-1] <= 1.0


def test_calendar_heatmap_monthly_returns():
    """Verify calendar monthly returns aggregation."""
    # 2 years of business days
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")
    prices = pd.Series(100.0 * np.cumprod(1.0 + np.random.normal(0.0002, 0.01, len(dates))), index=dates)

    cal = compute_calendar_heatmap(prices, is_prices=True)

    assert len(cal) == 24  # 12 months * 2 years
    for row in cal:
        assert row["year"] in [2023, 2024]
        assert 1 <= row["month"] <= 12
        assert isinstance(row["return"], float)


def test_analog_forward_outcomes():
    """Verify that declustered analogs accurately compute forward 5D and 20D returns."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    # Linear price path: P_t = 100 + t
    prices = pd.Series(100.0 + np.arange(100, dtype=float), index=dates)

    analogs = compute_declustered_analogs(prices, target_idx=99, top_k=3, min_gap_days=20)
    assert len(analogs) > 0
    # For any episode where t + 5 < 100, fwd_5d should equal (P_{t+5} - P_t) / P_t
    for a in analogs:
        if a["fwd_5d"] is not None:
            dt = pd.to_datetime(a["date"])
            idx = prices.index.get_loc(dt)
            expected = round(float(prices.iloc[idx + 5] / prices.iloc[idx] - 1.0), 4)
            assert a["fwd_5d"] == expected
