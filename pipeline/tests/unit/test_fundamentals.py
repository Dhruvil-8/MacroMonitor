"""Unit tests for P9: EIA Energy Inventories & Seasonal Bands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.metrics.fundamentals import (
    compute_seasonal_bands_for_series,
    compute_fundamentals_payload,
)


def test_seasonal_band_calculation_and_current_year_exclusion():
    """Verify seasonal band excludes the current year strictly and computes 5-year min/max/mean."""
    # Generate 7 years of weekly data: 2019 to 2025
    dates = pd.date_range("2019-01-04", "2025-11-21", freq="W-FRI")
    values = []
    for dt in dates:
        w = dt.isocalendar().week
        # Base seasonal sinusoidal wave
        seasonal = 400.0 + 30.0 * np.sin(2 * np.pi * w / 52)
        # In current year (2025), plant a huge inventory spike of 600.0
        if dt.year == 2025 and w == 20:
            seasonal = 600.0
        values.append(seasonal)

    s = pd.Series(values, index=dates)

    res = compute_seasonal_bands_for_series(s, "crude_test", "Crude Test", as_of_year=2025)
    assert res["id"] == "crude_test"
    assert len(res["weeks"]) == 52

    # Week 20 5-year max should NOT include the 2025 spike of 600!
    # Because 2025 is the current year, the band is formed strictly from 2020-2024.
    w20 = next(w for w in res["weeks"] if w["week"] == 20)
    assert w20["max_5y"] < 500.0
    # Current year value in week 20 should be 600.0
    assert w20["current_year"] == 600.0


def test_compute_fundamentals_payload():
    dates = pd.date_range("2020-01-03", periods=250, freq="W-FRI")
    crude = pd.Series(420.0 + np.random.normal(0, 10, size=250), index=dates)

    series_dict = {
        "eia_crude": crude,
    }

    payload = compute_fundamentals_payload(series_dict)
    assert "inventories" in payload
    assert len(payload["inventories"]) == 4
    crude_item = payload["inventories"][0]
    assert crude_item["id"] == "crude_oil"
    assert crude_item["current_level"] > 300
    assert crude_item["deviation_percentile"] >= 0
    assert "EIA Weekly" in payload["lag_label"]
