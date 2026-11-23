"""Unit tests for P8: Slow Global Layer (BIS REER, TIC Foreign Holdings, Central Bank Balance Sheets)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.metrics.global_slow import (
    compute_reer_valuations,
    compute_foreign_treasury_holdings,
    compute_central_bank_assets,
    compute_india_macro_profile,
    compute_global_slow_payload,
)


def test_reer_valuation_expanding():
    dates = pd.date_range("2020-01-31", periods=60, freq="ME")
    # Upward trending REER
    usd_vals = np.linspace(95.0, 115.0, 60)
    s = pd.Series(usd_vals, index=dates)

    series_dict = {"reer_usd": s}
    reers = compute_reer_valuations(series_dict)
    usd = next(r for r in reers if r["id"] == "usd")
    assert usd["current_index"] == 115.0
    assert usd["percentile"] == 100.0  # Max of history
    assert usd["valuation"] == "overvalued"


def test_compute_global_slow_payload():
    payload = compute_global_slow_payload({})
    assert "reer" in payload
    assert len(payload["reer"]) == 6
    assert len(payload["foreign_treasury_holdings"]) >= 5
    assert len(payload["central_bank_assets"]) == 3
    assert payload["india_macro"]["fx_reserves_bn_usd"] > 500
    assert "lag_label" in payload
