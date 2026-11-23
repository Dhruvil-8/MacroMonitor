"""Unit tests for P3: Cboe Options-Implied Volatility and Sentiment Layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.metrics.volatility import (
    classify_slope_state,
    compute_trailing_realized_vol,
    compute_implied_vol_payload,
)


def test_classify_slope_state():
    assert classify_slope_state(0.05) == "contango"
    assert classify_slope_state(0.021) == "contango"
    assert classify_slope_state(0.01) == "flat"
    assert classify_slope_state(0.0) == "flat"
    assert classify_slope_state(-0.019) == "flat"
    assert classify_slope_state(-0.025) == "backwardation"
    assert classify_slope_state(-0.15) == "backwardation"
    assert classify_slope_state(None) == "flat"


def test_compute_trailing_realized_vol():
    dates = pd.date_range("2025-01-01", periods=50, freq="B")
    # Constant series
    const_prices = pd.Series(100.0, index=dates)
    real_const = compute_trailing_realized_vol(const_prices, window=21).dropna()
    assert len(real_const) > 0
    assert real_const.iloc[-1] == 0.0

    # Series with 1% daily volatility
    np.random.seed(42)
    ret = np.random.normal(0, 0.01, size=50)
    prices = pd.Series(100.0 * np.exp(np.cumsum(ret)), index=dates)
    real_vol = compute_trailing_realized_vol(prices, window=21).dropna()
    assert len(real_vol) > 0
    assert 5.0 < real_vol.iloc[-1] < 25.0


def test_compute_implied_vol_payload():
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    vix = pd.Series(np.linspace(15.0, 20.0, 300), index=dates)
    vix3m = vix * 1.08  # 8% upward slope (contango)
    vix9d = vix * 0.95
    vix6m = vix * 1.12
    vvix = pd.Series(85.0, index=dates)
    skew = pd.Series(130.0, index=dates)
    spx = pd.Series(5000.0 * np.exp(np.linspace(0, 0.1, 300)), index=dates)

    series_dict = {
        "vix": vix,
        "vix3m": vix3m,
        "vix9d": vix9d,
        "vix6m": vix6m,
        "vvix": vvix,
        "skew": skew,
        "spx": spx,
    }

    payload = compute_implied_vol_payload(series_dict, min_obs=50)
    assert payload["current_state"] == "contango"
    assert payload["slope_3m"] > 0.05
    assert len(payload["curve"]) == 4
    assert payload["curve"][0]["tenor"] == "9D"
    assert payload["curve"][1]["tenor"] == "30D"
    assert payload["curve"][2]["tenor"] == "3M"
    assert payload["curve"][3]["tenor"] == "6M"

    # Verify fear tail indices
    fear_ids = [x["id"] for x in payload["fear_tail_indices"]]
    assert "vix" in fear_ids
    assert "vvix" in fear_ids
    assert "skew" in fear_ids

    # Verify Implied vs Realized
    assert len(payload["implied_vs_realized"]) >= 1
    us_gap = payload["implied_vs_realized"][0]
    assert us_gap["market"] == "US (S&P 500)"
    assert us_gap["vrp_proxy"] is not None
