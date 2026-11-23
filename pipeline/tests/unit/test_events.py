"""Unit tests for P6: Event Studies on Central Bank Decision Days."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.metrics.events import (
    estimate_market_model,
    compute_event_car_path,
    compute_event_study_payload,
)


def test_estimate_market_model_recovery():
    """Verify market model recovers beta on pre-event estimation window."""
    dates = pd.date_range("2024-01-01", periods=350, freq="B")
    event_date = dates[320]  # Event near the end

    m_ret = pd.Series(np.random.normal(0, 0.01, size=350), index=dates)
    planted_beta = 1.35
    planted_alpha = 0.0002
    y_ret = planted_alpha + planted_beta * m_ret + pd.Series(np.random.normal(0, 0.003, size=350), index=dates)

    alpha, beta, res_var = estimate_market_model(y_ret, m_ret, event_date, pre_window_days=200, pre_gap_days=30)
    assert abs(beta - planted_beta) < 0.10
    assert abs(alpha - planted_alpha) < 0.005


def test_event_car_path_zero_lookahead():
    """Verify event CAR path evaluates abnormal returns correctly."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    event_date = dates[50]

    m_ret = pd.Series(0.01, index=dates)
    y_ret = pd.Series(0.01, index=dates)
    # On event day (index 50), target jumps by +2%
    y_ret.iloc[50] += 0.02

    path = compute_event_car_path(y_ret, m_ret, event_date, alpha=0.0, beta=1.0, window_pre=5, window_post=10)
    assert path is not None
    assert path["rel_days"] == list(range(-5, 11))
    assert len(path["abnormal_returns"]) == 16
    # Day 0 abnormal return should reflect the +2% jump (2.0%)
    assert abs(path["day_0_ar"] - 2.0) < 0.01
    # Subsequent CAR should remain at +2%
    assert abs(path["car"][-1] - 2.0) < 0.01


def test_compute_event_study_payload():
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    m_ret = pd.Series(np.random.normal(0, 0.01, size=500), index=dates)
    y_ret = m_ret * 1.2 + pd.Series(np.random.normal(0, 0.005, size=500), index=dates)

    payload = compute_event_study_payload(y_ret, m_ret, target_name="Test SPX", event_type="fomc")
    assert payload["event_type"] == "fomc"
    assert payload["n_events"] > 0
    assert len(payload["ribbon"]) == 16
    assert payload["magnitude_ratio"] > 0
    assert "by_decision" in payload
