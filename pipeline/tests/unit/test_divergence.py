# pipeline/tests/unit/test_divergence.py
import math
import numpy as np
import pandas as pd
import pytest

from radar.metrics.divergence import (
    compute_cusum_series,
    compute_divergence_series,
    detect_divergences,
)


def test_divergence_known_answer():
    # Assets A and B with 20D momentums:
    # Z(mom_A) = +2.0, Z(mom_B) = -1.5 -> D_t = 2.0 - (-1.5) = +3.5
    z_a = pd.Series([2.0, 2.1, 2.2], index=pd.date_range("2026-01-01", periods=3, freq="B"))
    z_b = pd.Series([-1.5, -1.4, -1.3], index=pd.date_range("2026-01-01", periods=3, freq="B"))

    div = compute_divergence_series(z_a, z_b)
    assert math.isclose(div.iloc[0], 3.5, rel_tol=1e-6)
    assert math.isclose(div.iloc[1], 3.5, rel_tol=1e-6)


def test_divergence_requires_consecutive_days():
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    # Threshold = 2.5
    # Day 0: 2.6 (1 day) -> Not triggered
    # Day 1: 2.7 (2 days) -> Not triggered
    # Day 2: 2.8 (3 days) -> Triggered!
    # Day 3: 1.0 (Reset) -> Not triggered
    # Day 4: 2.9 (1 day) -> Not triggered
    d_series = pd.Series([2.6, 2.7, 2.8, 1.0, 2.9], index=dates)

    flags = detect_divergences(d_series, threshold=2.5, consecutive_days=3)
    assert not flags.iloc[0]
    assert not flags.iloc[1]
    assert flags.iloc[2]
    assert not flags.iloc[3]
    assert not flags.iloc[4]


def test_cusum_upward_shift_detected():
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    # Base 0 for first 15 days, then sudden upward step of +3.0
    vals = [0.0] * 15 + [3.0] * 15
    s = pd.Series(vals, index=dates)

    res = compute_cusum_series(s, slack_k=0.5, threshold_h=3.5)
    assert len(res["shifts"]) >= 1
    assert any(shift["direction"] == "upward" for shift in res["shifts"])
    assert len(res["dates"]) == 30
    assert len(res["cusum_pos"]) == 30
    assert len(res["cusum_neg"]) == 30


def test_cusum_downward_shift_detected():
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    # Base 0 for first 15 days, then sudden downward step of -3.0
    vals = [0.0] * 15 + [-3.0] * 15
    s = pd.Series(vals, index=dates)

    res = compute_cusum_series(s, slack_k=0.5, threshold_h=3.5)
    assert len(res["shifts"]) >= 1
    assert any(shift["direction"] == "downward" for shift in res["shifts"])


def test_cusum_zero_false_alarms_on_noise():
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=200, freq="B")
    # Pure standard normal noise
    noise = pd.Series(np.random.normal(0.0, 1.0, 200), index=dates)

    res = compute_cusum_series(noise, slack_k=0.5, threshold_h=4.5)
    # At h=4.5 and k=0.5 on pure Gaussian noise, shifts should be zero
    assert len(res["shifts"]) == 0


def test_cusum_short_series_graceful():
    s = pd.Series([1.0, 2.0, 3.0])
    res = compute_cusum_series(s)
    assert res["dates"] == []
    assert res["shifts"] == []

