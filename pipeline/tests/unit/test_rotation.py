# pipeline/tests/unit/test_rotation.py
import math
import numpy as np
import pandas as pd
import pytest

from radar.metrics.rotation import (
    classify_rotation_phase,
    compute_ratio_momentum,
    compute_relative_strength_ratio,
)


def test_relative_strength_known_answer():
    # Asset price P_A = 200, Benchmark P_B = 100 -> RS = 2.0
    # If 60-day MA of RS is 2.0 -> Normalized RS Ratio = 100 * (1 + (2.0 - 2.0)/2.0) = 100.0
    # If RS rises to 2.2 -> RS Ratio = 100 * (1 + (2.2 - 2.0)/2.0) = 100 * (1 + 0.1) = 110.0
    dates = pd.date_range("2025-01-01", periods=70, freq="B")
    p_a = pd.Series([200.0] * 69 + [220.0], index=dates)
    p_b = pd.Series([100.0] * 70, index=dates)

    rs_ratio = compute_relative_strength_ratio(p_a, p_b, window=60)
    assert math.isclose(rs_ratio.iloc[-2], 100.0, rel_tol=1e-5)
    assert rs_ratio.iloc[-1] > 105.0


def test_rotation_phase_classification():
    # Four phases:
    # RS >= 100 and Mom >= 100 -> Leading
    # RS >= 100 and Mom < 100 -> Fading
    # RS < 100 and Mom < 100 -> Lagging
    # RS < 100 and Mom >= 100 -> Recovering

    assert classify_rotation_phase(102.5, 101.0) == "Leading"
    assert classify_rotation_phase(102.5, 98.5) == "Fading"
    assert classify_rotation_phase(97.0, 98.5) == "Lagging"
    assert classify_rotation_phase(97.0, 101.5) == "Recovering"
