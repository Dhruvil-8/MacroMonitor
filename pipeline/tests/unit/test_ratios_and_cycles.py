# pipeline/tests/unit/test_ratios_and_cycles.py
"""Unit tests for cross-asset ratios, fair-value models, and 4-quadrant macro cycles."""

import numpy as np
import pandas as pd

from radar.metrics.ratios import compute_ratios_and_valuations
from radar.metrics.cycles import classify_quadrant, compute_macro_cycles


def _make_series(start_val: float, drift: float, vol: float, n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    rets = rng.normal(drift, vol, size=n)
    vals = start_val * np.exp(np.cumsum(rets))
    return pd.DataFrame({"value": vals}, index=dates)


def _make_level_series(start_val: float, drift: float, vol: float, n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    steps = rng.normal(drift, vol, size=n)
    vals = start_val + np.cumsum(steps)
    return pd.DataFrame({"value": vals}, index=dates)


def test_classify_quadrant_boundaries():
    assert classify_quadrant(0.5, -0.2) == "Recovery"
    assert classify_quadrant(0.5, 0.2) == "Overheat"
    assert classify_quadrant(-0.5, 0.2) == "Stagflation"
    assert classify_quadrant(-0.5, -0.2) == "Reflation"


def test_ratios_and_gold_valuation_computation():
    pts = {
        "gold": _make_series(2000.0, 0.0005, 0.01, n=400, seed=1),
        "silver": _make_series(25.0, 0.0003, 0.015, n=400, seed=2),
        "copper": _make_series(4.0, 0.0002, 0.012, n=400, seed=3),
        "brent": _make_series(80.0, 0.0001, 0.018, n=400, seed=4),
        "wti": _make_series(76.0, 0.0001, 0.018, n=400, seed=5),
        "dxy": _make_series(103.0, 0.00005, 0.004, n=400, seed=6),
        "us10y_real": _make_level_series(1.8, 0.001, 0.03, n=400, seed=7),
        "us10y": _make_level_series(4.2, 0.001, 0.04, n=400, seed=8),
        "hy_spread": _make_level_series(3.5, 0.0, 0.08, n=400, seed=9),
        "vix": _make_level_series(16.0, 0.0, 1.2, n=400, seed=10),
    }
    res = compute_ratios_and_valuations(pts)
    ids = {r["id"] for r in res["ratios"]}
    assert "gold_silver" in ids
    assert "copper_gold" in ids
    assert "brent_wti" in ids
    assert res["gold_valuation"]["current_fair_value"] > 0
    assert "cu_au_z" in res["cu_au_vs_10y"]
    assert "dislocation" in res["credit_vs_vol"]


def test_macro_cycles_computation():
    pts = {
        "copper": _make_series(4.0, 0.0004, 0.01, n=400, seed=11),
        "gold": _make_series(2000.0, 0.0002, 0.01, n=400, seed=12),
        "xly": _make_series(180.0, 0.0005, 0.012, n=400, seed=13),
        "xlp": _make_series(75.0, 0.0002, 0.007, n=400, seed=14),
        "brent": _make_series(80.0, 0.0001, 0.015, n=400, seed=15),
        "spx": _make_series(5000.0, 0.0004, 0.01, n=400, seed=16),
        "us2s10s": _make_level_series(0.1, 0.001, 0.03, n=400, seed=17),
        "hy_spread": _make_level_series(3.5, 0.0, 0.05, n=400, seed=18),
        "us10y_breakeven": _make_level_series(2.3, 0.0, 0.02, n=400, seed=19),
        "us10y_real": _make_level_series(1.9, 0.0, 0.02, n=400, seed=20),
        "us10y": _make_level_series(4.2, 0.0, 0.03, n=400, seed=21),
    }
    res = compute_macro_cycles(pts)
    assert res["current_quadrant"] in {"Recovery", "Overheat", "Stagflation", "Reflation"}
    assert len(res["trail"]) > 0
    assert len(res["quadrant_returns"]) == 4
    assert len(res["gauges"]) == 4
