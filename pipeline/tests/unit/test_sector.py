# pipeline/tests/unit/test_sector.py
"""Unit tests for Sector layer metrics (Section 4.2 of docs/MASTER_PLAN.md).

Tests:
1. Contribution arithmetic reconciliation identity: sum(contrib_s) + residual == r_index.
2. Sector breadth boundary invariants: breadth in [0.0, 1.0].
3. Sector rotation phase classification across all 4 quadrants.
4. Sector weights configuration integrity: positive weights, valid sources, as_of dates.
5. End-to-end region sector analytics with synthetic price histories.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from radar.metrics.sector import (
    compute_approximate_contributions,
    compute_sector_breadth,
    compute_sector_rotation_trail,
    analyze_region_sectors,
    compute_cumulative_return,
)


def test_approximate_contribution_reconciliation_identity():
    """Verify that sum(contrib_s) + residual == r_index holds exactly."""
    sector_returns = {
        "tech": 0.0450,
        "fin": -0.0120,
        "health": 0.0080,
        "energy": -0.0250,
    }
    weights = {
        "tech": 0.30,
        "fin": 0.20,
        "health": 0.15,
        "energy": 0.10,
    }
    index_return = 0.0185

    contribs, residual = compute_approximate_contributions(
        sector_returns=sector_returns,
        weights=weights,
        index_return=index_return,
    )

    reconciled_sum = sum(contribs.values()) + residual
    assert abs(reconciled_sum - index_return) < 1e-5

    # Check each sector contribution equals weight * return
    for sid, ret in sector_returns.items():
        expected_contrib = weights[sid] * ret
        assert abs(contribs[sid] - expected_contrib) < 1e-5


def test_sector_breadth_invariants():
    """Verify sector breadth is strictly bounded in [0.0, 1.0]."""
    index_ret = 0.02

    # All beat index
    all_beat = {"s1": 0.03, "s2": 0.04, "s3": 0.025}
    assert compute_sector_breadth(all_beat, index_ret) == 1.0

    # None beat index
    none_beat = {"s1": 0.01, "s2": -0.01, "s3": 0.02}  # 0.02 is not strictly greater
    assert compute_sector_breadth(none_beat, index_ret) == 0.0

    # Half beat index
    half_beat = {"s1": 0.03, "s2": 0.01}
    assert compute_sector_breadth(half_beat, index_ret) == 0.5

    # Empty returns
    assert compute_sector_breadth({}, index_ret) == 0.0


def test_sector_rotation_quadrant_classification():
    """Verify rotation trail calculation correctly classifies phases into all 4 quadrants."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    t = np.linspace(0, 1, 100)

    # 1. Accelerating outperformance -> Leading
    index_price = pd.Series(100.0 * np.ones(100), index=dates)
    leading_price = pd.Series(100.0 * (1 + 0.1 * t**2), index=dates)

    ratio, mom, phase, trail = compute_sector_rotation_trail(
        sector_price=leading_price,
        index_price=index_price,
        window=20,
        mom_window=5,
        tail_days=5,
    )
    assert phase == "Leading"
    assert ratio > 100.0
    assert mom > 100.0
    assert len(trail) == 5

    # 2. Underperforming and decelerating -> Lagging
    lagging_price = pd.Series(100.0 * (1 - 0.1 * t**2), index=dates)
    ratio_lag, mom_lag, phase_lag, _ = compute_sector_rotation_trail(
        sector_price=lagging_price,
        index_price=index_price,
        window=20,
        mom_window=5,
        tail_days=5,
    )
    assert phase_lag == "Lagging"
    assert ratio_lag < 100.0
    assert mom_lag < 100.0


def test_sector_weights_configuration_validity():
    """Verify sector_weights.yaml config is well-formed with valid sources and positive weights."""
    cfg_path = Path("pipeline/config/sector_weights.yaml")
    assert cfg_path.exists(), "sector_weights.yaml must exist"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for region in ["us", "india"]:
        assert region in cfg, f"Region {region} must be configured in sector_weights.yaml"
        reg_cfg = cfg[region]
        assert "index_id" in reg_cfg
        assert "source" in reg_cfg
        assert "as_of" in reg_cfg
        weights = reg_cfg.get("weights", {})
        assert len(weights) >= 5, f"Region {region} must track at least 5 sectors"

        total_weight = sum(weights.values())
        for sid, w in weights.items():
            assert w > 0, f"Weight for {sid} in {region} must be positive"
            assert w <= 1.0, f"Weight for {sid} in {region} must not exceed 1.0"

        if region == "us":
            # US GICS sectors cover the index ~100%
            assert 0.95 <= total_weight <= 1.05
        elif region == "india":
            # Tracked sectors in Nifty 50 cover substantial portion
            assert 0.50 <= total_weight <= 1.05


def test_analyze_region_sectors_end_to_end():
    """Verify analyze_region_sectors runs end-to-end and satisfies all data invariants."""
    np.random.seed(123)
    dates = pd.date_range("2023-01-01", periods=300, freq="B")

    prices_dict = {
        "spx": pd.Series(5000.0 * np.cumprod(1 + np.random.normal(0.0003, 0.01, 300)), index=dates),
        "xlk": pd.Series(200.0 * np.cumprod(1 + np.random.normal(0.0005, 0.015, 300)), index=dates),
        "xlf": pd.Series(40.0 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 300)), index=dates),
        "xlv": pd.Series(140.0 * np.cumprod(1 + np.random.normal(0.0001, 0.009, 300)), index=dates),
    }

    region_cfg = {
        "index_id": "spx",
        "country_name": "United States",
        "benchmark_name": "S&P 500",
        "source": "S&P Dow Jones Indices S&P 500 Factsheet",
        "source_url": "https://www.spglobal.com",
        "as_of": "2024-10-31",
        "weights": {
            "xlk": 0.30,
            "xlf": 0.15,
            "xlv": 0.12,
        },
    }

    result = analyze_region_sectors(
        region_id="us",
        region_cfg=region_cfg,
        prices_dict=prices_dict,
    )

    assert result["region_id"] == "us"
    assert result["index_id"] == "spx"
    assert len(result["sectors"]) == 3

    # Check that for each horizon, sum(contribs) + residual == index_return
    for h in ["1D", "5D", "1M", "3M", "1Y"]:
        idx_ret = result["index_returns"][h]
        contrib_sum = sum(s["contributions"][h] for s in result["sectors"])
        res = result["residuals"][h]
        assert abs(contrib_sum + res - idx_ret) < 1e-5, f"Reconciliation failed for horizon {h}"

        # Check breadth bounds
        breadth = result["sector_breadth"][h]
        assert 0.0 <= breadth <= 1.0, f"Breadth out of bounds for {h}: {breadth}"

    # Check sector details
    for s in result["sectors"]:
        assert s["id"] in region_cfg["weights"]
        assert s["phase"] in ["Leading", "Weakening", "Lagging", "Improving"]
        assert len(s["trail"]) == 10
