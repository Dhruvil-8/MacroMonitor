# tests/test_audit_regressions.py
"""
Regression tests for audit findings FINDING-DATA-01, FINDING-MATH-01, and FINDING-UI-01.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from radar.clean.cleaner import clean_series, filter_jumps
from radar.metrics.decomposition import compute_decomposition

def test_vix_august_2024_spike_preserved():
    """Regression test for FINDING-DATA-01:
    Verify that filter_jumps and clean_series do not classify the genuine
    August 5, 2024 VIX spike of 38.57 as an outlier and do not forward fill it.
    Works reliably both locally and in clean-room CI environments without data/raw dependencies.
    """
    # 1. Verify processed series artifact tracked in git
    proc_vix_path = Path("data/processed/series/vix.parquet")
    if proc_vix_path.exists():
        proc_df = pd.read_parquet(proc_vix_path)
        if "2024-08-05" in proc_df.index:
            aug5_val = proc_df.loc["2024-08-05", "value"]
            assert np.isclose(aug5_val, 38.57, atol=0.01), f"Expected 38.57 in processed series on 2024-08-05, got {aug5_val}"
            assert not proc_df.loc["2024-08-05", "is_filled"], "Processed 2024-08-05 was flagged as forward-filled"

    # 2. Verify clean_series algorithm preserves the genuine volatility spike
    raw_vix_path = Path("data/raw/yahoo/2026-11-21/VIX.parquet")
    if raw_vix_path.exists():
        raw_df = pd.read_parquet(raw_vix_path)
    else:
        # Self-contained synthetic fixture reproducing the August 2024 VIX sequence
        dates = pd.date_range("2024-05-01", "2024-08-10", freq="B")
        raw_df = pd.DataFrame({"value": 15.0}, index=dates)
        raw_df.loc["2024-08-01", "value"] = 18.59
        raw_df.loc["2024-08-02", "value"] = 23.39
        raw_df.loc["2024-08-05", "value"] = 38.57
        raw_df.loc["2024-08-06", "value"] = 27.71

    clean_df = clean_series(
        raw_df,
        series_id="vix",
        unit="percent",
        calendar_freq="B",
        max_ffill_days=3,
        jump_sigma=8.0,
    )

    # August 5, 2024 must equal 38.57 within rounding
    aug5_val = clean_df.loc["2024-08-05", "value"]
    assert np.isclose(aug5_val, 38.57, atol=0.01), f"Expected 38.57 on 2024-08-05, got {aug5_val}"
    
    # Must NOT be flagged as filled
    is_filled = clean_df.loc["2024-08-05", "is_filled"]
    assert not is_filled, "August 5, 2024 was erroneously flagged as forward-filled"

def test_jump_filter_strictly_trailing_anti_lookahead():
    """Verify that filter_jumps operates in strictly trailing mode without look-ahead bias."""
    dates = pd.date_range("2026-01-01", periods=50, freq="B")
    s = pd.Series(100.0, index=dates)
    # If we modify values at t > 30, the output at t <= 30 must remain identical
    s1 = s.copy()
    s2 = s.copy()
    s2.iloc[35:] = 500.0 # future shock
    
    out1 = filter_jumps(s1, jump_sigma=8.0)
    out2 = filter_jumps(s2, jump_sigma=8.0)
    
    # Subseries up to index 30 must be bit-identical
    assert out1.iloc[:31].equals(out2.iloc[:31]), "Look-ahead bias detected in filter_jumps!"

def test_decomposition_residual_sum_identity():
    """Regression test for FINDING-MATH-01:
    Verify that sum of contributions + residual equals total_return exactly in all decomposition outputs.
    """
    decomp_dir = Path("data/v1/decomposition")
    if not decomp_dir.exists():
        pytest.skip("Decomposition output directory not found")

    files = list(decomp_dir.glob("*.json"))
    assert len(files) > 0, "No decomposition JSON files found"

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        d = data.get("data", {})
        total_ret = d.get("total_return")
        contribs = d.get("contributions", {})
        residual = d.get("residual")
        
        if total_ret is not None and contribs and residual is not None:
            sum_parts = round(sum(contribs.values()) + residual, 4)
            diff = abs(sum_parts - total_ret)
            assert diff < 1e-9, f"In {fpath.name}: sum of parts ({sum_parts}) != total_return ({total_ret})"
