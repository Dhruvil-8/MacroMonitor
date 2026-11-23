# pipeline/tests/unit/test_cleaner.py
import numpy as np
import pandas as pd
import pytest

from radar.clean.cleaner import clean_series, apply_forward_fill, filter_jumps


def test_forward_fill_limit():
    """Verify that forward-fill fills at most max_ffill_days and sets is_filled flag."""
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    # Gap of 2 days: index 1 and 2 are NaN
    # Gap of 4 days: index 5, 6, 7, 8 are NaN
    values = [100.0, np.nan, np.nan, 103.0, 104.0, np.nan, np.nan, np.nan, np.nan, 110.0]
    s = pd.Series(values, index=dates)

    filled, is_filled = apply_forward_fill(s, max_ffill_days=3)

    # Gap 1 (2 days): both should be filled with 100.0 and flagged
    assert filled.iloc[1] == 100.0
    assert filled.iloc[2] == 100.0
    assert is_filled.iloc[1] is True or is_filled.iloc[1] == 1
    assert is_filled.iloc[2] is True or is_filled.iloc[2] == 1

    # Original values should NOT be flagged as filled
    assert not is_filled.iloc[0]
    assert not is_filled.iloc[3]

    # Gap 2 (4 days): first 3 should be filled with 104.0, 4th must remain NaN
    assert filled.iloc[5] == 104.0
    assert filled.iloc[6] == 104.0
    assert filled.iloc[7] == 104.0
    assert is_filled.iloc[5] is True or is_filled.iloc[5] == 1
    assert is_filled.iloc[6] is True or is_filled.iloc[6] == 1
    assert is_filled.iloc[7] is True or is_filled.iloc[7] == 1

    assert np.isnan(filled.iloc[8])
    assert not is_filled.iloc[8]


def test_jump_filter_rejects_extreme_outliers():
    """Verify that an extreme 20-sigma price spike is detected and rejected."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=200, freq="B")
    # Series with std ~ 1% daily changes
    changes = np.random.normal(0, 0.01, size=200)
    prices = 100.0 * np.exp(np.cumsum(changes))
    # Inject a 50% erroneous jump at index 100
    prices[100] = prices[99] * 1.50

    s = pd.Series(prices, index=dates)
    filtered = filter_jumps(s, jump_sigma=8.0)

    # Point 100 should be rejected to NaN
    assert np.isnan(filtered.iloc[100])
    # Neighboring points should remain valid
    assert not np.isnan(filtered.iloc[99])
    assert not np.isnan(filtered.iloc[101])


def test_clean_series_end_to_end():
    """Test full cleaning pipeline: negative price removal, calendar alignment, jump filter and ffill."""
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-09"])
    values = [100.0, -5.0, 101.0, 102.0]  # negative price is invalid
    df_raw = pd.DataFrame({"value": values}, index=dates)

    clean_df = clean_series(
        df_raw,
        series_id="test_asset",
        unit="index",
        calendar_freq="B",
        max_ffill_days=3,
        jump_sigma=8.0,
    )

    # Must contain expected columns
    assert "value" in clean_df.columns
    assert "is_filled" in clean_df.columns
    # Jan 06 was negative -> should be filled from Jan 05
    assert clean_df.loc["2026-01-06", "value"] == 100.0
    assert clean_df.loc["2026-01-06", "is_filled"] is True or clean_df.loc["2026-01-06", "is_filled"] == 1
    # Jan 08 was missing -> should be filled from Jan 07
    assert clean_df.loc["2026-01-08", "value"] == 101.0
    assert clean_df.loc["2026-01-08", "is_filled"] is True or clean_df.loc["2026-01-08", "is_filled"] == 1
