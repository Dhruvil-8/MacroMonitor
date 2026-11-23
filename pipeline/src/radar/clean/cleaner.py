# pipeline/src/radar/clean/cleaner.py
"""Data cleaning layer.

Implements Section 7.1 and docs/spec/s00-data-foundation.md:
- Business calendar alignment
- Forward-fill limitation (at most max_ffill_days) with is_filled tracking
- Outlier detection and jump rejection via robust MAD z-scores
- Non-negative validation for price series
"""

import numpy as np
import pandas as pd


def apply_forward_fill(
    s: pd.Series, max_ffill_days: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Forward-fill missing points up to `max_ffill_days`.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (filled_series, is_filled_boolean_mask)
    """
    is_originally_nan = s.isna()

    # Compute consecutive NaN counts
    mask = is_originally_nan
    cum_nans = mask.cumsum()
    nan_consecutive_count = cum_nans - cum_nans.where(~mask).ffill().fillna(0)

    # Allow forward fill only where consecutive NaN count <= max_ffill_days
    ffilled = s.ffill()
    allowed_fill = nan_consecutive_count <= max_ffill_days

    result = s.copy()
    can_fill = is_originally_nan & allowed_fill
    result[can_fill] = ffilled[can_fill]

    is_filled_mask = can_fill
    return result, is_filled_mask


def filter_jumps(
    s: pd.Series, jump_sigma: float = 8.0, window: int = 60
) -> pd.Series:
    """Detect and replace extreme jumps with NaN using rolling Median Absolute Deviation (MAD).

    Points where absolute deviation from rolling median exceeds `jump_sigma * (1.4826 * MAD)`
    are flagged as outliers and replaced with NaN.
    """
    s_clean = s.copy()
    roll_median = s_clean.rolling(window=window, min_periods=10, center=True).median()
    # Fallback to non-centered for the boundary edges
    roll_median = roll_median.fillna(s_clean.rolling(window=window, min_periods=5).median())

    abs_dev = (s_clean - roll_median).abs()
    roll_mad = abs_dev.rolling(window=window, min_periods=10, center=True).median()
    roll_mad = roll_mad.fillna(abs_dev.rolling(window=window, min_periods=5).median())

    # Scale factor for normal consistency
    norm_mad = 1.4826 * roll_mad.replace(0, np.nan)
    z_score = (abs_dev / norm_mad).fillna(0)

    outliers = z_score > jump_sigma
    s_clean[outliers] = np.nan
    return s_clean


def clean_series(
    df: pd.DataFrame,
    series_id: str,
    unit: str = "index",
    calendar_freq: str = "B",
    max_ffill_days: int = 3,
    jump_sigma: float = 8.0,
) -> pd.DataFrame:
    """Clean a raw single series dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and 'value' column.
    series_id : str
        Series identifier.
    unit : str
        Unit of the series. If 'index', 'usd', or 'inr', prices <= 0 are rejected.
    calendar_freq : str
        Business day frequency (e.g. 'B').
    max_ffill_days : int
        Maximum consecutive missing days to forward-fill.
    jump_sigma : float
        Outlier threshold for jump filter.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with ['value', 'is_filled'] columns indexed by DatetimeIndex.
    """
    if df.empty or "value" not in df.columns:
        return pd.DataFrame(columns=["value", "is_filled"], index=pd.DatetimeIndex([]))

    # Standardize index to timezone-naive dates
    df_clean = df.copy()
    df_clean.index = pd.to_datetime(df_clean.index).tz_localize(None).normalize()
    df_clean = df_clean[~df_clean.index.duplicated(keep="last")]
    df_clean = df_clean.sort_index()

    # Reject non-positive values for price/index assets
    if unit in {"index", "usd", "inr"}:
        df_clean.loc[df_clean["value"] <= 0, "value"] = np.nan

    # Reindex to full business day calendar
    start_date = df_clean.index.min()
    end_date = df_clean.index.max()
    full_calendar = pd.date_range(start_date, end_date, freq=calendar_freq)
    reindexed = df_clean["value"].reindex(full_calendar)

    # Jump filter
    jump_filtered = filter_jumps(reindexed, jump_sigma=jump_sigma)

    # Forward fill with limits and tracking
    filled, is_filled = apply_forward_fill(jump_filtered, max_ffill_days=max_ffill_days)

    out_df = pd.DataFrame({
        "value": filled,
        "is_filled": is_filled
    }, index=full_calendar)

    return out_df
