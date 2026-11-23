# pipeline/src/radar/clean/health.py
"""Data health monitoring and staleness evaluation.

Implements Section 4.2 (series_health schema) and Section 7.2 (failure policy).
"""

import datetime
from typing import Any
import numpy as np
import pandas as pd


def evaluate_series_health(
    series_id: str,
    df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    stale_after_hours: float = 36.0,
) -> dict[str, Any]:
    """Evaluate health metrics and status for a cleaned series.

    Returns
    -------
    dict with keys:
        - series_id: str
        - last_point: str (YYYY-MM-DD) or None
        - coverage_pct: float
        - fill_ratio: float
        - status: 'ok' | 'stale' | 'failed'
        - checked_at: str (ISO)
    """
    now = as_of if as_of is not None else pd.Timestamp.now(tz=None)

    if df.empty or "value" not in df.columns or df["value"].dropna().empty:
        return {
            "series_id": series_id,
            "last_point": None,
            "coverage_pct": 0.0,
            "fill_ratio": 0.0,
            "status": "failed",
            "checked_at": now.isoformat(),
        }

    valid_mask = df["value"].notna()
    last_point_ts = df.index[valid_mask].max()
    last_point_str = last_point_ts.strftime("%Y-%m-%d")

    # Coverage percentage
    total_expected = len(df)
    total_valid = valid_mask.sum()
    coverage_pct = float(total_valid / total_expected) if total_expected > 0 else 0.0

    # Fill ratio
    if "is_filled" in df.columns:
        total_filled = (df["is_filled"] & valid_mask).sum()
        fill_ratio = float(total_filled / total_valid) if total_valid > 0 else 0.0
    else:
        fill_ratio = 0.0

    # Staleness evaluation:
    # Account for weekends when evaluating hours gap:
    # Gap in business hours or calendar hours
    calendar_diff = now - last_point_ts
    diff_hours = calendar_diff.total_seconds() / 3600.0

    # If day difference is >= 2 calendar days, check if it's over a weekend
    # E.g. Friday to Monday is ~72 calendar hours but ~0 business days.
    # Count business days between last_point_ts and now
    b_days_gap = len(pd.date_range(last_point_ts, now, freq="B")) - 1

    if coverage_pct < 0.50:
        status = "failed"
    elif b_days_gap > 3 or (b_days_gap > 1 and diff_hours > stale_after_hours + 48):
        status = "stale"
    else:
        status = "ok"

    return {
        "series_id": series_id,
        "last_point": last_point_str,
        "coverage_pct": round(coverage_pct, 4),
        "fill_ratio": round(fill_ratio, 4),
        "status": status,
        "checked_at": now.isoformat(),
    }


def evaluate_pipeline_health(
    health_reports: list[dict[str, Any]], max_stale_pct: float = 0.20
) -> tuple[str, float]:
    """Evaluate overall pipeline health across all series.

    Returns
    -------
    tuple[str, float]
        (overall_status, stale_pct)
        where overall_status is 'ok' or 'failed'.
    """
    if not health_reports:
        return "failed", 1.0

    total = len(health_reports)
    stale_or_failed = sum(1 for r in health_reports if r.get("status") in {"stale", "failed"})
    stale_pct = float(stale_or_failed / total)

    if stale_pct > max_stale_pct:
        return "failed", round(stale_pct, 4)
    return "ok", round(stale_pct, 4)
