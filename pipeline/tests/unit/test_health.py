# pipeline/tests/unit/test_health.py
import datetime
import numpy as np
import pandas as pd
import pytest

from radar.clean.health import evaluate_series_health, evaluate_pipeline_health


def test_health_ok_for_fresh_series():
    as_of = pd.Timestamp("2026-01-20")
    dates = pd.date_range("2025-01-01", "2026-01-20", freq="B")
    df = pd.DataFrame({
        "value": np.linspace(100, 150, len(dates)),
        "is_filled": [False] * len(dates)
    }, index=dates)

    health = evaluate_series_health("spx", df, as_of=as_of, stale_after_hours=36)
    assert health["series_id"] == "spx"
    assert health["status"] == "ok"
    assert health["coverage_pct"] > 0.95
    assert health["fill_ratio"] == 0.0


def test_health_stale_for_old_series():
    as_of = pd.Timestamp("2026-01-20")
    # Last point is 10 days old
    dates = pd.date_range("2025-01-01", "2026-01-10", freq="B")
    df = pd.DataFrame({
        "value": np.linspace(100, 150, len(dates)),
        "is_filled": [False] * len(dates)
    }, index=dates)

    health = evaluate_series_health("stale_asset", df, as_of=as_of, stale_after_hours=36)
    assert health["series_id"] == "stale_asset"
    assert health["status"] == "stale"


def test_health_failed_for_empty_or_low_coverage():
    as_of = pd.Timestamp("2026-01-20")
    df_empty = pd.DataFrame({"value": [], "is_filled": []}, index=pd.DatetimeIndex([]))

    health = evaluate_series_health("empty_asset", df_empty, as_of=as_of)
    assert health["status"] == "failed"


def test_pipeline_abort_threshold():
    health_reports = [
        {"series_id": f"s_{i}", "status": "ok"} for i in range(80)
    ] + [
        {"series_id": f"s_{i}", "status": "stale"} for i in range(80, 100)
    ]  # 20% stale -> exactly at boundary

    status, stale_pct = evaluate_pipeline_health(health_reports, max_stale_pct=0.20)
    assert status == "ok"

    # Add 1 more stale series -> 21 / 101 > 20%
    health_reports.append({"series_id": "s_extra", "status": "stale"})
    status, stale_pct = evaluate_pipeline_health(health_reports, max_stale_pct=0.20)
    assert status == "failed"
