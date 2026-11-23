# pipeline/tests/unit/test_store.py
from pathlib import Path
import pandas as pd
import pytest

from radar.db.store import DataStore


def test_local_datastore_roundtrip(tmp_path):
    store = DataStore(base_dir=tmp_path)

    # 1. Upsert metadata
    meta = [
        {
            "id": "spx",
            "name": "S&P 500",
            "source": "yahoo",
            "source_symbol": "^GSPC",
            "group": "equity_index",
            "region": "US",
            "unit": "index",
            "frequency": "D",
            "transform_default": "log_return",
            "start_date": "2000-01-01",
        }
    ]
    store.upsert_meta(meta)
    read_meta = store.get_meta("spx")
    assert read_meta is not None
    assert read_meta["name"] == "S&P 500"

    # 2. Upsert points
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    df = pd.DataFrame({
        "value": [100.0, 101.0, 102.0, 103.0, 104.0],
        "is_filled": [False, False, True, False, False]
    }, index=dates)
    store.upsert_points("spx", df)

    read_points = store.get_points("spx")
    assert len(read_points) == 5
    assert "value" in read_points.columns
    assert "is_filled" in read_points.columns
    assert read_points.iloc[2]["is_filled"] is True or read_points.iloc[2]["is_filled"] == 1

    # 3. Upsert health
    health = [{
        "series_id": "spx",
        "last_point": "2026-01-07",
        "coverage_pct": 0.99,
        "fill_ratio": 0.20,
        "status": "ok",
    }]
    store.upsert_health(health)
    read_health = store.get_health()
    assert len(read_health) == 1
    assert read_health[0]["status"] == "ok"
