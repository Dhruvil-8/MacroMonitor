# pipeline/tests/unit/test_resilience.py
import time
import pandas as pd
import pytest
from radar.ingest.base import CircuitBreaker, BaseIngestor, SourceMetadata


class MockIngestor(BaseIngestor):
    def fetch_series(self, symbol: str, start_date: str = "2000-01-01", end_date: str | None = None) -> pd.DataFrame:
        return pd.DataFrame({"value": [10.0, 11.0]}, index=pd.date_range("2026-01-01", periods=2))


def test_circuit_breaker_transitions():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.4)
    assert cb.state == "CLOSED"
    assert cb.can_attempt() is True

    cb.record_failure()
    assert cb.state == "CLOSED"
    assert cb.can_attempt() is True

    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.can_attempt() is False

    time.sleep(0.45)
    assert cb.can_attempt() is True
    assert cb.state == "HALF_OPEN"

    cb.record_success()
    assert cb.state == "CLOSED"
    assert cb.failure_count == 0


def test_schema_drift_detection(tmp_path):
    ingestor = MockIngestor("test_source", cache_dir=tmp_path)
    df_good = pd.DataFrame({"value": [1.0, 2.0], "volume": [100, 200]})
    assert ingestor.validate_schema(df_good, ["value", "volume"]) is True

    df_drifted = pd.DataFrame({"val": [1.0, 2.0]})
    assert ingestor.validate_schema(df_drifted, ["value"]) is False


def test_sanity_validation(tmp_path):
    ingestor = MockIngestor("test_source", cache_dir=tmp_path)
    dates = pd.date_range("2026-01-01", periods=3)
    df_wild = pd.DataFrame({"value": [100.0, 300.0, -50.0]}, index=dates)

    violations = ingestor.validate_sanity(df_wild, plausible_min=0.0, plausible_max=200.0, max_jump_pct=0.5)
    assert len(violations) >= 2
    assert any("below plausible min" in v for v in violations)
    assert any("exceed 50%" in v for v in violations)
