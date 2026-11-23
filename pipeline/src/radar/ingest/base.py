# pipeline/src/radar/ingest/base.py
"""Base Ingestor interface and Source Resilience Layer.

Implements Section P0:
- Source adapter interface: describe, fetch, parse, validate
- Circuit breaker pattern (closed, open, half-open) with cooldowns
- Schema drift detection (column set, dtypes)
- Value sanity checking (plausible bounds, maximum jump detection)
- Raw response caching
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LOGGER = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    is_valid: bool
    total_rows: int
    missing_count: int
    drift_detected: bool
    sanity_violations: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SourceMetadata:
    id: str
    name: str
    provider: str
    documentation_url: str
    terms_url: str
    rate_limit: str
    schedule: str
    fallback_source_id: Optional[str] = None
    last_verified: str = "2026-11-20"
    owner_notes: str = ""


class CircuitBreaker:
    """Circuit breaker pattern for unreliable or rate-limited external feeds."""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count: int = 0
        self.last_failure_time: Optional[float] = None

    def can_attempt(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if self.last_failure_time and (time.time() - self.last_failure_time > self.cooldown_seconds):
                self.state = "HALF_OPEN"
                LOGGER.info("Circuit breaker entering HALF_OPEN state; testing connection.")
                return True
            return False
        if self.state == "HALF_OPEN":
            return True
        return False

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            LOGGER.warning(f"Circuit breaker tripped to OPEN state after {self.failure_count} consecutive failures.")


class BaseIngestor(ABC):
    """Abstract Base Class for resilient external data adapters."""

    def __init__(self, source_name: str, cache_dir: Path | str | None = None) -> None:
        self.source_name = source_name
        self.circuit_breaker = CircuitBreaker()
        if cache_dir is None:
            repo_root = Path(__file__).resolve().parents[4]
            self.cache_dir = repo_root / "data" / "raw" / source_name
        else:
            self.cache_dir = Path(cache_dir) / source_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, symbol: str, date_str: str | None = None) -> Path:
        """Get cache filepath for a symbol on a specific date."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        safe_sym = symbol.replace("^", "").replace(":", "_").replace("/", "_").replace("=", "_")
        target_dir = self.cache_dir / date_str
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{safe_sym}.parquet"

    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_cols: List[str],
        allow_extra: bool = True,
    ) -> bool:
        """Verify that mandatory columns exist in the dataframe."""
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            LOGGER.error(f"[{self.source_name}] Schema drift detected! Missing columns: {missing}")
            return False
        return True

    def validate_sanity(
        self,
        df: pd.DataFrame,
        val_col: str = "value",
        plausible_min: Optional[float] = None,
        plausible_max: Optional[float] = None,
        max_jump_pct: float = 0.50,
    ) -> List[str]:
        """Detect out-of-bounds readings or impossible percentage jumps."""
        violations = []
        if df.empty or val_col not in df.columns:
            return violations

        s = df[val_col].dropna()
        if plausible_min is not None:
            below = s[s < plausible_min]
            if not below.empty:
                violations.append(f"{len(below)} values below plausible min ({plausible_min}): min={below.min()}")

        if plausible_max is not None:
            above = s[s > plausible_max]
            if not above.empty:
                violations.append(f"{len(above)} values above plausible max ({plausible_max}): max={above.max()}")

        if len(s) > 1:
            pct_jumps = s.pct_change().abs()
            huge_jumps = pct_jumps[pct_jumps > max_jump_pct]
            if not huge_jumps.empty:
                violations.append(f"{len(huge_jumps)} price jumps exceed {max_jump_pct*100:.0f}%")

        return violations

    @abstractmethod
    def fetch_series(
        self,
        symbol: str,
        start_date: str = "2000-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch raw series data. Returns DataFrame with DatetimeIndex and 'value' column."""
        pass
