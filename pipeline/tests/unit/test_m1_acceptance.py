# pipeline/tests/unit/test_m1_acceptance.py
"""Acceptance criteria test for Milestone M1 (Data Foundation).

Acceptance Criteria (Section 17):
- 60 series loaded
- series_health populated
- Simulated stale series handled
- No secrets in repo
"""

import os
from pathlib import Path
import re
import pandas as pd
import pytest

from radar.clean.health import evaluate_pipeline_health, evaluate_series_health
from radar.db.store import DataStore


def test_m1_acceptance_series_configured_and_meta_loaded():
    store = DataStore()
    meta_list = store.get_meta()
    assert meta_list is not None
    assert len(meta_list) >= 60, f"Expected at least 60 series loaded, got {len(meta_list)}"


def test_m1_acceptance_simulated_stale_series():
    # Simulate a stale series whose last data point is 10 business days in the past
    now = pd.Timestamp.now()
    old_dates = pd.date_range(now - pd.Timedelta(days=60), now - pd.Timedelta(days=14), freq="B")
    df_stale = pd.DataFrame({
        "value": range(len(old_dates)),
        "is_filled": [False] * len(old_dates)
    }, index=old_dates)

    health = evaluate_series_health("simulated_stale", df_stale, as_of=now, stale_after_hours=36.0)
    assert health["status"] == "stale", f"Expected 'stale', got {health['status']}"

    # Verify that pipeline aborts / flags failure if stale series exceed max_stale_pct
    reports = [{"series_id": f"s_{i}", "status": "ok"} for i in range(10)] + [
        {"series_id": f"stale_{i}", "status": "stale"} for i in range(5)
    ]  # 5/15 = 33% > 20%
    status, stale_pct = evaluate_pipeline_health(reports, max_stale_pct=0.20)
    assert status == "failed"
    assert stale_pct > 0.20


def test_m1_acceptance_no_secrets_in_repo():
    repo_root = Path(__file__).resolve().parents[3]

    # Secret patterns
    secret_patterns = [
        re.compile(r"ey[A-Za-z0-9_-]{20,}\.[A-Za-z0-9._-]{20,}"),  # JWT token
        re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9]{20,}['\"]"),
        re.compile(r"(?i)service[_-]?role[_-]?key\s*[:=]\s*['\"][A-Za-z0-9]{20,}['\"]"),
        re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
    ]

    scanned_extensions = {".py", ".ts", ".tsx", ".yaml", ".yml", ".json", ".sql", ".env.example"}
    ignored_dirs = {".git", ".pytest_cache", "node_modules", ".next", "__pycache__"}

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for f in files:
            path = Path(root) / f
            if path.suffix in scanned_extensions:
                content = path.read_text(encoding="utf-8", errors="ignore")
                for pat in secret_patterns:
                    assert not pat.search(content), f"Potential secret matched in {path}"
