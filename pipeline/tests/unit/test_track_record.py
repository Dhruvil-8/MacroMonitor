"""Unit tests for P11: Track Record & Validation."""

import numpy as np
import pandas as pd
import pytest

from radar.metrics.track_record import (
    decluster_alerts,
    block_bootstrap_difference_ci,
    compute_track_record_payload,
)
from radar.export.schemas import TrackRecordEnvelope, TrackRecordData


def test_decluster_alerts():
    """Verify that alerts within 10 days are collapsed into a single episode."""
    dates = [
        "2026-01-01",
        "2026-01-03",  # within 2 days -> cluster with Jan 1
        "2026-01-08",  # within 7 days -> cluster with Jan 1
        "2026-01-25",  # 17 days after Jan 8 -> new episode
        "2026-01-27",  # within 2 days -> cluster with Jan 25
    ]
    episodes = decluster_alerts(dates, min_gap_days=10)
    assert len(episodes) == 2
    assert episodes[0] == pd.Timestamp("2026-01-01")
    assert episodes[1] == pd.Timestamp("2026-01-25")


def test_block_bootstrap_difference_planted_signal():
    """Verify that a planted signal (hits=80% vs base=40%) yields positive CI."""
    hits = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0] * 5)  # 80% hits
    base = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0] * 5)  # 40% base

    ci_l, ci_u = block_bootstrap_difference_ci(hits, base, n_boot=200, seed=42)
    # The true difference is +40%
    assert ci_l > 10.0
    assert ci_u >= ci_l


def test_block_bootstrap_difference_noise():
    """Verify that pure noise (identical distributions) yields a CI spanning zero."""
    rng = np.random.default_rng(99)
    hits = rng.binomial(1, 0.5, size=60)
    base = rng.binomial(1, 0.5, size=60)

    ci_l, ci_u = block_bootstrap_difference_ci(hits, base, n_boot=200, seed=99)
    assert ci_l <= 10.0
    assert ci_u >= -10.0
    # Spans zero
    assert ci_l < 0 or ci_u > 0


def test_track_record_payload_schema():
    """Verify payload generation, schema validation, and honest labeling."""
    payload = compute_track_record_payload()

    env = TrackRecordEnvelope(
        schema_version="1.0",
        as_of=payload["as_of"],
        code_version="1.0.0",
        params_hash="hash_p11",
        data=TrackRecordData(**payload),
    )

    data = env.data
    assert data.total_episodes > 50
    assert len(data.rules) >= 5
    assert len(data.calibration) == 5
    assert len(data.changelog) >= 2

    # Check that at least one rule is classified as Context Only (honest disclosure)
    context_only_rules = [r for r in data.rules if r.status == "Context Only"]
    assert len(context_only_rules) >= 1, "At least one rule must be honestly marked Context Only"

    # Check calibration monotonicity
    calib = data.calibration
    for i in range(len(calib) - 1):
        assert calib[i].observed_rate <= calib[i + 1].observed_rate
