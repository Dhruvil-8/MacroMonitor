# pipeline/tests/unit/test_insight_engine.py
from radar.insights.engine import InsightEngine
from radar.insights.rules import evaluate_severity


def test_severity_mapping():
    assert evaluate_severity(92.0) == "low"
    assert evaluate_severity(96.5) == "medium"
    assert evaluate_severity(99.2) == "high"


def test_cooldown_suppresses_rapid_successive_alerts():
    engine = InsightEngine(cooldown_days=10, daily_cap=8)

    # First trigger on day 1 -> Accepted
    obs1 = engine.process_candidate(
        rule_id="vol_shift_v1",
        asset="spx",
        as_of="2026-01-05",
        severity="medium",
        headline="SPX Volatility Shift",
        body="20-day volatility shifted above 95th percentile.",
        metrics={"vol20": 0.22, "percentile": 96.0},
    )
    assert obs1 is not None

    # Immediate second trigger on day 2 -> Suppressed by cooldown
    obs2 = engine.process_candidate(
        rule_id="vol_shift_v1",
        asset="spx",
        as_of="2026-01-06",
        severity="medium",
        headline="SPX Volatility Shift",
        body="20-day volatility shifted above 95th percentile.",
        metrics={"vol20": 0.23, "percentile": 97.0},
    )
    assert obs2 is None


def test_daily_cap_limits_alert_volume():
    engine = InsightEngine(cooldown_days=10, daily_cap=3)

    candidates = [
        {
            "rule_id": "vol_shift_v1",
            "asset": f"asset_{i}",
            "as_of": "2026-01-05",
            "severity": "high" if i == 0 else "medium",
            "headline": f"Asset {i} Alert",
            "body": "Detail",
            "metrics": {},
        }
        for i in range(5)
    ]

    emitted = []
    for c in candidates:
        res = engine.process_candidate(**c)
        if res:
            emitted.append(res)

    assert len(emitted) == 3
    # First candidate (highest severity) must be present
    assert emitted[0]["asset"] == "asset_0"


def test_batch_processing_prioritizes_high_severity():
    engine = InsightEngine(cooldown_days=10, daily_cap=2)

    candidates = [
        {
            "rule_id": "vol_shift_v1",
            "asset": "asset_low",
            "as_of": "2026-01-05",
            "severity": "low",
            "headline": "Low alert",
            "body": "Detail",
            "metrics": {},
        },
        {
            "rule_id": "vol_shift_v1",
            "asset": "asset_high",
            "as_of": "2026-01-05",
            "severity": "high",
            "headline": "High alert",
            "body": "Detail",
            "metrics": {},
        },
        {
            "rule_id": "vol_shift_v1",
            "asset": "asset_medium",
            "as_of": "2026-01-05",
            "severity": "medium",
            "headline": "Medium alert",
            "body": "Detail",
            "metrics": {},
        },
    ]

    emitted = engine.process_candidates_batch(candidates)
    assert len(emitted) == 2
    # High and medium must win over low
    severities = [obs["severity"] for obs in emitted]
    assert "high" in severities
    assert "medium" in severities
    assert "low" not in severities

