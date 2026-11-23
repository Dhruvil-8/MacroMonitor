"""Unit tests for P2: CFTC Positioning and Crowding Ladder."""

from __future__ import annotations

import pytest
from radar.metrics.positioning import compute_cot_contract_metrics, build_positioning_payload


def test_cot_contract_metrics_expanding():
    # Build 60 synthetic weekly rows
    weekly_rows = []
    base_oi = 100000
    for i in range(60):
        # Gradual trend
        longs = 40000 + i * 500
        shorts = 30000
        weekly_rows.append({
            "date": f"2025-{(i%12)+1:02d}-{(i%28)+1:02d}",
            "open_interest": base_oi,
            "noncomm_long": longs,
            "noncomm_short": shorts,
        })

    metrics = compute_cot_contract_metrics(weekly_rows, min_obs=52)
    assert len(metrics) == 60
    # First 52 rows (0..51) should have None because len(sorted_hist) < 52
    assert metrics[0]["percentile"] is None
    assert metrics[51]["percentile"] is None
    # Index 52 has 52 prior observations, so percentile is computed
    assert metrics[52]["percentile"] is not None
    # Latest should be 100th percentile because longs reached maximum
    assert metrics[-1]["percentile"] == 100.0
    assert metrics[-1]["net_pct_oi"] > 0


def test_positioning_payload_hysteresis_and_deltas():
    # Construct 160 weekly rows
    weekly_rows = []
    for i in range(160):
        longs = 50000 if i < 155 else 90000  # Jump to extreme at the end
        shorts = 10000
        weekly_rows.append({
            "date": f"2024-{(i%12)+1:02d}-{(i%28)+1:02d}",
            "open_interest": 100000,
            "noncomm_long": longs,
            "noncomm_short": shorts,
        })

    cot_raw = {
        "contracts": {
            "13874A": weekly_rows  # E-mini S&P 500 CFTC code
        }
    }
    params = {
        "positioning": {
            "min_expanding_weeks": 52,
            "tail_weeks": 156,
        }
    }

    payload = build_positioning_payload(cot_raw, params)
    assert "contracts" in payload
    # Find sp500_emini
    es = next((c for c in payload["contracts"] if c["id"] == "sp500_emini"), None)
    assert es is not None
    assert es["cftc_code"] == "13874A"
    assert es["chg_1w_pct_oi"] is not None
    assert es["chg_4w_pct_oi"] is not None
    assert es["percentile_4w_ago"] is not None
    assert es["lag_label"] == "Positions as of Tuesday, published Friday"
    # Because both latest and prev_1w are at top quantile (>95), hysteresis should trigger extreme
    assert es["is_extreme"] is True
    assert es["extreme_type"] == "crowded_long"
