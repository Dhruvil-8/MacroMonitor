import numpy as np
import pandas as pd
import pytest

from radar.metrics.india_flows import (
    classify_oi_buildup,
    compute_fii_dii_rolling_flows,
    compute_put_call_ratio,
    compute_delivery_metrics,
)


def test_classify_oi_buildup_all_regimes():
    """Verify all 4 open interest buildup regimes."""
    # 1. Price UP, OI UP -> Long Buildup
    assert classify_oi_buildup(0.5, 2.0) == "Long Buildup"

    # 2. Price DOWN, OI UP -> Short Buildup
    assert classify_oi_buildup(-1.2, 3.5) == "Short Buildup"

    # 3. Price UP, OI DOWN -> Short Covering
    assert classify_oi_buildup(0.8, -1.5) == "Short Covering"

    # 4. Price DOWN, OI DOWN -> Long Unwinding
    assert classify_oi_buildup(-0.7, -2.4) == "Long Unwinding"

    # Flat edge cases
    assert classify_oi_buildup(0.0, 0.0) == "Neutral"


def test_hand_checked_dates_fixtures():
    """Verify hand-checked dates from Section 3 of technical specification."""
    # Date 1: 2024-06-04 (Election Volatility)
    # Price -5.93%, OI +8.4% -> Short Buildup
    assert classify_oi_buildup(-5.93, 8.4) == "Short Buildup"

    # Date 2: 2024-11-04 (Consolidation Pullback)
    # Price -1.27%, OI -2.1% -> Long Unwinding
    assert classify_oi_buildup(-1.27, -2.1) == "Long Unwinding"

    # Date 3: 2026-11-20 (Current Baseline)
    # Price +0.42%, OI +1.8% -> Long Buildup
    assert classify_oi_buildup(0.42, 1.8) == "Long Buildup"


def test_fii_dii_rolling_flows():
    """Verify multi-horizon rolling sums and expanding percentile calculations."""
    dates = pd.bdate_range("2026-01-01", periods=30)
    # Constant daily net flow of 100 Cr
    fii_daily = pd.Series(100.0, index=dates)
    dii_daily = pd.Series(150.0, index=dates)

    df_flows = compute_fii_dii_rolling_flows(fii_daily, dii_daily, windows=[5, 20])
    assert "fii_5d" in df_flows.columns
    assert "fii_20d" in df_flows.columns
    assert "dii_5d" in df_flows.columns
    assert "dii_20d" in df_flows.columns
    assert "fii_20d_pct" in df_flows.columns

    # 5-day sum of 100 should be 500
    assert np.isclose(df_flows["fii_5d"].iloc[-1], 500.0)
    # 20-day sum of 100 should be 2000
    assert np.isclose(df_flows["fii_20d"].iloc[-1], 2000.0)
    # 5-day sum of 150 should be 750
    assert np.isclose(df_flows["dii_5d"].iloc[-1], 750.0)


def test_put_call_ratio():
    """Verify PCR calculation and threshold properties."""
    put_oi = 1500000.0
    call_oi = 1000000.0
    pcr = compute_put_call_ratio(put_oi, call_oi)
    assert pcr == 1.5

    # Zero division protection
    assert compute_put_call_ratio(1000.0, 0.0) == 0.0


def test_delivery_metrics():
    """Verify volume-weighted delivery percentage calculation."""
    traded = pd.Series([1000, 2000, 3000])
    delivered = pd.Series([500, 800, 1500])  # Total traded = 6000, total delivered = 2800 -> 46.67%
    res = compute_delivery_metrics(delivered, traded)
    assert np.isclose(res["delivery_pct"], 46.67, atol=1e-2)
