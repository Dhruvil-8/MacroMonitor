import numpy as np
import pandas as pd
import pytest

from radar.metrics.stress import (
    compute_stress_composite,
    classify_stress_regime,
    validate_crisis_episodes,
)
from radar.insights.briefing import generate_daily_briefing


def test_stress_composite_zero_imputation():
    """Verify composite stress score is the exact mean of available components without imputation."""
    components = {
        "equity": pd.Series([50.0, 80.0, 95.0]),
        "rates": pd.Series([40.0, 70.0, 90.0]),
        "fx": pd.Series([30.0, 60.0, 85.0]),
        "commodity": pd.Series([np.nan, 50.0, 80.0]),  # Missing at t=0
        "correlation": pd.Series([np.nan, np.nan, 75.0]),  # Missing at t=0, 1
    }

    df_stress = compute_stress_composite(components, min_components=2)
    # At t=0: Available are equity(50), rates(40), fx(30). Mean = (50+40+30)/3 = 40.0
    assert np.isclose(df_stress["composite"].iloc[0], 40.0)
    # At t=1: Available are equity(80), rates(70), fx(60), comm(50). Mean = 65.0
    assert np.isclose(df_stress["composite"].iloc[1], 65.0)
    # At t=2: Available are all 5: (95+90+85+80+75)/5 = 85.0
    assert np.isclose(df_stress["composite"].iloc[2], 85.0)


def test_stress_regimes_classification():
    """Verify stress regime classification thresholds."""
    assert classify_stress_regime(25.0) == "Low Stress"
    assert classify_stress_regime(55.0) == "Normal Market Conditions"
    assert classify_stress_regime(75.0) == "Elevated Stress"
    assert classify_stress_regime(92.0) == "High Macro Stress"


def test_validate_crisis_episodes():
    """Verify historical crisis episodes pass threshold checks."""
    episodes = [
        {"name": "GFC Peak", "date": "2008-10-15", "stress": 92.5},
        {"name": "COVID Crash", "date": "2020-03-16", "stress": 94.8},
        {"name": "August 2024 Volatility Spike", "date": "2024-08-05", "stress": 71.2},
    ]
    results = validate_crisis_episodes(episodes)
    assert results["all_passed"] is True
    assert len(results["validated"]) == 3


def test_briefing_generator_safety():
    """Verify briefing generator produces structured text with zero forbidden claims."""
    briefing = generate_daily_briefing(
        as_of="2026-11-20",
        stress_score=52.4,
        stress_regime="Normal Market Conditions",
        component_scores={"equity": 58.2, "rates": 44.1, "fx": 51.0, "commodity": 49.5},
        top_observations=[
            {"asset": "SPX", "type": "volatility_shift", "headline": "S&P 500 Realized Volatility at 35th percentile."}
        ],
        net_liquidity_chg_4w=23.04,
        fii_net_1d=1250.40,
        dii_net_1d=1840.15,
        stablecoins_total_bn=282.58,
    )

    assert "headline" in briefing
    assert "executive_summary" in briefing
    assert "sections" in briefing

    # Safety scan: reject forbidden trading words with word boundaries
    import re
    forbidden = ["buy", "sell", "target", "will rally", "will crash", "accumulate", "recommendation"]
    full_text = f"{briefing['headline']} {briefing['executive_summary']} {str(briefing['sections'])}".lower()
    for word in forbidden:
        match = re.search(rf"\b{word}\b", full_text)
        assert not match, f"Forbidden term '{word}' found in briefing!"
