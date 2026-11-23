import re
from pathlib import Path
import pytest

from radar.insights.briefing import generate_daily_briefing
from radar.export.share_card import render_share_card_svg

FORBIDDEN_TERMS = [
    r"\bbuy\b",
    r"\bsell\b",
    r"\bbullish\b",
    r"\bbearish\b",
    r"\btarget\b",
    r"\bwill rise\b",
    r"\bwill fall\b",
    r"\brecommend\b",
    r"\bsignal to\b",
    r"\boutperform\b",
    r"\bunderperform\b",
]

STOCK_TICKERS = [
    r"\bAAPL\b",
    r"\bMSFT\b",
    r"\bTSLA\b",
    r"\bNVDA\b",
    r"\bRELIANCE\b",
    r"\bTCS\b",
    r"\bINFY\b",
    r"\bHDFCBANK\b",
]


def test_daily_briefing_generation():
    res = generate_daily_briefing(
        as_of="2026-11-20",
        stress_score=48.5,
        stress_regime="Normal Market Conditions",
        component_scores={"equity": 45.0, "rates": 50.0, "fx": 52.0, "commodity": 48.0},
        top_observations=[
            {"asset": "spx", "type": "volatility_shift", "headline": "S&P 500 Realized Volatility at 35th percentile."},
            {"asset": "brent", "type": "commodity_shock", "headline": "Unusual price move recorded in Brent Crude."},
        ],
        net_liquidity_chg_4w=23.04,
        fii_net_1d=1250.40,
        dii_net_1d=1840.15,
        stablecoins_total_bn=282.58,
    )

    assert res["as_of"] == "2026-11-20"
    assert res["stress_score"] == 48.5
    assert res["stress_regime"] == "Normal Market Conditions"
    assert len(res["sections"]) == 3
    assert "expanded" in res["executive_summary"]
    assert "1,250.4" in res["executive_summary"]

    # Verify zero forbidden trading jargon
    full_text = f"{res['headline']} {res['executive_summary']} " + " ".join(
        f"{s['title']} {s['content']}" for s in res["sections"]
    )

    for pat in FORBIDDEN_TERMS:
        match = re.search(pat, full_text, re.IGNORECASE)
        assert not match, f"Forbidden term '{match.group(0)}' found in generated briefing"

    # Verify zero individual stock tickers
    for ticker_pat in STOCK_TICKERS:
        match = re.search(ticker_pat, full_text)
        assert not match, f"Prohibited stock ticker '{match.group(0)}' found in generated briefing"


def test_render_share_card_svg(tmp_path):
    card_path = tmp_path / "test_card.svg"
    sz = render_share_card_svg(
        as_of="2026-11-20",
        headline="Global Financial Stress: Normal Market Conditions",
        stress_score=48.5,
        stress_regime="Normal Market Conditions",
        net_liquidity_val=6120.5,
        net_liquidity_chg=23.04,
        fii_net=1250.40,
        dii_net=1840.15,
        top_observation="Realized volatility elevated across multiple horizons for CBOE VIX",
        output_path=card_path,
    )

    assert sz > 0
    assert card_path.exists()
    content = card_path.read_text(encoding="utf-8")
    assert '<svg width="1200" height="630"' in content
    assert "MACROMONITOR" in content
    assert "2026-11-20" in content
    assert "48.5" in content
