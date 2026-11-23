"""Automated Daily Briefing Generator.

Section 8.7 & Section 13:
- Rules-based summary generated nightly without manual editing
- Includes Stress Composite score, dominant regime, component breakdown
- Top observations from Insight Engine
- Cross-asset divergences & rotations
- Net Liquidity and Institutional flows pulse
- Zero investment advice, zero forecasting, purely descriptive statistics
"""

from typing import Any


def generate_daily_briefing(
    as_of: str,
    stress_score: float,
    stress_regime: str,
    component_scores: dict[str, float],
    top_observations: list[dict[str, Any]],
    net_liquidity_chg_4w: float,
    fii_net_1d: float,
    dii_net_1d: float,
    stablecoins_total_bn: float,
) -> dict[str, Any]:
    """Generate structured daily briefing adhering strictly to safety guidelines.

    Args:
        as_of: Snapshot date string YYYY-MM-DD.
        stress_score: Composite stress index score [0-100].
        stress_regime: Regime string ('Low Stress', 'Normal Market Conditions', etc.).
        component_scores: Breakdown of available component percentiles.
        top_observations: List of top observation dicts from engine.
        net_liquidity_chg_4w: 4-week change in US Net Liquidity (USD B).
        fii_net_1d: Latest FII net cash purchases (INR Cr).
        dii_net_1d: Latest DII net cash purchases (INR Cr).
        stablecoins_total_bn: Total circulating USD stablecoins (USD B).

    Returns:
        Structured dict containing headline, executive_summary, and detailed sections.
    """
    # 1. Executive Summary
    liq_direction = "expanded" if net_liquidity_chg_4w >= 0 else "contracted"
    fii_direction = "recorded net inflows" if fii_net_1d >= 0 else "recorded net outflows"
    dii_direction = "recorded net inflows" if dii_net_1d >= 0 else "recorded net outflows"

    headline = f"MacroMonitor Briefing: {stress_regime} ({stress_score:.1f}/100)"
    summary = (
        f"As of {as_of}, the Global Financial Stress Composite registered {stress_score:.1f}/100, "
        f"reflecting {stress_regime.lower()}. US Net Liquidity has {liq_direction} by "
        f"${abs(net_liquidity_chg_4w):.1f}B over the trailing 4-week window, while institutional cash "
        f"participants in India {fii_direction} (FII: ₹{fii_net_1d:,.1f} Cr) and {dii_direction} "
        f"(DII: ₹{dii_net_1d:,.1f} Cr)."
    )

    # 2. Section 1: Stress & Volatility Landscape
    stress_detail = []
    for comp, val in component_scores.items():
        stress_detail.append(f"{comp.capitalize()}: {val:.1f}th percentile")
    stress_section = {
        "title": "Financial Stress & Volatility Breakdown",
        "content": (
            f"The composite score is derived as the arithmetic mean of available cross-asset frictions: "
            f"{', '.join(stress_detail)}. No values are imputed for missing series."
        ),
    }

    # 3. Section 2: Primary Statistical Observations
    obs_lines = []
    for obs in top_observations[:4]:
        obs_lines.append(f"• {obs.get('headline', '')}")
    if not obs_lines:
        obs_lines.append("• No statistically unusual return or volatility threshold breaches detected across active series.")

    obs_section = {
        "title": "Primary Statistical Observations",
        "content": "\n".join(obs_lines),
    }

    # 4. Section 3: Liquidity, Funding & Flows
    flows_section = {
        "title": "Funding & Institutional Positioning",
        "content": (
            f"Federal Reserve Net Liquidity stands at its current historical baseline with a 4-week delta of "
            f"{'+' if net_liquidity_chg_4w >= 0 else ''}{net_liquidity_chg_4w:.1f}B. Total circulating USD "
            f"stablecoins aggregate is ${stablecoins_total_bn:.1f}B. In Indian equities, cumulative institutional "
            f"absorption continues to reflect countercyclical domestic liquidity provisioning."
        ),
    }

    return {
        "as_of": as_of,
        "headline": headline,
        "executive_summary": summary,
        "stress_score": round(stress_score, 1),
        "stress_regime": stress_regime,
        "sections": [stress_section, obs_section, flows_section],
    }
