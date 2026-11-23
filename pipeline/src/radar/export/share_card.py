# pipeline/src/radar/export/share_card.py
"""Render publication-grade 1200x630 SVG share card and Open Graph image.

Implements Section 8 and Section 10 of MASTER_PLAN.md:
- 1200x630 pixel canvas (Open Graph standard)
- Institutional dark editorial styling (#07090d background, #1d2633 line, #2f6fb4 accent)
- Summary metrics: Stress Composite, Net Liquidity, Institutional Flows, Top Observation
- Zero forecasts, zero ticker symbols, fully traceable
"""

from pathlib import Path
from typing import Optional
import html


def render_share_card_svg(
    as_of: str,
    headline: str,
    stress_score: float,
    stress_regime: str,
    net_liquidity_val: float,
    net_liquidity_chg: float,
    fii_net: float,
    dii_net: float,
    top_observation: str,
    output_path: Path,
) -> int:
    """Render 1200x630 SVG share card and write to output_path.

    Returns:
        Number of bytes written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stress_color = "#3fb6a8" if stress_score < 40 else ("#d88c3a" if stress_score < 70 else "#d9534f")
    liq_color = "#3fb6a8" if net_liquidity_chg >= 0 else "#d9534f"
    fii_color = "#3fb6a8" if fii_net >= 0 else "#d9534f"
    dii_color = "#3fb6a8" if dii_net >= 0 else "#d9534f"

    esc_headline = html.escape(headline[:60], quote=True)
    esc_top_obs = html.escape(top_observation[:110], quote=True)

    svg = f"""<svg width="1200" height="630" viewBox="0 0 1200 630" xmlns="http://www.w3.org/2000/svg">
  <!-- Dark Editorial Background -->
  <rect width="1200" height="630" fill="#07090d" />
  
  <!-- Subtle Gradient Glow in Corner -->
  <radialGradient id="glow" cx="0%" cy="0%" r="50%">
    <stop offset="0%" stop-color="#2f6fb4" stop-opacity="0.15" />
    <stop offset="100%" stop-color="#07090d" stop-opacity="0" />
  </radialGradient>
  <rect width="1200" height="630" fill="url(#glow)" />

  <!-- Outer Border -->
  <rect x="20" y="20" width="1160" height="590" rx="12" fill="none" stroke="#1d2633" stroke-width="1.5" />

  <!-- Brand Masthead Header -->
  <g transform="translate(60, 65)">
    <!-- Brand Icon -->
    <rect x="0" y="0" width="28" height="28" rx="6" fill="#2f6fb4" />
    <text x="14" y="19" font-family="monospace" font-size="14" font-weight="bold" fill="#ffffff" text-anchor="middle">M</text>
    
    <text x="38" y="20" font-family="sans-serif" font-size="16" font-weight="bold" fill="#e8edf5" letter-spacing="-0.02em">MACROMONITOR</text>
    <text x="175" y="19" font-family="monospace" font-size="12" fill="#6f7c90">/ MACRO STATISTICS</text>

    <!-- Date Badge -->
    <rect x="940" y="0" width="140" height="28" rx="4" fill="#0c1017" stroke="#1d2633" />
    <text x="1010" y="18" font-family="monospace" font-size="12" fill="#a9b4c4" text-anchor="middle">AS OF {as_of}</text>
  </g>

  <!-- Divider Line -->
  <line x1="60" y1="110" x2="1140" y2="110" stroke="#1d2633" stroke-width="1" />

  <!-- Headline & Subtitle -->
  <g transform="translate(60, 160)">
    <text x="0" y="0" font-family="serif" font-size="34" font-weight="600" fill="#e8edf5" letter-spacing="-0.01em">
      {esc_headline}
    </text>
    <text x="0" y="32" font-family="sans-serif" font-size="15" fill="#8a99ad">
      Cross-asset financial conditions, liquidity flows, and statistical regime diagnostics.
    </text>
  </g>

  <!-- 3 KPI Feature Metric Cards -->
  <!-- Card 1: Financial Stress -->
  <g transform="translate(60, 230)">
    <rect x="0" y="0" width="340" height="150" rx="8" fill="#0c1017" stroke="#1d2633" />
    <text x="20" y="30" font-family="monospace" font-size="11" font-weight="600" fill="#6f7c90" text-transform="uppercase">FINANCIAL STRESS COMPOSITE</text>
    <text x="20" y="78" font-family="monospace" font-size="36" font-weight="bold" fill="{stress_color}">{stress_score:.1f}<tspan font-size="18" fill="#6f7c90">/100</tspan></text>
    <rect x="20" y="102" width="8" height="8" rx="4" fill="{stress_color}" />
    <text x="35" y="110" font-family="sans-serif" font-size="13" font-weight="500" fill="#e8edf5">{stress_regime}</text>
  </g>

  <!-- Card 2: US Net Liquidity -->
  <g transform="translate(430, 230)">
    <rect x="0" y="0" width="340" height="150" rx="8" fill="#0c1017" stroke="#1d2633" />
    <text x="20" y="30" font-family="monospace" font-size="11" font-weight="600" fill="#6f7c90" text-transform="uppercase">US NET LIQUIDITY</text>
    <text x="20" y="78" font-family="monospace" font-size="36" font-weight="bold" fill="#e8edf5">${net_liquidity_val:,.0f}B</text>
    <text x="20" y="110" font-family="monospace" font-size="13" fill="{liq_color}">4W Delta: {'+' if net_liquidity_chg >= 0 else ''}${net_liquidity_chg:,.1f}B</text>
  </g>

  <!-- Card 3: India Institutional Cash -->
  <g transform="translate(800, 230)">
    <rect x="0" y="0" width="340" height="150" rx="8" fill="#0c1017" stroke="#1d2633" />
    <text x="20" y="30" font-family="monospace" font-size="11" font-weight="600" fill="#6f7c90" text-transform="uppercase">INDIA CASH FLOWS (1D)</text>
    <text x="20" y="65" font-family="monospace" font-size="15" fill="#e8edf5">FII: <tspan font-weight="bold" fill="{fii_color}">₹{fii_net:+,.1f} Cr</tspan></text>
    <text x="20" y="95" font-family="monospace" font-size="15" fill="#e8edf5">DII: <tspan font-weight="bold" fill="{dii_color}">₹{dii_net:+,.1f} Cr</tspan></text>
    <text x="20" y="125" font-family="sans-serif" font-size="11" fill="#6f7c90">Net Domestic/Foreign Cash Inflow</text>
  </g>

  <!-- Key Observation Highlight Box -->
  <g transform="translate(60, 410)">
    <rect x="0" y="0" width="1080" height="110" rx="8" fill="#0c1017" stroke="#1d2633" />
    <text x="24" y="32" font-family="monospace" font-size="11" font-weight="bold" fill="#2f6fb4" letter-spacing="0.05em">PRIMARY STATISTICAL OBSERVATION</text>
    <text x="24" y="68" font-family="sans-serif" font-size="16" font-weight="500" fill="#e8edf5">
      {esc_top_obs}
    </text>
    <text x="24" y="92" font-family="sans-serif" font-size="12" fill="#8a99ad">
      Identified via multi-horizon expanding z-score and persistent divergence filtering.
    </text>
  </g>

  <!-- Footer Disclaimers & Sources -->
  <g transform="translate(60, 560)">
    <line x1="0" y1="0" x2="1080" y2="0" stroke="#1d2633" stroke-width="1" />
    <text x="0" y="24" font-family="sans-serif" font-size="11" fill="#6f7c90">
      Deterministic Open-Data Research • Strictly Descriptive Statistics • Zero Advisory Claims • Sources: Federal Reserve, NSE India, NY Fed, Yahoo Finance
    </text>
    <text x="1080" y="24" font-family="monospace" font-size="11" fill="#2f6fb4" text-anchor="end">
      macromonitor.org
    </text>
  </g>
</svg>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)

    return len(svg.encode("utf-8"))
