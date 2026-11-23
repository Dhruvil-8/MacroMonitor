# ADR 0021: Daily Cross-Asset Briefing & Open Graph Share Cards

## Context
Section 8 and Section 10 of `docs/MASTER_PLAN.md` require an automated nightly Daily Briefing (`/briefing`) and server-rendered 1200x630 Open Graph share cards (`/v1/og/<as_of>.svg`). The briefing must synthesize live quantitative metrics across market stress, liquidity, institutional flows, and top statistical observations without human editorial intervention, while enforcing strict linguistic safety and zero individual equity mentions.

## Decision
1. **Deterministic Rules-Generated Briefing Architecture**:
   The module `pipeline/src/radar/insights/briefing.py` generates a structured daily digest by ingesting:
   - Financial Stress composite score and classified regime.
   - Component friction percentiles across equity, rates, FX, and commodities.
   - Top observations from `InsightEngine` sorted by severity (`high` > `medium` > `low`).
   - US Net Liquidity 4-week rate of change.
   - FII and DII net cash market purchases.
   - Total circulating USD stablecoins.

2. **Automated Linguistic Safety & Ticker Screening**:
   In strict compliance with Section 8 of the Master Plan, all generated strings are subject to automated verification in `test_briefing.py`:
   - **Zero Advisory Jargon**: Scanned against prohibited terms (`buy`, `sell`, `bullish`, `bearish`, `target`, `will rise`, `will fall`, `recommend`, `signal to`, `outperform`, `underperform`).
   - **Zero Individual Tickers**: Scanned against single-stock tickers and company names (`AAPL`, `MSFT`, `TSLA`, `NVDA`, `RELIANCE`, `TCS`, `INFY`, `HDFCBANK`), preserving strict macro index-level scope.

3. **Server-Rendered 1200x630 Open Graph Share Card**:
   The module `pipeline/src/radar/export/share_card.py` renders an institutional 1200x630 pixel SVG card containing:
   - Header masthead with date stamp.
   - Main headline with regime classification.
   - 3 Key KPI cards: Financial Stress, US Net Liquidity level & 4W delta, India FII/DII net cash.
   - Primary Statistical Observation highlight.
   - Statutory footer and source citations.
   Written atomically to `web/public/data/v1/og/<as_of>.svg` and `web/public/data/v1/og/latest.svg`.

4. **Frontend Editorial Interface**:
   The `/briefing` route displays:
   - Executive Summary card.
   - Interactive Stress Gauge and trailing 300-day crisis episode timeline.
   - Interactive 1200x630 Share Card preview with a one-click SVG download button.
   - Detailed thematic sections (Frictions, Observations, Funding & Flows).

## Consequences
- Every night, an updated briefing and share card are generated automatically, requiring zero manual authoring.
- Analysts and institutional users can download and embed the high-resolution vector share card into research presentations and social channels.
