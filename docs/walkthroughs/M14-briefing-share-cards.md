# Milestone Walkthrough: M14 — Briefing & Share Cards

## 1. Goal & Requirements
Milestone M14 completes the automated Daily Briefing page, archive, and server-rendered 1200x630 share card as specified in Section 8 and Section 10 of `docs/MASTER_PLAN.md`:
1. Rules-based daily briefing text generator ingesting live quantitative pipeline metrics.
2. Mandatory linguistic neutrality verifying zero forbidden trading terms and zero individual stock tickers.
3. Server-side 1200x630 SVG share card and Open Graph image generation.
4. Export of dated and latest JSON envelopes (`/v1/briefing/<as_of>.json` and `/v1/briefing/latest.json`).
5. Web UI on `/briefing` featuring executive summary, stress timeline, share card preview with download, and breakdown sections.

---

## 2. Implementation Summary

### Pipeline Statistics & Rendering Engine
- **Module**: `pipeline/src/radar/insights/briefing.py`
  - `generate_daily_briefing(...)`: Generates headline, executive summary, and 3 thematic sections (Frictions, Observations, Funding & Flows).
- **Module**: `pipeline/src/radar/export/share_card.py`
  - `render_share_card_svg(...)`: Renders 1200x630 pixel SVG card with masthead, headline, 3 KPI cards (Stress, Liquidity, India Flows), top observation callout, and statutory disclaimer footer.
- **Export Pipeline**: `pipeline/src/radar/export/writers.py`
  - Passes live `accepted_observations[:4]` and computed flow numbers into briefing.
  - Outputs dated `/v1/briefing/<as_of>.json`, `/v1/briefing/latest.json`, `/v1/og/<as_of>.svg`, and `/v1/og/latest.svg`.
- **Unit Test Suite**: `pipeline/tests/unit/test_briefing.py` (2/2 tests passing, 95% coverage on briefing, 100% on share_card).

### Frontend Web UI
- **Page View**: `web/src/app/briefing/page.tsx`
  - Upgraded with 1200x630 share card preview container and direct SVG download button.
  - Executive summary card and stress gauge/timeline integration.

---

## 3. Verification & Acceptance
- Pytest unit tests: `pytest pipeline/tests/unit/test_briefing.py` -> 2 passed in 0.89s.
- Next.js static build: prerendered `/briefing` cleanly (84/84 routes).
- HTTP verification: Confirmed HTTP 200 on `http://localhost:3000/briefing/index.html` and `http://localhost:3000/data/v1/og/latest.svg`.
