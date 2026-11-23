# Milestone Walkthrough: M12 — Liquidity Monitor

## 1. Goal & Requirements
Milestone M12 implements the Global Liquidity & Funding Monitor (`/liquidity`) as specified in Section 8.8 and Section 13.6 of `docs/MASTER_PLAN.md`:
1. Computation of US Net Liquidity: $L_t = \frac{\text{WALCL}}{1000} - \frac{\text{WTREGEN}}{1000} - \text{RRPONTSYD}$ with unit conversion to USD Billions.
2. Forward-fill frequency alignment bridging weekly Wednesday balance sheet releases with daily overnight reverse repo data.
3. 4-Week (20D) and 13-Week (65D) changes with expanding historical percentiles.
4. S&P 500 equity overlay with rolling 60-day Pearson correlation.
5. Stablecoin total circulating supply and 30-day percentage momentum.
6. Hand-checked numerical verification against known historical dates.
7. Web UI with StatBlocks, ChartCard, Method & Audit Drawers, and mandatory causality caveats.

---

## 2. Implementation Summary

### Pipeline Statistics Engine
- **Module**: `pipeline/src/radar/metrics/liquidity.py`
  - `compute_net_liquidity(fed_assets, tga, rrp, max_ffill_days=7)`
  - `compute_liquidity_changes(net_liquidity, windows=[20, 65])`
  - `compute_liquidity_equity_overlay(net_liquidity, spx, corr_window=60)`
  - `compute_stablecoin_metrics(stablecoin_supply, window=30)`
- **Unit Test Suite**: `pipeline/tests/unit/test_liquidity.py` (4/4 tests passing, 92% coverage).
  - Verified against 3 hand-checked historical dates (2024-01-03, 2025-01-08, 2026-11-18).
  - Verified forward-fill frequency alignment and 4W/13W change percentiles.
  - Verified rolling correlation against S&P 500.
  - Verified stablecoin metrics.
- **Data Export**: `pipeline/src/radar/export/writers.py` exports `web/public/data/v1/liquidity.json`.

### Frontend Web UI
- **Components**:
  - `web/src/components/charts/NetLiquidityChart.tsx`: Stacked area/lines for Fed Assets, TGA, RRP with Net Liquidity line, dual-axis S&P 500 overlay, and rolling 60-day correlation strip below.
  - `web/src/app/liquidity/page.tsx`: Editorial dashboard featuring 5 KPI StatBlocks, ChartCard with Lineage Audit, and Narrative & Empirical Caution panels.

---

## 3. Verification & Acceptance
- Pytest unit tests: `pytest pipeline/tests/unit/test_liquidity.py` -> 4 passed in 9.62s.
- Next.js static build: prerendered `/liquidity` cleanly.
- HTTP status: Verified HTTP 200 at `http://localhost:3000/liquidity/index.html`.
