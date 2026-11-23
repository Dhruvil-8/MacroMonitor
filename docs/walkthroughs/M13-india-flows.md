# Milestone Walkthrough: M13 — India Flows (Trimmed)

## 1. Goal & Requirements
Milestone M13 implements the India Flows & Positioning module (`/india`) as specified in Section 8.9 and Section 1.3 of `docs/MASTER_PLAN.md`:
1. 4-quadrant F&O derivative Open Interest (OI) buildup classification: Long Buildup, Short Buildup, Short Covering, Long Unwinding.
2. FII and DII cash flow tracking: daily net, 5-day and 20-day cumulative sums with expanding percentiles.
3. Index option Put-Call Ratio (PCR) calculation.
4. Volume-weighted delivery percentage across Nifty 50 constituents.
5. India VIX historical expanding percentile ribbon.
6. Hand-checked numerical verification against known market dates.
7. Web UI with StatBlocks, `FiiDiiFlowChart.tsx`, `OiBuildupMap.tsx`, and Method & Audit Drawers.

---

## 2. Implementation Summary

### Pipeline Statistics Engine
- **Module**: `pipeline/src/radar/metrics/india_flows.py`
  - `classify_oi_buildup(price_change_pct, oi_change_pct)`
  - `compute_fii_dii_rolling_flows(fii_daily, dii_daily, windows=[5, 20])`
  - `compute_put_call_ratio(put_oi, call_oi)`
  - `compute_delivery_metrics(delivered_qty, traded_qty)`
- **Unit Test Suite**: `pipeline/tests/unit/test_india_flows.py` (5/5 tests passing, 95% coverage).
  - Verified 4-quadrant OI classification across all price/OI combinations.
  - Verified against 3 hand-checked historical dates (2024-06-04 Election Volatility, 2024-11-04 Consolidation, 2026-11-20 Baseline).
  - Verified multi-horizon rolling sums and expanding percentile bounds.
  - Verified PCR division-by-zero protection.
  - Verified volume-weighted delivery percentage.
- **Data Export**: `pipeline/src/radar/export/writers.py` exports `web/public/data/v1/india_flows.json`.

### Frontend Web UI
- **Components**:
  - `web/src/components/charts/FiiDiiFlowChart.tsx`: Stacked/grouped bar charts for FII and DII daily flows with 20D rolling trend lines.
  - `web/src/components/charts/OiBuildupMap.tsx`: 4-quadrant derivative positioning map (Price % vs OI Change %) for NIFTY, BANKNIFTY, NIFTYIT, etc.
  - `web/src/app/india/page.tsx`: Comprehensive institutional overview with StatBlocks for FII Net, DII Net, India VIX, PCR, and Delivery %, and complete Lineage Audit Drawers.

---

## 3. Verification & Acceptance
- Pytest unit tests: `pytest pipeline/tests/unit/test_india_flows.py` -> 5 passed in 5.67s.
- Next.js static build: prerendered `/india` cleanly.
- HTTP status: Verified HTTP 200 at `http://localhost:3000/india/index.html`.
