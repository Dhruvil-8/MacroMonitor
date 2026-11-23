# Milestone Walkthrough: M11 — Radar & Observation Engine

## 1. Goal & Requirements
Milestone M11 completes the Cross-Asset Radar and Change Feed (`/radar`) as specified in Section 4.6, Section 5.1 (Chart C24), and Section 9.1-9.5 of `docs/MASTER_PLAN.md`:
1. Two-sided Cumulative Sum (CUSUM) Control Chart (C24) on standardized innovations with slack $k=0.5$ and decision threshold $h=4.0$.
2. Implementation of all 10 Section 4.6 observation rules: country divergence, currency-driven move, sector leadership, driver sensitivity shift, commodity shock, dollar regime, breadth deterioration, PC1 synchronization spike, downside correlation jump, and volatility cone breach.
3. Batch candidate ranking and prioritization (`high` > `medium` > `low`) enforcing the daily cap of 8 alerts and 10-day cooldown suppression.
4. Total linguistic safety verifying zero prohibited trading terms (`test_templates_safety.py`).
5. Interactive frontend view on `/radar` featuring:
   - Category filtering (All, Vol & Risk, Cross-Market, Macro) and severity buttons (All, High, Medium, Low).
   - Change feed observation cards with method and audit drawer links.
   - Pre-registered divergence panels with an inline toggle between Z-score spread history and the two-sided CUSUM control chart (C24).

---

## 2. Implementation Summary

### Pipeline Statistics Engine
- **Module**: `pipeline/src/radar/metrics/divergence.py`
  - `compute_cusum_series(series, slack_k=0.5, threshold_h=4.0)`
    Computes $S_t^+ = \max(0, S_{t-1}^+ + z_t - k)$ and $S_t^- = \max(0, S_{t-1}^- - z_t - k)$, detecting discrete shift events and resetting accumulators.
- **Insight Engine**: `pipeline/src/radar/insights/engine.py`
  - `process_candidates_batch(candidates)`
    Sorts incoming candidates by severity (`high` > `medium` > `low`) so critical tail risk and volatility cone breaches are preserved under volume caps.
  - Enforces 10-day cooldown per `(asset, rule_id)` and daily cap of 8 alerts.
- **Templates**: `pipeline/config/templates.yaml`
  - Added deterministic, audited templates for all Section 4.6 observation rules.
- **Export Engine**: `pipeline/src/radar/export/writers.py`
  - Gathers candidate observations across all macro layers (Country, Sectors, Drivers, Commodities, Currencies, and Risk Lab).
  - Precomputes CUSUM control series for all 10 pre-registered divergence pairs.
  - Exports atomic envelopes `web/public/data/v1/observations.json` and `web/public/data/v1/divergences.json`.

### Frontend Web UI
- **Components**:
  - `web/src/components/charts/CusumChangeChart.tsx` (C24): Interactive ECharts rendering Upward CUSUM ($S_t^+$ in emerald), Downward CUSUM ($S_t^-$ in rose), threshold reference line ($h=4.0$), and summary stat badges.
  - `web/src/components/charts/DivergencePanel.tsx`: Enhanced with a segmented toggle between "Spread" and "CUSUM" charts.
- **Page View**:
  - `web/src/app/radar/page.tsx`: Upgraded with category filter pills (All, Vol & Risk, Cross-Market, Macro) and severity controls.

---

## 3. Verification & Acceptance
- **Unit Tests**:
  - `pipeline/tests/unit/test_divergence.py`: 6/6 tests passing (step detection, downward detection, zero false alarms on Gaussian noise, short series handling).
  - `pipeline/tests/unit/test_insight_engine.py`: 4/4 tests passing (severity mapping, cooldown suppression, daily cap, batch priority sorting).
  - `pipeline/tests/unit/test_templates_safety.py`: Verified zero forbidden terms.
- **Pipeline Export**: All 140 datasets exported without error.
- **Static Site Build**: `npm run build` completed with all 84 static routes prerendered cleanly.
- **HTTP Verification**: `http://localhost:3000/radar/index.html` verified with HTTP 200.
