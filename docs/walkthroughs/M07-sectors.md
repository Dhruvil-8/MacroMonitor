# Milestone Walkthrough: M7 Sectors Layer

**Milestone:** M7 (Sectors Layer)  
**Governing Plan:** `docs/MASTER_PLAN.md` (Section 4.2 & Section 5.1, C05)  
**Status:** Completed  

---

## 1. Objectives & Deliverables

1. **Universe & Weights Foundation**:
   - Implemented `pipeline/config/sector_weights.yaml` with published monthly index factsheet weights for US (11 GICS sectors) and India (7 Nifty sectors).
   - Documented explicit source URLs and `as_of` dates.
2. **Quantitative Analytics Engine**:
   - `pipeline/src/radar/metrics/sector.py`:
     - Approximate contribution: $\text{contrib}_s = w_s \times r_s$.
     - Explicit residual: $\text{residual} = r_{\text{index}} - \sum_s \text{contrib}_s$, guaranteeing exact arithmetic reconciliation.
     - Relative rotation cycle: 60-day relative strength ratio, 10-day rate of change momentum, and 10-day history trajectory trail across all 4 quadrants (Leading, Weakening, Lagging, Improving).
     - Multi-horizon performance heatmap: returns across {1D, 5D, 1M, 3M, 1Y} with anti-lookahead expanding historical percentile ranks $P(r) \in [0.0, 100.0]$.
     - Sector breadth: share of tracked sectors outperforming the benchmark index $\in [0.0, 1.0]$.
3. **Data Contracts & Export**:
   - Added `SectorsEnvelope`, `SectorsData`, `SectorRegionData`, `SectorItem`, `SectorRotationPoint` to Python schemas (`schemas.py`) and TypeScript Zod schemas (`schemas.ts`).
   - Integrated atomic export into `writers.py` generating `web/public/data/v1/sectors.json` (41.4 kB).
4. **Editorial Web Interface**:
   - Created `/sectors` (`web/src/app/sectors/page.tsx` and `SectorClientView.tsx`).
   - Region switcher: US (S&P 500) | India (Nifty 50).
   - Horizon selector: 1D, 5D, 1M, 3M, 1Y.
   - StatBlocks: Benchmark Move, Sector Breadth, Top Contributor, Top Detractor, Residual / Drift.
   - Interactive Visualizations:
     - `SectorContributionWaterfall.tsx`: Horizontal waterfall bar chart with explicit residual and total benchmark bar.
     - `SectorRotationMap.tsx`: 4-quadrant scatter chart with trajectory tails (10 days) and toggle.
     - `SectorHeatmap.tsx`: Multi-horizon performance table color-shaded by historical percentile rank with sortable columns.
   - Method and Audit Drawers for every chart with KaTeX mathematical formulas, parameters, and code reproduction links.
   - Prerendered static pages (82/82 static routes).

---

## 2. Test Evidence

- `pipeline/tests/unit/test_sector.py`: 5/5 unit tests passing:
  - `test_approximate_contribution_reconciliation_identity`: Reconciles sum of contributions + residual to index return within $10^{-5}$.
  - `test_sector_breadth_invariants`: Verifies breadth is bounded in $[0.0, 1.0]$.
  - `test_sector_rotation_quadrant_classification`: Verifies all 4 quadrant classifications (Leading, Weakening, Lagging, Improving) and trajectory trails.
  - `test_sector_weights_configuration_validity`: Validates positive weights and sums $\le 1.05$.
  - `test_analyze_region_sectors_end_to_end`: Synthetic end-to-end integration test.
- `pipeline/tests/unit/test_contracts.py`: Added `test_sectors_schema_serialization` (Pydantic v2 validation).
- `pipeline/tests/unit/test_html_artifacts.py`: Verified `sectors/index.html` static export.
