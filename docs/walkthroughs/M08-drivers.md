# Milestone Walkthrough: M8 Macro Drivers Layer

**Milestone:** M8 (Driver Sensitivity Layer)  
**Governing Plan:** `docs/MASTER_PLAN.md` (Section 4.3 & Section 5.1, Charts C06, C07, C08, C09)  
**Status:** Completed  
**Checkpoint:** Mandatory Human Checkpoint after M8  

---

## 1. Objectives & Deliverables

1. **Econometric Specification & Estimation Engine**:
   - Built `pipeline/src/radar/metrics/drivers.py`:
     - Rolling multi-factor regressions over **250-day** and **120-day** estimation windows using Newey-West HAC covariance.
     - Collinearity protection: VIF computation with automatic Ridge fallback when $\text{VIF} > 5.0$.
     - Stability metric: fraction of rolling windows sharing the same sign as the current estimate $\in [0.0, 1.0]$.
     - Pre-registered economic priors matching (oil importers vs exporters, dollar tightness proxy, cyclical copper proxy).
     - Bivariate scatter and OLS fit with confidence interval bands.
2. **Data Contracts & Atomic Export**:
   - Added `DriversEnvelope`, `DriversData`, `DriverTargetMatrixRow`, `DriverSensitivityItem`, `DriverScatterData` to Python and TypeScript schemas (`schemas.py` and `schemas.ts`).
   - Integrated atomic export into `writers.py` generating `web/public/data/v1/drivers.json` (1.28 MB) with full coverage for all 10 launch economies.
3. **Editorial Web Interface (`/drivers`)**:
   - Created `web/src/app/drivers/page.tsx` and `DriverClientView.tsx`.
   - Controls: Target economy selector and 250D / 120D estimation window toggles.
   - StatBlocks: Active Target, Model $R^2$, Dominant Driver, Sensitivity Stability Ratio, Economic Priors Alignment.
   - Visualizations:
     - **C06**: `DriverSensitivityHeatmap.tsx`: Cross-asset sensitivity heatmap with significance hatching and stability alerts.
     - **C08**: `DriverForestPlot.tsx`: Horizontal forest plot displaying point estimates and 95% Newey-West HAC confidence intervals on one chart.
     - **C07**: `RollingBetaChart.tsx`: Trailing time-series chart of rolling betas with confidence ribbons.
     - **C09**: `DriverBivariateScatter.tsx`: Daily scatter with fitted OLS regression line and confidence bands.
   - Standardized Method and Audit Drawers for every chart with KaTeX mathematical formulas and reproduction links.
   - Prerendered static pages (**83/83 static routes** in Next.js build).

---

## 2. Test Evidence

- `pipeline/tests/unit/test_drivers.py`: 6/6 tests passing:
  - `test_planted_coefficient_recovery_hac`: Planted coefficients ($\beta_1 = 0.85, \beta_2 = -0.50$) recovered within 0.05, 95% CI covers truth.
  - `test_noise_fixture_rejection`: Unassociated noise factors produce non-significant betas whose 95% CIs span zero.
  - `test_stability_metric_invariants`: Boundary and sign invariants verified.
  - `test_economic_prior_matching`: Verified structural priors for oil importers vs exporters and DXY.
  - `test_rolling_beta_anti_lookahead`: Strict anti-lookahead guarantee verified across truncated windows.
  - `test_bivariate_scatter_and_fit`: Verified scatter points, line, and confidence band bounds.
- Full pipeline integration: 19/19 targeted tests passing in 12.9s.
- HTTP Verification: `http://localhost:3000/drivers/` returns HTTP 200 with 17.4 kB content.
