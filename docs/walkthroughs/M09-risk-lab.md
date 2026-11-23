# Milestone Walkthrough: M09 — Institutional Risk Lab

## 1. Goal & Requirements
Milestone M9 delivers the Institutional Risk Lab (`/risk`) as specified in Section 4.4 and Section 5.1 (Charts C10 to C17) of `docs/MASTER_PLAN.md`:
1. Realized Volatility Cones across {10D, 20D, 60D, 120D, 250D} with historical percentile bands.
2. Historical vs. Gaussian Tail Risk (VaR and Expected Shortfall at 95% and 99%) and Tail Ratio.
3. ARCH Volatility Clustering via Autocorrelation Functions of returns $r_t$ and squared returns $r_t^2$ with $N_{\text{eff}}$ bands.
4. Continuous Underwater Drawdowns series with an audit table of the top 5 historical crisis episodes (peak, trough, recovery, depth, duration).
5. Global Market Synchronization and PCA Factor Structure across 10 global equity markets in USD, enforcing the strictly positive US loading sign-alignment invariant.
6. Downside Stress Correlation and Stress Correlation Premium ($\Delta \rho = \rho_{\text{downside}} - \rho_{\text{all}}$) on the worst 10% benchmark sessions.

---

## 2. Implementation Summary

### Pipeline Statistics Engine
- **Module**: `pipeline/src/radar/metrics/risk.py`
  - `compute_volatility_cone(prices, windows, annualization_factor)`
  - `compute_ewma_volatility(returns, decay=0.94)`
  - `compute_distribution_shape_and_qq(returns, num_qq_points=50)`
  - `compute_tail_risk(returns, confidence_levels=[0.95, 0.99])`
  - `compute_autocorrelations(returns, max_lags=20)`
  - `compute_underwater_drawdowns(prices, top_k=5)`
  - `compute_pca_factor_structure(country_returns_df, us_col='spx', window=250)`
  - `compute_downside_correlation(target_returns, benchmark_returns, stress_quantile=0.10)`
- **Unit Test Suite**: `pipeline/tests/unit/test_risk.py` (8/8 tests passing, 90% coverage).
- **Data Contract**: `pipeline/src/radar/export/schemas.py` and `web/src/lib/data/schemas.ts` defining `RiskData` and `RiskEnvelope`.
- **Exported Dataset**: `web/public/data/v1/risk.json` (1.06 MB, verified via Pydantic & Zod).

### Frontend Web UI & Visualization System
- **Components**:
  - `web/src/components/charts/VolatilityConeChart.tsx` (C10)
  - `web/src/components/charts/TailRiskBarChart.tsx` (C13)
  - `web/src/components/charts/AutocorrelationBarChart.tsx` (C14)
  - `web/src/components/charts/UnderwaterDrawdownChart.tsx` (C15)
  - `web/src/components/charts/PcaSynchronizationChart.tsx` (C16)
  - `web/src/components/charts/DownsideCorrelationChart.tsx` (C17)
- **Client & Page Views**:
  - `web/src/app/risk/page.tsx`
  - `web/src/app/risk/RiskClientView.tsx` with asset selector (S&P 500, Nifty 50, FTSE 100, DAX 40, Nikkei 225, Brent, Gold, Copper, DXY, US 10Y, VIX).
  - Navigation integrated into top bar (`web/src/app/layout.tsx`).

---

## 3. Verification & Acceptance
- Pytest unit tests: `pytest pipeline/tests/unit/test_risk.py` -> 8 passed in 5.13s.
- Pytest contract tests: `pytest pipeline/tests/unit/test_contracts.py` -> 8 passed.
- Next.js static build verified with all static pages prerendered.
