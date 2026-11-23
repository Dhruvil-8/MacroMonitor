# Milestone Walkthrough: M10 — Context Pages & Distribution Modeling

## 1. Goal & Requirements
Milestone M10 implements the Historical Context layer (`/context/[id]`) across the entire universe as specified in Section 4.5 and Section 5.1 of `docs/MASTER_PLAN.md`:
1. Histogram with Gaussian Kernel Density Estimate (Silverman's rule of thumb) and Normal $\mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$ overlay (C11).
2. Quantile-Quantile (QQ) plot (C12) comparing empirical standardized returns against theoretical Gaussian quantiles with 45-degree reference line.
3. Empirical Cumulative Distribution Function (ECDF) with current "You are here" percentile marker (C20).
4. Calendar monthly seasonality heatmap (C25) with mandatory chance caveat banner.
5. Declustered historical analogs (C26) with realized forward 5D and 20D return outcomes.

---

## 2. Implementation Summary

### Pipeline Statistics Engine
- **Module**: `pipeline/src/radar/metrics/context.py`
  - `compute_distribution_histogram(s, num_bins=25)`
  - `compute_kde_and_normal_overlay(returns, grid_points=60)` using Silverman's $h = 0.9 \min(\hat{\sigma}, \frac{\text{IQR}}{1.34}) n^{-1/5}$.
  - `compute_qq_quantiles(returns, num_points=50)`
  - `compute_ecdf(returns, num_points=100)`
  - `compute_calendar_heatmap(prices_or_returns, is_prices=True)` supporting both price assets and interest rates/spreads without numerical overflow.
  - `compute_declustered_analogs(s, target_idx=-1, top_k=5, min_gap_days=20)` computing realized forward 5-day and 20-day returns.
- **Unit Test Suite**: `pipeline/tests/unit/test_context.py` (7/7 tests passing, 90% coverage).
- **Data Contract**: `pipeline/src/radar/export/schemas.py` and `web/src/lib/data/schemas.ts` defining `ContextData` and `ContextEnvelope`.
- **Exported Datasets**: Precomputed `/context/[id].json` across all 60 global series.

### Frontend Web UI
- **Components**:
  - `web/src/components/charts/DistributionHistogram.tsx` (C11 with dual-axis KDE and Normal overlay)
  - `web/src/components/charts/ContextQqPlot.tsx` (C12 with 45° reference line and tooltip departure delta)
  - `web/src/components/charts/CalendarHeatmap.tsx` (C25 with statutory warning banner)
- **Client & Page Views**:
  - `web/src/app/context/[id]/page.tsx`
  - `web/src/app/context/[id]/ContextClientView.tsx` with StatBlocks, MethodDrawer, AuditDrawer, and forward analogs table.

---

## 3. Verification & Acceptance
- Pytest unit tests: `pytest pipeline/tests/unit/test_context.py` -> 7 passed in 7.71s.
- Pytest contract tests: `pytest pipeline/tests/unit/test_contracts.py` -> 9 passed in 1.11s.
- Next.js static build prerenders all 60 static context routes cleanly.
