# 36. C11 Return Distribution Percentile Alignment & US 10Y Beta Scaling

Date: 2026-11-24
Status: Accepted

## Context

Two analytical and visual discrepancies were audited in the presentation layers:

1. **C11 Return Distribution & Expanding Percentile ("98th percentile everywhere")**:
   - The Context Layer (`/context/[id]`) computes an expanding percentile on the asset price series $P_t$. Because equity indices (e.g. DAX, S&P 500, Nifty 50) have been trading near all-time highs, their price level percentile sits at the 97th–99th historical rank.
   - However, Chart C11 (*Distribution Modeling*) displays the empirical histogram and Gaussian KDE of **stationary daily returns** $r_t = \ln(P_t / P_{t-1})$.
   - The frontend `DistributionHistogram.tsx` incorrectly used the price level percentile `ctxData.percentile` to index return bins (`Math.round((clampedPct / 100) * (bins.length - 1))`). Consequently, on every equity asset near ATH, the markLine was pinned to Bin 24 (the extreme positive tail: $+3.5\%$ to $+4.0\%$) and labeled `Today (98th pct)`, even when today's actual return was negative (e.g. Nifty 50: $-1.76\%$, which is the 5.3th percentile).
   - Furthermore, pinning the markLine at the extreme right edge caused text collisions with the chart legend and boundary within the 280px container.

2. **C06 Cross-Economy Matrix & US 10Y Betas (`-0.0001`, `0.0000`)**:
   - In the multi-factor regression $y_t = \sum \beta_k X_{k, t} + \epsilon_t$, target equity return $y_t$ is in decimal log returns ($1.0\% = 0.010$).
   - Equity and commodity factors (ROW Equity, Brent, Copper, Gold, DXY) are also in decimal log returns ($1.0\% = 0.010$), yielding dimensionless elasticity betas ($\beta \approx 0.1$ to $0.5$).
   - Per `pipeline/config/drivers.yaml`, US 10Y yield changes are modeled in **basis points** ($\Delta Y_t \times 100$, where $1\text{ bp} = 1.0$).
   - Thus, $\beta_{\text{us10y}} = \frac{\Delta \text{Decimal Equity Return}}{\Delta \text{Basis Points}} \approx 10^{-4}$.
   - A beta of $-0.00010$ represents $-0.010\%$ equity return per $1$ bp, which corresponds to $-1.0\%$ equity return per $100$ bps ($+1.00\%$ yield shift) — a standard equity duration sensitivity.
   - When displayed unscaled alongside percentage-on-percentage betas with 3 or 4 decimals without explicit unit labeling, it collapsed to `-0.0001` or `0.0000`, creating the false appearance of an inert or defective calculation.

## Decision

1. **Context Export & C11 Realization Alignment**:
   - Added `latest_return` and `return_percentile` to `ContextData` in `pipeline/src/radar/export/writers.py` and `web/src/lib/data/schemas.ts`.
   - Updated `DistributionHistogram.tsx` to search `bins` for the exact bin closest to `latestReturn` rather than interpolating price rank.
   - Labeled the markLine explicitly with both the actual return and return rank: `Today: +2.1% (95th pct)` or `Today: -1.8% (5th pct)`.
   - Increased container height to 320px, added 55px top grid margin, centered the legend, and added badge padding to prevent collisions.
   - Renamed the Context overview StatBlock to `"Price Level Percentile"` (unit: `"Price Rank (ATH=100)"`) to distinguish asset price rank from daily return rank.

2. **US 10Y Beta Unit Disclosure & Heatmap Scaling**:
   - Preserved underlying regression mathematics and basis-point modeling per `drivers.yaml` and `MASTER_PLAN.md`.
   - Updated `DriverSensitivityHeatmap.tsx` column header to show `US 10Y Yield (per 1 bp)` and tooltip to include the 100 bps translation: `Beta: -0.0001 (-1.0% per 100 bps yield change)`.
   - Adjusted heatmap color intensity scaling for `us10y` ($\times 2500$ vs $\times 25$) so that economically significant rate sensitivities are highlighted rather than washed out to gray.
   - Annotated the summary table in `DriverClientView.tsx` with unit disclosure (`US 10Y (per bp)`) and an explicit footnote.

## Consequences

- Full mathematical transparency: eliminates false impressions of calculation error.
- Accurate distribution diagnostics: realizations now sit on the true return bin.
- No regression breaking changes: move decomposer and multi-factor regressions remain mathematically continuous.
