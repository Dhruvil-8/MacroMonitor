# 4. Pulse Page and Traceability Architecture

Date: 2026-11-21
Status: Accepted

## Context
Milestone M4 delivers the flagship Pulse view (`/`), presenting cross-asset macroeconomic health at a glance while fulfilling the core brand promise: "Every number is traceable."

## Decisions
1. **Interactive Traceability Matrix (`AuditDrawer`)**:
   - Universal drawer triggered from any chart card, stat block, or table row.
   - Discloses exact mathematical definition, hyperparameters from `params.yaml`, input series with source attribution, raw observation count ($N$), effective sample size ($N_{\text{eff}}$), percentile rank, code version, and deep-link to `/methodology`.

2. **Market State Map**:
   - 2D scatter of 60 core assets across signed $t$-statistic of 60-day log price slope (X-axis) and 20-day realized volatility percentile (Y-axis).
   - Four discrete statistical regimes: *Orderly trend*, *Violent trend*, *Quiet range*, and *Choppy*.
   - Point sizing driven by absolute 20-day momentum change.

3. **Volatility Landscape**:
   - Multi-horizon matrix across 10D, 20D, and 60D annualized volatility with compression/expansion trend ratio.
   - CVD-safe heatmap styling with tabular right-aligned monospace numbers.

4. **Zero-Layout-Shift Static Hydration**:
   - Static shells prerendered at build time with loading states matching final chart heights.
   - Client loaders hydrate real data from `/data/v1/` with Zod runtime validation and automated staleness alerting (>36 hours).
