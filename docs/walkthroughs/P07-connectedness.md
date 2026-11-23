# Walkthrough: P7 Connectedness (Diebold-Yilmaz Spillover Map)

## Overview
Module P7 measures directional shock transmission and systemic interdependence across launch economies:
1. **VAR & Generalized Forecast Error Variance Decomposition (GFEVD):** Fits a vector autoregression on weekly USD returns across 8 launch economies over a rolling 260-week window. Applies order-invariant Pesaran-Shin GFEVD at a 10-week horizon.
2. **Spillover Metrics:**
   - **Total Connectedness Index (TCI):** Percentage of total forecast variance attributable to cross-country spillovers.
   - **Directional Spillovers:** "To Others" (transmissions) and "From Others" (receptions).
   - **Net Spillover:** $\text{Net}_i = \text{To}_i - \text{From}_i$, identifying structural shock transmitters and receivers.
3. **Connectedness UI (`/connectedness`):**
   - **Directed Heatmap:** From/To pairwise spillover matrix.
   - **Net Spillover Bars:** Diverging bar chart sorted by net shock transmission.
   - **TCI Timeline:** Historical evolution of global systemic integration.
   - **Chord View:** Circular graph representing directed international transmission links.

## Key Files
- `pipeline/src/radar/metrics/connectedness.py`: Diebold-Yilmaz GFEVD spillover engine.
- `web/src/app/connectedness/page.tsx`, `ConnectednessMatrix.tsx`, `NetSpilloverBars.tsx`, `ChordView.tsx`: Frontend components.
- `pipeline/tests/unit/test_connectedness.py`: Unit tests verifying matrix properties and invariant ordering.
- `docs/decisions/0030-p7-diebold-yilmaz-connectedness.md`: Architecture Decision Record.
- Learning note: `docs/learn/var-and-fevd.md`.
