# Walkthrough: P4 Yield Curve Lab

## Overview
Module P4 implements a comprehensive rates and term-structure laboratory:
1. **Nelson-Siegel 3-Factor Model (Diebold-Li form):**
   $$y(\tau) = L + S \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau}\right) + C \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}\right)$$
   with $\lambda = 0.0609$ (monthly maturities). Daily OLS estimation produces Level ($L$), Slope ($S$), and Curvature ($C$) with RMSE fit tracking.
2. **Benchmark Spreads:** Tracks 10Y-2Y, 10Y-3M, and 30Y-5Y spreads with inversion detection badges.
3. **10-Year Yield Anatomy:** Decomposes nominal yield into TIPS real yield and breakeven inflation rate.
4. **Yield Curve Lab UI (`/rates`):** Nelson-Siegel smooth ribbon, 10Y anatomy lines, par yield heatmap across maturities, factor cards, and spread monitors.

## Key Files
- `pipeline/src/radar/metrics/yield_curve.py`: Nelson-Siegel fitting and spread analytics engine.
- `pipeline/config/curves.yaml`: Sovereign yield curve configurations.
- `web/src/app/rates/page.tsx`, `YieldHeatmap.tsx`, `CurveRibbon.tsx`, `FactorPanel.tsx`: Frontend interfaces.
- `pipeline/tests/unit/test_yield_curve.py`: Unit tests verifying Nelson-Siegel parameter recovery and spread properties.
- `docs/decisions/0028-p4-yield-curve-lab.md`: Architecture Decision Record.
- Learning notes: `docs/learn/nelson-siegel.md` and `docs/learn/yield-curve-basics.md`.
