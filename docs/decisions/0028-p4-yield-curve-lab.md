# 0028. P4 Yield Curve Lab & Nelson-Siegel 3-Factor Modeling

Date: 2026-11-22
Status: Accepted

## Context
Module P4 of `PUSH_PACK_PLAN.md` deepens the rates and sovereign yields layer using US Treasury constant maturity series and TIPS inflation breakevens.

## Decisions Made
1. **Nelson-Siegel Diebold-Li Formulation**:
   $$y(\tau) = L + S \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau}\right) + C \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}\right)$$
   where $\lambda = 0.0609$ is fixed based on standard Diebold-Li literature (maximizing curvature loading at $\tau = 30$ months).
2. **Daily Least Squares Fit**: Estimates Level ($L$), Slope ($S$), and Curvature ($C$) daily, accompanied by root-mean-square fit error (RMSE) and historical expanding percentiles.
3. **Inversion Tracker**: Formulates 10Y-2Y, 10Y-3M, and 30Y-5Y spreads with discrete inversion flags (`inverted: bool`).
4. **10-Year Yield Anatomy**: Decomposes nominal yield moves into real yield (TIPS) and breakeven inflation changes ($\Delta \text{Nominal} \approx \Delta \text{Real} + \Delta \text{Breakeven}$), fulfilling the Fisher identity.
5. **UI Curve Ribbon**: Renders multi-date curve snapshots alongside dense Nelson-Siegel fitted smooth lines in `web/src/app/rates/page.tsx`.

## Verification
- Unit tests in `pipeline/tests/unit/test_yield_curve.py` prove exact parameter recovery, Fisher identity consistency, and spread inversion tracking.
