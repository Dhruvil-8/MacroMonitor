# Walkthrough: P5 Time-Varying Betas (Kalman Filter)

## Overview
Module P5 upgrades the Move Decomposer and Macro Drivers engine from rolling boxcar regressions to a dynamic state-space Kalman filter:
1. **State-Space Formulation:**
   - Observation: $y_t = x_t^\top \beta_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)$
   - State Transition: $\beta_t = \beta_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q)$
2. **Point-in-Time Filtered Estimates:** Only filtered $\hat{\beta}_{t|t}$ is used for live attribution, eliminating look-ahead bias.
3. **Walk-Forward Model Selection:** Compares out-of-sample $R^2$ of Kalman filter vs. rolling OLS across the trailing 12 months, dynamically selecting the superior estimator and badging it on `/` (Today).
4. **Uncertainty Bands:** Calculates posterior parameter standard errors from the filter covariance matrix $P_{t|t}$.

## Key Files
- `pipeline/src/radar/metrics/kalman.py`: Kalman filter state-space estimator and EM variance learning.
- `pipeline/src/radar/metrics/decomposition.py`: Integration into Move Decomposer.
- `web/src/components/charts/MoveDecomposerWaterfall.tsx`: UI badge and attribution display.
- `pipeline/tests/unit/test_kalman.py`: Simulation recovery tests and anti-lookahead checks.
- `docs/decisions/0027-p5-kalman-filter-betas.md`: Architecture Decision Record.
- Learning notes: `docs/learn/kalman-filter.md` and `docs/learn/state-space-models.md`.
