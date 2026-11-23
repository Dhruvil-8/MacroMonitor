# 0027. P5 Time-Varying Betas via Kalman Filter

Date: 2026-11-22
Status: Accepted

## Context
Module P5 of `PUSH_PACK_PLAN.md` enhances the Move Decomposer and Drivers layer by introducing a state-space Kalman filter estimator alongside rolling OLS.

## Decisions Made
1. **State-Space Model**:
   - Observation: $y_t = x_t^T \beta_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_{\epsilon}^2)$
   - State transition: $\beta_t = \beta_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q)$ where $Q = q \cdot \sigma_{\epsilon}^2 \cdot I$
2. **Point-in-Time Discipline**: Live attributions use the filtered state estimate $\beta_{t|t}$ (strictly conditioned only on information up to $t$). Filtered estimates at time $t$ remain strictly invariant when future data is introduced.
3. **Out-of-Sample Walk-Forward Selection**:
   - Compares one-step-ahead prediction errors between Kalman and rolling OLS over the trailing evaluation window.
   - The estimator achieving higher out-of-sample $R^2$ is selected.
4. **UI Integration**:
   - `MoveDecomposerWaterfall.tsx` displays the estimator badge (`⚡ Kalman Filter Betas` vs `Rolling OLS`) with hover tooltip detailing out-of-sample $R^2$ evaluation.

## Verification
- Unit tests in `pipeline/tests/unit/test_kalman.py` verify constant convergence, random-walk tracking, zero-lookahead invariance, and out-of-sample selection logic.
