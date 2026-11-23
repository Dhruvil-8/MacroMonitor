# Learning Note: Kalman Filter for Time-Varying Beta Estimation

## Overview
Traditional macro attribution relies on rolling-window Ordinary Least Squares (OLS) regressions (e.g. 60-day or 120-day trailing windows). Rolling OLS has severe limitations:
1. **Window-length arbitrariness**: A 60-day window reacts abruptly to single outliers and forgets information on day 61.
2. **Ghosting effects**: When a large shock exits the rolling window, the estimated beta shifts artificially even if market dynamics haven't changed.
3. **Delayed reaction**: Rolling averages lag true regime changes.

## The State-Space Formulation
We model the regression beta as an unobserved latent state that evolves smoothly over time:

$$y_t = x_t^T \beta_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)$$
$$\beta_t = \beta_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q)$$

Here:
- $y_t$ is the target asset's return at time $t$.
- $x_t$ is the vector of macro drivers (e.g., Dollar, Yields, Commodities).
- $\beta_t$ is the true driver sensitivity at time $t$.
- $Q$ is the state noise covariance ($Q = q \cdot \sigma_\epsilon^2 \cdot I$), controlling how rapidly betas can drift.

## Filtered vs. Smoothed Estimates
- **Filtered ($\beta_{t|t}$)**: Uses information up to time $t$ only. This is mandatory for real-time attribution and backtesting without look-ahead bias.
- **Smoothed ($\beta_{t|T}$)**: Uses the entire historical dataset (including future dates $t+1 \dots T$). Permissible only for historical retrospective analysis, with prominent visual disclosure.

## Walk-Forward Validation
To avoid model bias, we evaluate one-step-ahead out-of-sample forecast errors:
$$e_{t} = y_t - x_t^T \hat{\beta}_{t|t-1}$$
The platform compares out-of-sample $R^2$ against rolling OLS and automatically selects the superior model for each asset.
