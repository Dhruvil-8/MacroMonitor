# Learning Note: State-Space Models & Dynamic Parameter Tracking

## Intuition
Standard regression models assume that parameters (such as asset sensitivity $\beta$) are constant across time or only change slowly across fixed-width rolling windows. In financial reality, relationships change abruptly during policy regime shifts, liquidity squeezes, and macroeconomic shocks.

A **state-space model** separates a dynamic system into two structural equations:
1. **Observation Equation**: Relates observable asset returns $y_t$ to observable factor returns $x_t$ through a latent, time-varying parameter state vector $\beta_t$:
   $$y_t = x_t^\top \beta_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)$$

2. **Transition (State) Equation**: Specifies how the unobserved state $\beta_t$ evolves over time as a stochastic Markov process (typically a Gaussian random walk):
   $$\beta_t = \beta_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q)$$
   where $Q$ is the process noise covariance matrix governing parameter drift flexibility.

## Filtered vs. Smoothed Estimates
- **Filtered State $\hat{\beta}_{t|t}$ (Real-Time)**:
  Estimates parameter $\beta_t$ conditioning **only on observations up to and including time $t$** ($\mathcal{Y}_t = \{y_1, \dots, y_t\}$). This is the only legitimate estimate for live monitoring, real-time attribution, and trading evaluation. It guarantees zero look-ahead bias.
- **Smoothed State $\hat{\beta}_{t|T}$ (Historical Reconstruction)**:
  Estimates $\beta_t$ conditioning on the entire sample history up to time $T > t$. Smoothing produces cleaner, lower-variance historical paths, but uses future information and must always be explicitly labeled as "historical smoothed (uses later data)."

## In Market Radar
In the **Move Decomposer** and **Drivers Lab** (Module P5), we evaluate both the Kalman filter dynamic $\beta_{t|t}$ and rolling OLS $\beta_{\text{OLS}}$ on a walk-forward out-of-sample $R^2$ basis. The system dynamically reports the superior estimator badge and maintains rigorous point-in-time discipline.
