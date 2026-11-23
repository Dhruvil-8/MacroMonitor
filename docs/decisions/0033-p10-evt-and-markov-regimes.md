# 0033. P10 Extreme Value Theory Tails and Markov-Switching Regimes

Date: 2026-11-22
Status: Accepted

## Context
Financial asset returns exhibit pronounced non-Gaussian behavior: heavy tails (leptokurtosis) and volatility clustering (heteroskedasticity). Traditional normal distribution risk models systematically underestimate the probability and severity of tail events (the "black swan" underestimation problem). Furthermore, market dynamics alternate between low-volatility calm regimes and high-volatility stress regimes.

## Decision
1. **Peaks-Over-Threshold (POT) EVT Formulation:**
   - Losses ($L_t = -R_t$) are standardized by an expanding EWMA volatility ($\lambda = 0.94$) to filter out dynamic volatility clustering prior to fitting the static tail distribution.
   - Threshold $u$ is set at the 95th percentile of the standardized losses.
   - Excesses $Y = X - u \mid X > u$ are parameterized using the Generalized Pareto Distribution (GPD):
     $$G_{\xi, \beta}(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}$$
   - Parameters $\xi$ (shape) and $\beta$ (scale) are fitted via Maximum Likelihood Estimation (`scipy.stats.genpareto`).
   - We report 99.0% and 99.5% tail VaR and Expected Shortfall alongside Gaussian and empirical benchmarks, with the mean excess function $e(u)$ plotted to diagnose Pareto tail linearity.

2. **2-Regime Markov-Switching Volatility:**
   - A 2-state Markov regime-switching model with state-dependent variance is estimated on daily returns.
   - **Regime 0:** Low-volatility Calm state.
   - **Regime 1:** High-volatility Stress state.
   - Filtered marginal probabilities $P(S_t = 1 \mid Y_{1..t})$ are computed strictly point-in-time using the Hamilton filter to ensure zero look-ahead bias for real-time risk gauges.
   - Smoothed probabilities $P(S_t = 1 \mid Y_{1..T})$ via Kim's smoothing algorithm are provided for historical ribbon visualization.
   - Regime transition probabilities ($p_{00}, p_{01}, p_{10}, p_{11}$) and expected regime durations are computed: $E[D_i] = \frac{1}{1 - p_{ii}}$.

3. **Fallback Discipline:**
   - If numerical optimization does not converge within 50 iterations, an explicit heuristic fallback based on expanding rolling volatility quantiles is engaged and flagged transparently.

## Consequences
- Risk Lab gains rigorous parametric tail estimates that capture actual tail thickness rather than relying on Gaussian assumptions.
- Market participants can inspect whether current volatility reflects transitory noise or a persistent shift into the high-stress volatility regime.
