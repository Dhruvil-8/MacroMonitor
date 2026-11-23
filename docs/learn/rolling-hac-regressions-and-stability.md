# Pedagogical Learning Note: Rolling HAC Regressions, Stability, and Uncertainty

## 1. Why Standard OLS Fails in Macro Finance

When analyzing how equity indices respond to macro factors (e.g. oil, copper, the US dollar, or sovereign yields), standard Ordinary Least Squares (OLS) makes two assumptions that are routinely violated in financial return series:
1. **Homoskedasticity:** $\text{Var}(\epsilon_t \mid X_t) = \sigma^2$ (constant error variance across time).
2. **Serial Independence:** $\text{Cov}(\epsilon_t, \epsilon_{t-k} \mid X) = 0$ for $k \ne 0$ (no autocorrelation).

During market turbulence, volatility clusters dramatically (heteroskedasticity), and daily market microstructure creates residual serial dependence. Under standard OLS formulas, the estimated standard errors $\text{SE}(\hat{\beta})$ are artificially deflated—sometimes by a factor of 2 or 3. This leads analysts to claim statistical significance where none exists.

## 2. Newey-West HAC Covariance

To prevent this false sense of precision, we estimate standard errors using the **Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC)** estimator:

$$\text{Var}_{\text{HAC}}(\hat{\beta}) = (X'X)^{-1} \hat{\Omega} (X'X)^{-1}$$

where the spectral density matrix at frequency zero is estimated with Bartlett kernel weights:
$$\hat{\Omega} = \hat{\Gamma}_0 + \sum_{j=1}^L \left(1 - \frac{j}{L+1}\right) (\hat{\Gamma}_j + \hat{\Gamma}_j')$$
$$\hat{\Gamma}_j = \sum_{t=j+1}^T e_t e_{t-j} x_t x_{t-j}'$$

We set the truncation lag $L = \min(5, \lfloor N/10 \rfloor)$, providing robust confidence intervals $[CI_{\text{lower}}, CI_{\text{upper}}]$:
$$CI_{95\%} = \hat{\beta} \pm 1.96 \times \text{SE}_{\text{HAC}}(\hat{\beta})$$

## 3. The Stability Metric

Even when a coefficient is statistically significant over a 250-day window, the relationship may be unstable across economic cycles (e.g., gold transitioning from inflation hedge to safe-haven liquidity source).

We define the **Stability Ratio**:
$$\text{stability} = \frac{\sum_{t} \mathbb{I}(\text{sign}(\beta_t) == \text{sign}(\beta_{\text{current}}))}{N_{\text{windows}}}$$

### Operational Rule
- If $\text{stability} \ge 0.70$ and $0 \notin CI_{95\%}$, the macro sensitivity is considered **Persistent & Statistically Significant**.
- If $\text{stability} < 0.70$ or $0 \in CI_{95\%}$, the cell is **hatched and faded** on the heatmap (C06), warning researchers that the relationship is historically ephemeral.
