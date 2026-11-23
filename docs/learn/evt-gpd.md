# Learning Note: Extreme Value Theory (EVT) and the Generalized Pareto Distribution (GPD)

## 1. Why Normal Distributions Fail for Tail Risk
Standard risk models (such as classical parametric Value-at-Risk) assume that financial returns follow a Gaussian normal distribution. In a Gaussian distribution:
- A $3\sigma$ move occurs approximately once every 370 trading days (1.5 years).
- A $5\sigma$ move occurs approximately once every 14,000 years.
- A $7\sigma$ move has a probability of $1.3 \times 10^{-12}$ (virtually impossible in human history).

In real equity and commodity markets, $4\sigma$ to $8\sigma$ daily moves occur every few years during financial crises (e.g., 1987, 1998, 2008, 2020). Modeling returns with a Gaussian distribution drastically underestimates tail risk and leads to severe undercapitalization.

## 2. The Pickands-Balkema-de Haan Theorem
Just as the Central Limit Theorem states that the sum of independent random variables converges to a Normal distribution regardless of their underlying distribution, the **Pickands-Balkema-de Haan Theorem** states that for a large class of underlying distributions, the distribution of excesses above a sufficiently high threshold $u$ converges asymptotically to the **Generalized Pareto Distribution (GPD)**:

$$P(X - u \le y \mid X > u) \approx G_{\xi, \beta}(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}$$

Where:
- $\xi$ is the **shape parameter** (tail index).
  - $\xi > 0$: Heavy-tailed (Fréchet type, polynomial decay $x^{-1/\xi}$, typical of financial markets).
  - $\xi = 0$: Exponential tail (Gumbel type, light tails like Gaussian/lognormal).
  - $\xi < 0$: Short-tailed (Weibull type, finite upper bound).
- $\beta > 0$ is the **scale parameter**.
- $u$ is the **threshold cutoff** (typically chosen around the 95th percentile).

## 3. Peaks-Over-Threshold (POT) Workflow
1. **Vol-Standardization:** Filter out volatility clustering using an expanding EWMA ($\lambda = 0.94$):
   $$\tilde{L}_t = \frac{L_t}{\hat{\sigma}_t^{\text{EWMA}}}$$
2. **Threshold Selection:** Set $u$ at the 95th percentile of standardized losses $\tilde{L}$.
3. **Fit GPD:** Estimate $(\xi, \beta)$ on the sample of excesses $Y_i = \tilde{L}_i - u$ via Maximum Likelihood.
4. **Compute Tail Quantile (VaR):**
   $$\text{VaR}_\alpha = u + \frac{\beta}{\xi} \left[ \left(\frac{N}{N_u}(1 - \alpha)\right)^{-\xi} - 1 \right]$$
5. **Compute Expected Shortfall (ES):**
   $$\text{ES}_\alpha = \frac{\text{VaR}_\alpha}{1 - \xi} + \frac{\beta - \xi u}{1 - \xi}$$

## 4. Mean Excess Plot
To verify threshold stability, we examine the sample Mean Excess function:
$$e(u) = \frac{1}{N_u} \sum_{i=1}^{N_u} (X_i - u)$$
For a Generalized Pareto distribution with $\xi > 0$, $e(u)$ is a straight line with positive slope $\frac{\xi}{1 - \xi}$. An upward-sloping mean excess plot confirms heavy-tailed Pareto behavior.
