# Specification: Percentile, Z-Score, and Effective N (s03-percentile-zscore-neff)

## 1. Purpose
Provides historical context and standardization for any financial time series using strictly past observations and autocorrelation-adjusted sample size.

## 2. Mathematical Definition

### 2.1 Expanding Percentile Rank
For series $x_t$ evaluated at time $t$ using strictly prior history $s \in [0, t-1]$ (or $[0, t]$ where $t$ is included in the empirical CDF):
$$\text{Percentile}_t = \frac{1}{N_t} \sum_{s=1}^{N_t} \mathbf{1}_{\{x_s \le x_t\}} \times 100$$
Minimum required sample size: `min_obs = 750` (~3 years of daily trading history). If $N_t < \text{min\_obs}$, percentile returns NaN.

### 2.2 Expanding Z-Score (Anti-Lookahead)
Computed using expanding sample mean and standard deviation strictly excluding current point $t$:
$$\mu_{t-1} = \frac{1}{t-1}\sum_{s=1}^{t-1} x_s, \quad \sigma_{t-1} = \sqrt{\frac{1}{t-2}\sum_{s=1}^{t-1}(x_s - \mu_{t-1})^2}$$
$$z_t = \frac{x_t - \mu_{t-1}}{\sigma_{t-1}}$$
If $\sigma_{t-1} = 0$, $z_t = 0.0$.

### 2.3 Effective Sample Size ($N_{\text{eff}}$)
Financial metrics (especially rolling volatility and moving averages) exhibit high serial autocorrelation, rendering raw observation count $N$ misleadingly optimistic.
$$N_{\text{eff}} = N \frac{1 - \rho_1}{1 + \rho_1}$$
Where:
- $\rho_1 = \text{corr}(x_t, x_{t-1})$ is the lag-1 autocorrelation.
- $\rho_1$ is clipped to $[0.0, 0.99]$ (negative autocorrelation does not artificially inflate sample size above $N$).
- When $\rho_1 \to 1$, $N_{\text{eff}} \ll N$.
- Display requirement: Both $N$ and $N_{\text{eff}}$ must be presented in the Audit Drawer.
