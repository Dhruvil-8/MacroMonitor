# Learning Note: PCA Sign Alignment and Non-Parametric Tail Risk

## 1. The Numerical Arbitrariness of Eigenvector Signs
Principal Component Analysis decomposes a standardized return covariance/correlation matrix $\mathbf{\Sigma} \in \mathbb{R}^{K \times K}$ into eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_K$ and orthogonal eigenvectors $\mathbf{v}_1, \dots, \mathbf{v}_K$:
$$\mathbf{\Sigma} \mathbf{v}_k = \lambda_k \mathbf{v}_k, \quad \|\mathbf{v}_k\|_2 = 1$$

Notice that if $\mathbf{v}_k$ satisfies this equation, then $-\mathbf{v}_k$ also satisfies it identically:
$$\mathbf{\Sigma} (-\mathbf{v}_k) = -\mathbf{\Sigma} \mathbf{v}_k = -\lambda_k \mathbf{v}_k = \lambda_k (-\mathbf{v}_k)$$

In standard numerical packages (such as LAPACK, SciPy, or NumPy's `linalg.eigh`), the choice of sign is determined by numerical truncation and machine precision. When calculating rolling PCA across sliding 250-day windows $[t-249, t]$, day-to-day changes in sample correlations can cause the algorithm to arbitrarily negate the eigenvector from one window to the next.

If left unhandled in a macro dashboard:
- Yesterday, the US market had a loading of $+0.42$ on PC1 (interpreted as "positive global growth").
- Today, the algorithm outputs a loading of $-0.42$ for the US and $-0.38$ for India.
The explained variance $\lambda_1$ is unchanged, but the factor loadings plot flips upside down, misleading users into believing a regime inversion occurred.

### The Anchor Market Sign-Alignment Rule
To preserve temporal consistency and economic intuition:
$$\mathbf{v}_k \leftarrow \text{sign}(\mathbf{v}_{k, \text{anchor}}) \cdot \mathbf{v}_k$$
By anchoring to the primary global equity market (S&P 500, $\text{anchor} = \text{spx}$), we ensure $\mathbf{v}_{1, \text{spx}} > 0$. Every other country's loading reflects its co-movement relative to the US anchor.

---

## 2. Fat Tails, Expected Shortfall, and the Tail Ratio
A Gaussian parametric model assumes daily log returns $r_t \sim \mathcal{N}(\mu, \sigma^2)$.
Under normality, the Value at Risk at confidence level $\alpha$ (e.g. 95%) is:
$$\text{VaR}_\alpha^{\text{Gauss}} = -(\mu + z_\alpha \sigma), \quad \text{where } z_\alpha = \Phi^{-1}(1 - \alpha) \approx -1.645$$
And the Expected Shortfall (tail conditional expectation) is:
$$\text{ES}_\alpha^{\text{Gauss}} = - \left( \mu - \sigma \frac{\phi(z_\alpha)}{1 - \alpha} \right)$$

Empirical financial returns, however, exhibit excess kurtosis ($\kappa > 0$), negative skewness ($\gamma < 0$), and power-law tails:
$$\mathbb{P}(|R| > x) \sim x^{-\alpha}, \quad \alpha \approx 3 \text{ to } 4$$

We define the **Tail Ratio**:
$$\text{Tail Ratio} = \frac{\text{Historical Expected Shortfall}_\alpha}{\text{Gaussian Expected Shortfall}_\alpha}$$
- In a pure Gaussian world: $\text{Tail Ratio} = 1.00$.
- In macro equity and commodity markets: $\text{Tail Ratio} \in [1.20, 1.65]$.
A Tail Ratio of $1.35$ indicates that when a tail event strikes, the actual average loss is $35\%$ deeper than classical risk models predict.

---

## 3. ARCH Volatility Clustering via Autocorrelation of Squared Returns
Efficient market hypothesis asserts that log price changes have negligible linear autocorrelation:
$$\text{Corr}(r_t, r_{t-k}) \approx 0 \quad \forall k \ge 1$$
However, volatility is strongly dependent:
$$\text{Corr}(r_t^2, r_{t-k}^2) > 0 \quad \text{for many lags } k$$
By computing the Autocorrelation Function (ACF) of both $r_t$ and $r_t^2$ alongside effective degrees of freedom confidence bands $\pm 1.96 / \sqrt{N_{\text{eff}}}$, we visually demonstrate that while direction cannot be predicted from past returns, risk and volatility persist in pronounced regimes.
