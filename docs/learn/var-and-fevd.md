# Learning Note: Vector Autoregressions and GFEVD Connectedness

## The Diebold-Yilmaz Spillover Framework
Modern financial markets are interconnected networks where localized liquidity or macro shocks propagate across borders. The Diebold-Yilmaz (2012) framework quantifies these shock transmissions using Vector Autoregressions (VAR).

### Vector Moving Average (VMA) Representation
Consider a stable $N$-variable VAR(1):
$$Y_t = \Phi Y_{t-1} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \Sigma)$$
In moving average form:
$$Y_t = \sum_{h=0}^\infty A_h \epsilon_{t-h}$$
where $A_0 = I_N$ and $A_h = \Phi^h$.

### Why Order-Invariance (GFEVD) Matters
Traditional Cholesky orthogonalization yields lower-triangular decompositions where the first variable cannot be affected by contemporaneous shocks from subsequent variables. The order chosen (e.g., US first vs. China first) changes the result arbitrarily.

The Pesaran-Shin (1998) Generalized Forecast Error Variance Decomposition solves this by allowing correlated shocks without orthogonalization:
$$\theta_{ij}(H) = \frac{\sigma_{jj}^{-1} \sum_{h=0}^{H-1} \left(e_i^T A_h \Sigma e_j\right)^2}{\sum_{h=0}^{H-1} \left(e_i^T A_h \Sigma A_h^T e_i\right)}$$

Because shocks are not orthogonal, row sums $\sum_{j=1}^N \theta_{ij}(H) \neq 1$. We normalize each row to sum to $100\%$:
$$\tilde{\theta}_{ij}(H) = \frac{\theta_{ij}(H)}{\sum_{j=1}^N \theta_{ij}(H)} \times 100\%$$

- **Total Connectedness Index (TCI)**:
  $$TCI = \frac{\sum_{i \neq j} \tilde{\theta}_{ij}(H)}{N}$$
- **Net Spillover**:
  $$NS_i = \sum_{j \neq i} \tilde{\theta}_{ji} - \sum_{j \neq i} \tilde{\theta}_{ij}$$
