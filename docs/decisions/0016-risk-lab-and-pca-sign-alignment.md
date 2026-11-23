# ADR 0016: Risk Lab Layer and PCA Sign Alignment Invariant

## Context
Section 4.4 of `docs/MASTER_PLAN.md` specifies an institutional Risk Lab layer delivering non-parametric volatility cones across {10D, 20D, 60D, 120D, 250D} horizons, heavy-tail Expected Shortfall and Value at Risk at 95% and 99% levels, ARCH volatility clustering (ACF of raw returns vs squared returns), continuous underwater drawdowns with crisis episode tracking, global PCA market synchronization, and downside stress correlation premiums.

In financial econometric factor modeling using Principal Component Analysis (PCA), eigenvectors obtained from numerical solvers (e.g. `numpy.linalg.eigh`) have arbitrary sign: if $v$ is an eigenvector, $-v$ is equally valid ($\mathbf{\Sigma} v = \lambda v \iff \mathbf{\Sigma}(-v) = \lambda(-v)$). When rolling PCA across sequential daily windows, solvers may arbitrarily flip component signs between consecutive days ($v_t \approx -v_{t-1}$), leading to jarring, uninterpretable inversions in factor loading charts and visual confusion.

## Decision
1. **PCA Sign Alignment Invariant**:
   For the global equity market factor structure across 10 economies, we enforce an explicit economic invariant: the factor loading for the anchor market (United States / S&P 500) on PC1 and PC2 must be strictly positive:
   $$\text{If } v_{1, \text{us}} < 0, \quad v_1 \leftarrow -v_1$$
   $$\text{If } v_{2, \text{us}} < 0, \quad v_2 \leftarrow -v_2$$
   This guarantees temporal stability across rolling windows and preserves an intuitive economic interpretation where PC1 represents the positive global equity market mode.

2. **Tail Ratio Crash Metric**:
   We define the non-parametric fat-tail crash vulnerability ratio as:
   $$\text{Tail Ratio} = \frac{\text{Historical Expected Shortfall}_{95\%}}{\text{Gaussian Expected Shortfall}_{95\%}}$$
   Where $\text{Tail Ratio} > 1.0$ quantitatively establishes that empirical market crash losses significantly exceed standard bell-curve predictions.

3. **ARCH Volatility Clustering via ACF**:
   We compute autocorrelation functions up to 20 lags for both raw returns $r_t$ and squared returns $r_t^2$ with two-sided 95% Bartlett confidence bands adjusted by the effective sample size $N_{\text{eff}}$:
   $$\text{Band} = \pm \frac{1.96}{\sqrt{N_{\text{eff}}}}$$

4. **Downside Correlation & Stress Premium**:
   We measure conditional downside correlation $\rho_{\text{downside}}$ by restricting returns to sessions where the benchmark is in its worst 10% tail, defining the Stress Correlation Premium as $\Delta \rho = \rho_{\text{downside}} - \rho_{\text{all}}$.

## Consequences
- Complete, publication-grade risk profile available for all 10 launch equity indexes, plus key macro commodities, rates, FX, and volatility series.
- Factor loading charts remain stable and monotonic over time without arbitrary visual inversions.
- Data consumers can immediately inspect whether international diversification holds or breaks down under market stress.
