# ADR 0017: Historical Context Pages and Kernel Density Estimation (Silverman's Rule)

## Context
Section 4.5 and Section 5.1 (Charts C11, C12, C20, C25, C26) of `docs/MASTER_PLAN.md` specify historical context pages for every tracked asset (`/context/[id]`). The context pages must provide:
1. Return distribution histogram overlaid with non-parametric Gaussian Kernel Density Estimate (KDE) and normal bell curve equivalent.
2. Quantile-Quantile (QQ) plot exposing tail departures.
3. Calendar monthly seasonality heatmap with mandatory statistical chance caveats.
4. Declustered historical analogs (minimum 20-day temporal separation) with realized subsequent 5-day and 20-day return outcomes.

## Decision
1. **Silverman's Rule of Thumb for Gaussian KDE**:
   To prevent arbitrary subjective bandwidth tuning, we strictly implement Silverman's robust bandwidth estimator:
   $$h = 0.9 \cdot \min\left(\hat{\sigma}, \frac{\text{IQR}}{1.34}\right) \cdot n^{-1/5}$$
   Where $\text{IQR} = q_{0.75} - q_{0.25}$. Using the interquartile range normalized by $1.34$ prevents outliers and fat-tail crash days from artificially inflating the bandwidth and oversmoothing the distribution mode.

2. **Non-Negotiable Calendar Seasonality Warning**:
   Financial time series seasonality (e.g. "January effect", "September effect") is susceptible to small-sample data mining bias ($N \approx 10-25$ observations per month across a decade). We mandate that every calendar heatmap display the clear warning:
   *"May reflect chance; sample size too small for statistical significance. Displayed strictly for descriptive context."*

3. **Forward Analog Outcomes**:
   For each of the top 5 declustered historical episodes ($|x_t - x_{\text{hist}}| \to \min$ with $\Delta t \ge 20$ days), we compute the realized forward 5-day and 20-day returns from that historical point, providing objective empirical evidence of subsequent market trajectory rather than speculative predictions.

## Consequences
- Every tracked global asset (equities, rates, currencies, commodities) possesses a dedicated context page with publication-grade distribution modeling.
- The browser never executes heavy KDE kernels or analog loops; all grids are precomputed in the Python pipeline and delivered via compact JSON.
