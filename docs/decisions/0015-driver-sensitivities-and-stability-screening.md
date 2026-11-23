# 15. Macro Driver Sensitivities, Newey-West HAC Errors, and Stability Screening

Date: 2026-11-22
Status: Accepted

## Context

In the Driver Sensitivity layer (`/drivers`, Section 4.3 of `docs/MASTER_PLAN.md`), users examine how different national economies respond to global macro drivers (Brent crude, copper, gold, US dollar index, US 10-year yield, and rest-of-world equity).
However:
1. Financial market returns exhibit autocorrelation, volatility clustering, and heteroskedasticity. Standard OLS standard errors underestimate variance, producing artificially narrow confidence intervals.
2. Sensitivity coefficients can drift or flip signs during historical macro regime changes. Presenting a single point estimate without stability context can mislead research decisions.
3. Multi-collinearity among macro factors (e.g. commodities and dollar moves) can blow up coefficient variance.

## Decision

1. We estimate rolling multi-factor regressions over **250-day** (~1 year) and **120-day** (~6 months) windows using **Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent)** covariance with lag truncation $L = \min(5, \lfloor N/10 \rfloor)$.
2. If the Variance Inflation Factor (VIF) for any factor exceeds 5.0, the pipeline falls back to L2 Ridge regression ($\alpha = 1.0$) with conservative standard errors.
3. We define the **Stability Metric**:
   $$\text{stability} = \frac{\sum_{t} \mathbb{I}(\text{sign}(\beta_t) == \text{sign}(\beta_{\text{current}}))}{N_{\text{windows}}}$$
   Coefficients with stability $< 70\%$ or whose 95% confidence interval spans zero are visually hatched and faded in the sensitivity heatmap (C06).
4. We pre-register structural economic priors in `pipeline/config/drivers.yaml` (e.g., negative oil beta for oil importers such as India, Germany, and Japan, but positive oil beta for exporters like Canada and Brazil). The UI explicitly displays an alignment badge (`Aligned ✓` vs `Divergent ✕`).
5. We provide uncertainty visualizations:
   - **Forest Plot (C08)**: 95% Newey-West confidence intervals across all drivers for any selected economy.
   - **Rolling Beta Drift (C07)**: Trailing time-series chart with confidence ribbons.
   - **Bivariate Scatter (C09)**: Daily return scatter with OLS fit and confidence band.

## Consequences

- Statistical rigor: eliminates false precision by explicitly displaying estimation uncertainty.
- Editorial discipline: unstable relationships cannot masquerade as solid macro truths.
- Alignment with `MASTER_PLAN.md`: satisfies all criteria for Milestone 8 and charts C06, C07, C08, C09.
