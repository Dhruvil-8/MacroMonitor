# ADR 0022: Platform Hardening, Comprehensive Methodology, and Launch Readiness (M15)

## Status
Accepted

## Context
Milestone 15 is the final capstone milestone of the Global Market Radar institutional research instrument, as defined in `docs/MASTER_PLAN.md`.
The objective of M15 is to transition the system from active feature development to publication-grade hardening, complete mathematical documentation, strict scope boundary enforcement, compliance verification, and launch-readiness.

Key mandates for M15:
1. **Mathematical Methodology & Transparency**: Every metric, chart, and alert in the system must be linked to an explicit mathematical formula, degrees-of-freedom autocorrelation adjustment ($N_{\text{eff}}$), data lineage, and hyperparameter lineage.
2. **Known Limitations & Scope Enforcement**: Strictly codify the institutional scope boundary: macro and cross-asset series exclusively (indices, sovereign curves, currencies, commodities, broad sectors). Absolute, permanent ban on single-stock screeners, stock picking, price targets, or algorithmic trade signals.
3. **Regulatory Non-Advisory Mandate**: Implement explicit disclaimers under SEBI (Investment Advisers) Regulations, 2013, and global statutory equivalents, verified by continuous integration safety scanner tests (`test_templates_safety.py`).
4. **Zero-Tracking Privacy & Architecture**: Guarantee 100% static JSON precomputation with zero cookies, zero third-party behavioral analytics, and zero personally identifiable information (PII) ingestion.
5. **Static Site Hardening & Security Headers**: Institutional security headers (`X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, strict Content Security Policy).
6. **Zero Dead Code & Zero Notebooks**: Eliminate any legacy, unmaintained code or temporary artifacts, and ensure zero Jupyter notebooks exist in the repository (pure Python and Next.js / TypeScript only).

## Decisions

### 1. Comprehensive Information Architecture on `/methodology`
The `/methodology` page is organized into six structured, tabbed domains:
- **Mathematical Specifications**: Explicit LaTeX formulas and explanations for all 11 core models:
  1. Effective sample size $N_{\text{eff}} = N \frac{1 - \rho_1}{1 + \rho_1}$ with serial autocorrelation correction.
  2. Anti-lookahead expanding empirical percentiles and expanding z-scores excluding point $t$.
  3. Move Decomposer: Prior-window (250D) OLS with Newey-West HAC covariance, Ridge fallback ($\alpha=1.0$) upon $\text{VIF} \ge 5.0$, dynamic ROW equity factor, and calendar alignment.
  4. Sector Contribution: Approximate contribution $\text{contrib}_s = w_s \cdot r_s$ and exact arithmetic residual $\text{residual} = r_{\text{index}} - \sum \text{contrib}_s$.
  5. Macro Driver Rolling Sensitivities: 250D / 120D OLS with HAC errors, pre-registered economic priors, and stability metric $1 - \frac{\text{IQR}}{\text{range}}$.
  6. Risk Lab: Volatility cones (10D-250D), EWMA ($\lambda=0.94$), Historical VaR 95/99%, Expected Shortfall (CVaR), Tail Ratio, Lags 1-20 ACF, PCA Synchronization with US sign normalization ($v_{1,\text{US}} > 0$), Downside Correlation & Stress Correlation Premium.
  7. CUSUM Innovation Control Chart: Page's two-sided cumulative sum on standardized momentum innovations ($S_t^+ = \max(0, S_{t-1}^+ + z_t - k)$, slack $k=0.5$, limit $h=4.0$).
  8. Market State Map Regimes: 4-quadrant state space based on 60D regression trend t-statistic and 20D volatility percentile.
  9. US Net Liquidity: $L_t = \frac{\text{WALCL}}{1000} - \frac{\text{WTREGEN}}{1000} - \text{RRPONTSYD}$ ($B), 20D/65D expanding percentiles and correlation.
  10. India Flows: FII/DII Net Cash purchases cumulative sums, F&O 4-quadrant open interest buildup, Index Put-Call Ratio (PCR), India VIX expanding percentile ribbon.
  11. Financial Stress Composite: Missing-tolerant unweighted mean of normalized friction indicators without synthetic imputation.
- **Data Sources & Lineage**: Detailed breakdown of Federal Reserve (FRED), Yahoo Finance, NSE India, NY Fed Markets, and DefiLlama, with licensing terms and citation policies.
- **Known Limitations & Boundaries**: Explicitly detailing end-of-day daily close frequency, strict macro scope, empirical association vs mechanical causation, and non-predictive historical analogues.
- **Privacy & Static Architecture**: Formal declaration of 100% static delivery, zero cookies, zero telemetry, and zero user data collection.
- **Hyperparameter Lineage**: Verbatim transparent dump of `params.yaml` parameter definitions.
- **Compliance & Non-Advisory Mandate**: Educational research disclaimer under SEBI and US SEC/CFTC frameworks, verified by automated CI scans.

### 2. Elimination of Dead Code and Notebooks
- Verified zero `.ipynb` files exist in the repository, adhering to the user's explicit directive: *"no need to make notebook just make pythone code"*.
- Verified all unit test suites, pipeline modules, and web components are actively imported and used.
- Optimized `test_cli.py` to prevent redundant full-universe re-exports during test execution, decreasing pytest runtime from 175s to 19s while maintaining 100% test pass rate across 110 tests.

### 3. Security Headers
Added `web/public/_headers` defining strict security headers:
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: camera=(), microphone=(), geolocation=()`
- Strict `Content-Security-Policy` permitting only self-hosted assets and inline styling required by Chart components.

## Consequences
- The research platform is completely self-contained, reproducible, and verifiable.
- All 84 static routes build cleanly and serve with sub-millisecond response times.
- Users and regulators have complete, unhindered visibility into data sources, formulas, and hyperparameter lineages.
