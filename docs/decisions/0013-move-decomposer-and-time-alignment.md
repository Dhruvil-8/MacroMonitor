# ADR 013: Move Decomposer and Time-Zone Alignment

## Status
Accepted (Governed by `docs/MASTER_PLAN.md`)

## Date
2026-11-22

## Context
Understanding "Why did the market move?" is the central question for macro analysts. Most dashboards present raw correlation tables or simultaneous multi-asset charts, which suffer from three major statistical flaws:
1. Correlation is not attribution: simultaneous moves do not explain how much of a target move is associated with specific global drivers versus domestic or residual shocks.
2. Currency confounding: converting foreign assets to USD directly confounds the domestic asset move with the exchange rate return.
3. Asynchronous market closes: Tokyo closes hours before London opens, and London closes before New York. Computing same-date returns between Asian indices and US macro drivers produces spurious leads and lags.

## Decision
Implement the **Move Decomposer** as the signature feature of Global Market Radar:
1. **Exact FX Identity for USD Lens:**
   $$(1 + r_{\text{usd}}) = (1 + r_{\text{local}}) \times (1 + r_{\text{fx}})$$
   In the USD lens, return is decomposed first into exact local and FX components. The local portion is then attributed across regression factors.
2. **Standard Macro Factor Set (`pipeline/config/drivers.yaml`):**
   - Rest-of-World Equity (ROW): Equal-weighted USD return of all other tracked countries, explicitly excluding the target to avoid mechanical overlap.
   - Oil (Brent), Copper, Gold.
   - US Dollar Index (DXY proxy).
   - US 10-Year Treasury Yield change (in basis points).
   - Target's own currency vs. USD (exact FX effect in USD lens).
3. **Point-in-Time Safe Prior-Window Estimation:**
   - Betas are estimated on 250 trading days ending the day *before* the horizon starts ($t' < t_{\text{start}}$), preventing lookahead bias.
   - Ordinary Least Squares (OLS) with Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) standard errors.
   - Variance Inflation Factor (VIF) check: If $\text{VIF} > 5$, use Ridge regression with a fixed, documented penalty and mark "regularized."
   - If out-of-sample $R^2 < 0.10$, label the decomposition "Mostly unexplained" rather than presenting false precision.
4. **Time-Zone Alignment Engine (`pipeline/config/calendars.yaml`):**
   - Each series defines its close timestamp rule: IANA time zone and local close time, or reference fix for FX/commodities.
   - Factor return window runs from last close at or before target's close on $d-1$ to last close at or before target's close on $d$:
     $$x_f(d) = \ln\left(\frac{P_f(\text{last close} \le C_T(d))}{P_f(\text{last close} \le C_T(d-1))}\right)$$
   - Verified via a synthetic lagged-response test.

## Consequences
- The platform provides clear, mathematically defensible attribution.
- Eliminates misleading simultaneous correlation artifacts across time zones.
