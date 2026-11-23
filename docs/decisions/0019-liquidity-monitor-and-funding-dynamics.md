# ADR 0019: Global Liquidity Monitor & Central Bank Funding Dynamics

## Context
Section 8.8 and Section 13.6 (Chart C27: Stacked area plus line) of `docs/MASTER_PLAN.md` require an institutional-grade Global Liquidity & Funding Monitor (`/liquidity`). The module must track the net domestic balance sheet capacity of the Federal Reserve circulating in commercial bank reserves, measure historical expanding change percentiles across 4-week and 13-week horizons, monitor stablecoin circulating supply, and provide an equity overlay with rolling correlation while enforcing strict causality disclaimers.

## Decision
1. **Net Liquidity Mathematical Formulation**:
   We define US Net Liquidity in Billions of USD:
   $$L_t = \frac{\text{WALCL}}{1000} - \frac{\text{WTREGEN}}{1000} - \text{RRPONTSYD}$$
   Where:
   - `WALCL`: Federal Reserve Total Assets (FRED, Millions USD $\rightarrow$ divided by $1,000$ for Billions USD).
   - `WTREGEN`: US Treasury General Account operating cash at the Fed (FRED, Millions USD $\rightarrow$ divided by $1,000$ for Billions USD).
   - `RRPONTSYD`: Overnight Reverse Repurchase Agreements (FRED / NY Fed, Billions USD).

2. **Frequency Alignment & Forward-Filling**:
   While `RRPONTSYD` is published daily by the New York Fed, `WALCL` and `WTREGEN` are weekly H.4.1 Wednesday releases. We union all timestamps and forward-fill weekly points up to `max_ffill_days = 7`, preventing lookahead bias while maintaining synchronized daily multi-asset alignment.

3. **Expanding Percentiles of 4-Week and 13-Week Changes**:
   Rather than raw levels which possess non-stationary upward drift over multi-decade regimes, we track changes:
   - 4-Week Change ($\Delta_{20} = L_t - L_{t-20}$)
   - 13-Week Change ($\Delta_{65} = L_t - L_{t-65}$)
   Each delta is ranked against its historical expanding distribution ($\text{pct}_{20}$ and $\text{pct}_{65} \in [0, 100]$), exposing whether central bank reserves are rapidly expanding or draining relative to past episodes.

4. **Equity Overlay and Statutory Caveat**:
   We overlay Net Liquidity against the S&P 500 (`^GSPC`) and compute rolling 60-day Pearson correlation $\rho_{60}$. Because central bank reserves are often misconstrued by retail market commentators as mechanical stock-buying signals, we mandate that every display carry the prominent institutional caveat:
   *"Association, not mechanical causation: while changes in central bank reserves historically track shifts in equity market multiples, net liquidity is not a deterministic trading trigger. Foreign FX reserves, commercial bank loan creation, and leverage also drive broad financial conditions."*

5. **Traceability and Effective Sample Size ($N_{\text{eff}}$)**:
   The Audit Drawer reports exact inputs, parameters, sample size $N$, and degrees-of-freedom adjustment:
   $$N_{\text{eff}} = N \frac{1 - \rho_1}{1 + \rho_1}$$
   where $\rho_1$ is the lag-1 autocorrelation of net liquidity changes.

## Consequences
- The `/liquidity` page provides complete visibility into US central bank balance sheet plumbing, reverse repo exhaustion, and Treasury cash absorption.
- Precomputed datasets (`/v1/liquidity.json`) load instantly on the client without runtime calculation overhead.
