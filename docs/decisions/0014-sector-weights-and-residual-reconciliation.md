# 14. Published Factsheet Sector Weights and Exact Residual Reconciliation

Date: 2026-11-22
Status: Accepted

## Context

In the Sector Layer (`/sectors`, Section 4.2 of `docs/MASTER_PLAN.md`), users require an intuitive attribution of benchmark index returns (e.g. S&P 500, Nifty 50) across constituent macro sectors.
However:
1. True index constituents rebalance and float continuously with daily intraday market movements.
2. Full stock-level constituent weights are proprietary, change daily, and fall outside our strict macro-only scope (Section 0.1: no individual stocks/screeners).
3. Directly computing $\sum_s w_s r_s$ with fixed monthly weights will drift away from the published benchmark return $r_{\text{index}}$.

## Decision

1. We use official published index factsheet sector weights (monthly) stored in `pipeline/config/sector_weights.yaml` with explicit source attribution and `as_of` dates:
   - US: S&P Dow Jones Indices S&P 500 Factsheet (11 GICS sectors).
   - India: NSE Indices Nifty 50 Factsheet (7 major tracked sectors).
2. Approximate contribution is defined as:
   $$\text{contribution}_s = w_s \times r_s$$
3. We explicitly compute and report the **Residual (Unmodeled / Weighting Drift)**:
   $$\text{residual} = r_{\text{index}} - \sum_s \text{contribution}_s$$
   This guarantees exact mathematical reconciliation:
   $$\sum_s \text{contribution}_s + \text{residual} \equiv r_{\text{index}}$$
4. The UI prominently displays the "Approximate" label, cites the factsheet source and `as_of` date, and includes the residual as an explicit category in waterfalls, tables, and audit drawers.

## Consequences

- Zero reliance on individual equity tickers or proprietary real-time constituent data.
- Transparency: users immediately see both the explained sector contribution and the residual.
- Mathematically rigorous: no hidden gap between the benchmark move and its attribution.
