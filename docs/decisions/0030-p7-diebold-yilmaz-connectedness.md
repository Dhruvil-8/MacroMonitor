# 0030. P7 Diebold-Yilmaz GFEVD Connectedness Framework

Date: 2026-11-22
Status: Accepted

## Context
Module P7 of `PUSH_PACK_PLAN.md` introduces an empirical cross-border spillover and connectedness monitoring system across global equity and sovereign markets.

## Decisions Made
1. **Order-Invariant Pesaran-Shin GFEVD**:
   - Standard Cholesky variance decompositions depend arbitrarily on the chosen variable ordering.
   - We implement the generalized forecast error variance decomposition (Pesaran & Shin 1998, Diebold & Yilmaz 2012), which is mathematically order-invariant.
2. **Weekly Frequency Discipline**:
   - Uses Friday weekly USD-denominated log returns, eliminating non-synchronous time-zone closing discrepancies between Asian, European, and American exchanges.
3. **Horizon and Normalization**:
   - Horizon: $H = 10$ weeks.
   - Row-normalization forces the sum of shocks explaining each country's variance to equal 100%.
4. **Metrics Formulation**:
   - Total Connectedness Index (TCI): off-diagonal share of system variance.
   - Directional "To" (spillover transmitted) and "From" (spillover received).
   - Net Spillover: $\text{To} - \text{From}$ distinguishes net transmitters (e.g. US) from net receivers.
5. **UI `/connectedness`**:
   - `ConnectednessMatrix.tsx` renders directed matrix heatmap, diverging net transmitter bars, and rolling TCI history.

## Verification
- Unit tests in `pipeline/tests/unit/test_connectedness.py` confirm VAR(1) recovery, row normalization to 100%, and order invariance under variable permutation.
