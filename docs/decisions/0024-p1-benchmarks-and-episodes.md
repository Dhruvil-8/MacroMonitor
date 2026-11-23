# Decision 0024: P1 Benchmarks and Historical Episodes Layer

## Status
Accepted

## Context
As required by `docs/PUSH_PACK_PLAN.md` Section 4 (P1), our synthetic multi-asset financial stress composite must be validated against external institutional references (Chicago Fed NFCI and St. Louis Fed Financial Stress Index) and benchmarked against historical crisis episodes to ensure credibility without look-ahead bias.

## Decision
1. **Curated Episodes**: Authored `pipeline/config/episodes.yaml` cataloging major historical macro shocks (2008 GFC, 2011 US Debt Downgrade, 2013 Taper Tantrum, 2015 China Devaluation, 2018 Q4 Tightening, 2020 COVID Shock, 2022 Global Rate Shock, 2023 SVB Banking Turmoil).
2. **Benchmark Agreement Engine**: Built `pipeline/src/radar/metrics/benchmarks.py` calculating:
   - Pearson $r$ and Spearman rank $\rho$ correlation.
   - Lead-lag cross-correlation spectrum across $\pm 4$ week windows using index-preserving shifts.
   - Effective sample size $N_{\text{eff}}$ adjusted for serial autocorrelation $\rho_1$.
   - Episode alignment statistics (peak stress date, lead/lag days to official peak, elevated ratio).
3. **UI Integration**:
   - Built `StressCrossCheckPanel.tsx` mounted to the Institutional Risk Lab (`/risk`).
   - Added economic policy uncertainty context strip.
4. **Testing**: Implemented `pipeline/tests/unit/test_benchmarks.py` passing 100%.
