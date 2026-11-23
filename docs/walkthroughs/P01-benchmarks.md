# Walkthrough: P1 Benchmarks and Episodes

## Overview
Module P1 introduces independent external benchmarks and historical crisis episode context to validate internal stress and volatility metrics:
1. **External Stress Benchmarks:** Cross-references our internal Financial Stress Composite against the Chicago Fed National Financial Conditions Index (NFCI) and the St. Louis Fed Financial Stress Index (FSI).
2. **Agreement Statistics:** Calculates Pearson correlation, Spearman rank correlation, and lead-lag cross-correlations with effective sample size ($N_{\text{eff}}$) degrees-of-freedom adjustment.
3. **Episode Registry (`pipeline/config/episodes.yaml`):** Curates major global historical market episodes (2008 GFC, 2013 Taper Tantrum, 2015 China Devaluation, 2018 Q4 Selloff, 2020 Covid Shock, 2022 Global Rate Shock).
4. **Stress Cross-Check UI (`/risk`):** Interactive comparison panel on the Risk Lab with normalized standardized stress lines and external reference agreement metrics.

## Key Files
- `pipeline/src/radar/metrics/benchmarks.py`: Benchmark cross-correlation and episode alignment engine.
- `pipeline/config/episodes.yaml`: Historical crisis episode registry.
- `web/src/components/charts/StressCrossCheckPanel.tsx`: Stress cross-check comparison chart.
- `pipeline/tests/unit/test_benchmarks.py`: Unit tests for lead-lag cross-correlation and episode alignment.
- `docs/decisions/0024-p1-benchmarks-and-episodes.md`: Architecture Decision Record.
