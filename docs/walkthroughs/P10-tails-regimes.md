# Walkthrough: P10 Advanced Tails and Regimes

## Overview
Module P10 equips the Global Market Radar Risk Lab with non-linear, non-Gaussian statistical risk tools:
1. **Extreme Value Theory (EVT) Generalized Pareto Distribution (GPD):**
   - Implements the Peaks-Over-Threshold (POT) approach to parameterize heavy loss tails beyond the 95th percentile.
   - Computes 99.0% and 99.5% tail VaR and Expected Shortfall compared against Gaussian and empirical benchmarks.
   - Diagnoses Pareto tail linearity via the sample Mean-Excess function $e(u) = E[X - u \mid X > u]$.
2. **2-Regime Markov-Switching Volatility (Hamilton 1989):**
   - Models market dynamics as transitioning between Regime 0 (Calm / Low Volatility) and Regime 1 (Stress / High Volatility).
   - Point-in-time filtered probabilities $P(S_t = 1 \mid Y_{1..t})$ prevent look-ahead bias for real-time risk gauges.
   - Smoothed probabilities $P(S_t = 1 \mid Y_{1..T})$ provide optimal historical shading.
   - Computes expected regime durations and transition probability matrices.

## Key Files
- `pipeline/src/radar/metrics/tails_regimes.py`: EVT POT and Markov switching estimation engine.
- `web/src/components/charts/EvtRegimePanel.tsx`: Interactive dual-tab UI with tail comparison tables, mean-excess cards, regime ribbons, and transition matrix grid.
- `pipeline/tests/unit/test_tails_regimes.py`: Unit test suite verifying parameter recovery, anti-lookahead guarantees, and schema validation.
- `docs/decisions/0033-p10-evt-and-markov-regimes.md`: Architecture Decision Record.
- Learning notes: `docs/learn/evt-gpd.md` and `docs/learn/markov-switching.md`.

## Verification
- Pytest suite: 5/5 passing (100%).
- Data contract: `data/v1/tails_regimes.json`.
- Integrated directly into `/risk`.
