# ADR 0009: Financial Stress Composite and Automated Daily Briefing

## Status
Accepted

## Context
Market risk cannot be encapsulated by single-asset volatility indices (such as equity VIX) alone. During severe market shocks, dislocations occur across rates (yield volatility), foreign exchange (carry unwind), commodity markets, and cross-asset correlations. Furthermore, users require an automated, factual daily synthesis that extracts key statistical observations without human editorializing, subjective market predictions, or regulatory exposure to unregistered financial advice.

## Decisions
1. **Financial Stress Composite with Strictly Zero Imputation**:
   The composite score is defined as the arithmetic mean of available component percentile ranks:
   $$S_t = \frac{1}{|K_t|} \sum_{k \in K_t} c_{k, t}$$
   - Any missing component (such as unobserved credit spreads) is completely dropped from the calculation, reducing the denominator $|K_t|$ without synthesizing false historical numbers.
   - Missing components are explicitly enumerated in metadata and displayed in the UI.

2. **Historical Crisis Episode Alignment**:
   Historical shocks (2008 GFC, March 2020 COVID shock, August 2024 volatility jump) serve as hard mathematical regression benchmarks ($S_t \ge 85$ for global liquidity panics, $S_t \ge 65$ for acute carry unwinds).

3. **Automated Rules-Based Daily Briefing**:
   The daily briefing is generated purely from verified metrics, rule firings, and institutional flow deltas. An automated regex scan enforces the total absence of subjective recommendations or directional forecasts (`buy`, `sell`, `target`, `will rally`, `will crash`).

## Consequences
- Transparent multi-asset risk measurement that doesn't hide behind missing data imputation.
- Automated daily briefings with complete regulatory safety.
- Traceability guaranteed on every metric via the Audit Drawer.
