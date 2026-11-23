# 5. Insight Engine and Divergence Detector Strategy

Date: 2026-11-21
Status: Accepted

## Context
Milestone M5 establishes the deterministic rules engine and the pre-registered divergence detector delivering the `/radar` change feed without any runtime LLM market forecasts or advisory language.

## Decisions
1. **Pre-Registered Pairs Protocol**:
   - Divergence detection is strictly evaluated on the 10 pre-registered pairs in `pipeline/config/pairs.yaml`, each with documented economic rationale.
   - Expanding z-scores are computed on 20-day momentums: $D_t = Z(\text{mom}_{20, A}) - Z(\text{mom}_{20, B})$.
   - Persistent trigger condition: $|D_t| \ge 95\text{th}$ percentile for $\ge 3$ consecutive business days.
   - The UI explicitly discloses the false discovery caveat: "1 of 10 pre-registered pairs tested" and clarifies that divergence is a statistical disagreement, not a trading signal.

2. **Alert Discipline & Budgets**:
   - Cooldown period: 10 business days per asset per rule to suppress alert noise.
   - Daily cap: Maximum 8 observations emitted per day.
   - Severity bands: Low ($\ge 90$th percentile), Medium ($\ge 95$th), High ($\ge 99$th).

3. **Deterministic Templating & Safety**:
   - Observation copy is generated exclusively through verified templates in `pipeline/config/templates.yaml`.
   - Automated tests scan all templates and generated strings for forbidden financial advice terms (`buy`, `sell`, `bullish`, `bearish`, `target`, `will rise`, `will fall`, `recommend`, `signal to`).
