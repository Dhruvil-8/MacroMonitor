# Specification: Insight Engine and Divergence Detector (s06-insight-engine-divergence)

## 1. Principle
Descriptive statistics $\rightarrow$ deterministic rules $\rightarrow$ **Observation** objects $\rightarrow$ templated text.
Zero LLM generation at runtime. No market advice, no directional recommendations, no forecasting language.

## 2. Divergence Detector (Section 8.5)
- Evaluated exclusively for pre-registered pairs in `pipeline/config/pairs.yaml`.
- For assets $A$ and $B$, compute 20-day rate of change:
  $$\text{mom}_{20, A} = \frac{P_{t, A}}{P_{t-20, A}} - 1, \quad \text{mom}_{20, B} = \frac{P_{t, B}}{P_{t-20, B}} - 1$$
- Compute expanding z-scores using prior history up to $t-1$:
  $$Z_A = Z(\text{mom}_{20, A}), \quad Z_B = Z(\text{mom}_{20, B})$$
- Divergence spread:
  $$D_t = Z_A - Z_B$$
- Trigger condition:
  $$|D_t| \ge \text{Percentile}_{95}(|D|) \quad \text{for } \ge 3 \text{ consecutive business days}$$
- Caveat: Divergence indicates a statistical relationship breakdown, not a trade recommendation. The number of tested pairs ($K=10$) is displayed to account for multiple hypothesis testing.

## 3. Observation Rules & Alert Budget (Section 9.3 & 9.4)
- **Rules**:
  1. `volatility_shift`: 20-day realized volatility $\ge 90$th historical percentile.
  2. `unusual_return`: Absolute single-day return $> 3.0$ expanding standard deviations.
  3. `divergence`: Divergence $|D_t| \ge 95$th percentile for 3 consecutive days.
  4. `correlation_breakdown`: $|r_{60} - r_{250}| \ge 2.0\sigma$.
- **Severity Bands**:
  - `low`: Percentile $\in [90, 95)$
  - `medium`: Percentile $\in [95, 99)$
  - `high`: Percentile $\ge 99$ (or $|z| \ge 3.5$)
- **Discipline & Budgets**:
  - Cooldown: 10 business days per asset per rule.
  - Daily cap: maximum 8 items displayed, remaining collapsed.
  - Monthly cap: maximum 3 alerts per asset per month.

## 4. Forbidden Terms Policy (Section 9.5)
Any template or generated text containing words like `buy`, `sell`, `bullish`, `bearish`, `target`, `will rise`, `will fall`, `recommend`, or `signal to` is rejected by automated test scan.
