# ADR 0007: Liquidity Module Unit Alignment and Decomposition

## Status
Accepted

## Context
US Net Liquidity is computed as:
$$\text{Net Liquidity} = \text{Fed Assets} - \text{TGA} - \text{RRP}$$

However, the underlying data sources have divergent native units and reporting frequencies:
1. `WALCL` (Fed Total Assets) is reported weekly on Wednesdays in **Millions of USD**.
2. `WTREGEN` (Treasury General Account) is reported weekly on Wednesdays in **Millions of USD**.
3. `RRPONTSYD` (Overnight Reverse Repo) is reported daily in **Billions of USD**.
4. The S&P 500 equity index overlay is reported daily as a standard index point price.

Without strict unit alignment and frequency normalization, raw summation would produce errors of three orders of magnitude ($10^3$).

## Decisions
1. **Unit Normalization to Billions USD**:
   All balance sheet components are converted to Billions of USD (`usd_bn`):
   $$A_t^{\text{bn}} = \frac{\text{WALCL}_t}{1000}$$
   $$T_t^{\text{bn}} = \frac{\text{WTREGEN}_t}{1000}$$
   $$R_t^{\text{bn}} = \text{RRPONTSYD}_t$$
   $$\text{Net Liquidity}_t = A_t^{\text{bn}} - T_t^{\text{bn}} - R_t^{\text{bn}}$$

2. **Frequency Alignment**:
   Weekly Wednesday data is forward-filled onto the continuous business day calendar with `max_ffill_days = 7` to bridge weekend/holiday publication lags without introducing forward-looking bias.

3. **Hand-Checked Arithmetic Verification**:
   Three historical test points (2024-01-03, 2025-01-08, 2026-11-18) are hardcoded into automated regression tests to guarantee arithmetic fidelity.

4. **Equity Overlay and Correlation Capping**:
   Rolling 60-day Pearson correlation against the S&P 500 is clamped to $[-1.0, 1.0]$ to guard against floating-point inaccuracies, and explicitly accompanied by the disclaimer: *"Association, not mechanical causation"*.

## Consequences
- Data is consistent across charts, tooltips, and audit drawers.
- Unit alignment is documented transparently in the Audit Drawer.
- Zero floating-point boundary overflow on rolling correlation.
