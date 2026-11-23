# ADR 0018: Cross-Asset Radar, Section 4.6 Observation Rules, and CUSUM Change Detection

## Context
Section 4.6 and Section 5.1 (Chart C24: CUSUM / Change chart) of `docs/MASTER_PLAN.md` mandate a publication-grade Cross-Asset Radar and Change Feed (`/radar`). The engine must evaluate deterministic statistical rules across all macro layers, detect structural regime shifts in pre-registered asset relationships, and enforce strict institutional volume caps, cooldown suppression, and linguistic neutrality.

## Decision
1. **Two-Sided Cumulative Sum (CUSUM) Change Detection (Chart C24)**:
   For every pre-registered pair's momentum divergence spread $D_t = Z(\text{mom}_{20, A}) - Z(\text{mom}_{20, B})$, we compute the standardized innovations $z_t = \frac{D_t - \bar{D}}{\hat{\sigma}_D}$ and evaluate the two-sided CUSUM recursion:
   $$S_t^+ = \max(0, S_{t-1}^+ + z_t - k)$$
   $$S_t^- = \max(0, S_{t-1}^- - z_t - k)$$
   where slack allowance $k = 0.5$ and decision threshold $h = 4.0$. Upon breaching the decision threshold ($S_t^+ \ge h$ or $S_t^- \ge h$), a structural regime change event is logged and the corresponding accumulator resets to zero. This delivers rapid (3-5 session) detection of persistent macro decouplings while maintaining near-zero false alarm rates under Gaussian noise.

2. **Full Section 4.6 Observation Rules Expansion**:
   We implemented deterministic candidate generation across all 10 macro categories defined in Section 4.6:
   - `country_divergence_v1`: 20-day momentum divergence spread $|Z| \ge 2.0\sigma$ persisting for $\ge 3$ consecutive sessions.
   - `fx_driven_move_v1`: Currency share of USD equity return exceeding $50\%$ when $|r_{\text{usd}}| \ge 0.5\%$.
   - `sector_leadership_v1`: Sector taking the #1 momentum rank with $r_{20d} \ge 2.0\%$.
   - `driver_shift_v1`: Rolling factor regression coefficient exiting its historical 2-standard-error band ($p \le 0.05$).
   - `commodity_shock_v1`: Unusual 1-day commodity move ($|z| \ge 2.0\sigma$ or $|r| \ge 2.5\%$) highlighting historically sensitive economies.
   - `dollar_regime_v1`: US Dollar Index (DXY) 20-day momentum move reaching extreme percentiles ($|z| \ge 1.8\sigma$).
   - `breadth_deterioration_v1`: Country-level equity breadth falling below $50\%$ above their 200-day moving average.
   - `pc1_sync_spike_v1`: First principal component (PC1) variance share rising above $45\%$ ("markets moving together unusually").
   - `downside_corr_jump_v1`: Downside correlation exceeding unconditional correlation by $\ge 0.15$ (Stress Correlation Premium).
   - `vol_cone_breach_v1`: Realized volatility exceeding the 90th historical percentile across 2 or more horizons simultaneously.

3. **Batch Prioritization and Volume Discipline**:
   Candidate observations from all layers are batched and ranked by severity (`high` > `medium` > `low`) before ingestion by `InsightEngine`. This guarantees that critical tail-risk and volatility alerts always take precedence within the strict daily cap of 8 emitted alerts, with 10-day per-asset cooldown suppression preventing alert fatigue.

4. **Linguistic Neutrality and Safety Compliance**:
   All headlines and descriptions are populated deterministically from `pipeline/config/templates.yaml`. Automated unit tests (`test_templates_safety.py`) verify the complete absence of forbidden trading jargon (`buy, sell, bullish, bearish, target, will rise, will fall, recommend, signal to, outperform, underperform`).

## Consequences
- The `/radar` page presents an institutional-grade change feed with category filtering (All, Vol & Risk, Cross-Market, Macro) and severity controls.
- Each pre-registered divergence panel features an interactive toggle between the Z-score spread chart and the two-sided CUSUM control chart (C24), allowing analysts to verify whether a divergence represents transient noise or a structural regime break.
