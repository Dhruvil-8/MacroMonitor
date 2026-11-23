# Track Record Pre-Registration Specification

**Governed by:** PUSH_PACK_PLAN.md Section 16 & MASTER_PLAN.md Section 11  
**Pre-Registration Date:** 2026-11-22  
**Evaluation Standard:** Out-of-Sample Point-in-Time Forward Verification  

## 1. Objective and Anti-Spurious Guarantee
To prevent post-hoc narrative rationalization and p-hacking, every observation rule produced by Global Market Radar is tied to a strictly pre-registered mathematical hypothesis. We evaluate whether an observation rule provided statistically significant predictive divergence relative to the unconditioned historical **base rate** on random days.

Rules that show no improvement over the base rate are labeled **"Context Only"** and published openly.

## 2. Pre-Registered Hypotheses by Rule Family

| Rule ID | Family | Pre-Registered Hypothesis | Evaluation Horizon | Success Condition (Hit) |
|---|---|---|---|---|
| `VOL_PERSISTENCE` | Volatility Shift | After an abnormal volatility spike, realized volatility remains elevated | 20 trading days | $\sigma_{t, t+20} > \text{Median}(\sigma_{t-60, t})$ |
| `DRAWDOWN_ACCELERATION` | Downside Shock | Severe drawdown alerts precede further downside acceleration | 20 trading days | $\min_{\tau \in [t, t+20]} (P_\tau / P_t - 1) < -3.0\%$ |
| `TREND_EXHAUSTION` | Regime Exhaustion | Extended 3σ price extensions indicate mean-reverting deceleration | 20 trading days | $|R_{t, t+20}| < |R_{t-20, t}| \times 0.70$ |
| `CROSS_ASSET_DIVERGENCE` | Spread Divergence | Extreme historical spread divergences narrow back towards mean | 20 trading days | $|\Delta z_{t+20}| < |\Delta z_t| \times 0.80$ |
| `LIQUIDITY_CONTRACTION` | Liquidity Shock | US Net Liquidity contractions accompany equity multiple compression | 20 trading days | $R^{\text{SPX}}_{t, t+20} < 0$ |
| `COT_EXTREME_CROWDING` | Positioning | Speculative positioning above 95th percentile precedes consolidation | 20 trading days | $R_{t, t+20} \le +1.0\%$ |

## 3. Declustering Protocol
Because market alerts cluster tightly during stress periods (e.g. 5 alerts across 4 days during an FOMC rate hike), treating every alert as independent falsely inflates statistical significance.
- **Rule:** Consecutive alerts for the same rule within **10 trading days** are grouped into a single **independent episode**.
- The episode date is defined as the initial trigger date $t_{\text{init}}$.

## 4. Base Rate & Block-Bootstrap Inference
For each rule:
1. **Empirical Hit Rate ($H$):**
   $$H = \frac{N_{\text{hits}}}{N_{\text{episodes}}}$$
2. **Unconditional Base Rate ($B$):**
   The proportion of all non-alerted dates $t \in T$ that satisfy the exact same success condition.
3. **Statistical Edge:**
   $$\Delta = H - B$$
4. **Confidence Intervals:**
   We compute 95% stationary block-bootstrap confidence intervals ($K=1,000$ iterations, block length $b=20$ days) over the difference $\Delta$.
5. **Classification Status:**
   - **Validated ($p < 0.05$):** Lower bound of 95% CI $> 0$.
   - **Marginal:** $\Delta > 0$ but 95% CI spans zero.
   - **Context Only:** $\Delta \le 0$ or no discernible statistical edge over random guessing.

## 5. Point-in-Time Discipline
No outcome is computed until the full 20-day forward evaluation horizon has elapsed. Alerts triggered within the last 20 trading days are classified as "Active - Horizon Pending" and excluded from performance ratios.
