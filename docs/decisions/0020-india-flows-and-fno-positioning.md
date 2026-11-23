# ADR 0020: India Flows & F&O Derivative Positioning

## Context
Section 8.9 and Section 1.3 ("India-first wedge") of `docs/MASTER_PLAN.md` mandate dedicated domestic positioning analytics for the Indian equity market (`/india`). The module must track institutional cash market flows (FII and DII), derivative open interest buildup regimes, index Put-Call Ratio (PCR), delivery percentages, and India VIX percentile ribbons, with full mathematical lineage and zero advisory claims.

## Decision
1. **Four-Quadrant F&O Open Interest Buildup Classification**:
   For any derivative instrument (index or single stock), we classify daily price change ($\Delta P$) and open interest change ($\Delta \text{OI}$) into classical market microstructure regimes:
   - **Long Buildup**: $\Delta P > 0, \Delta \text{OI} > 0$ (fresh long contracts initiated; buyers aggressive).
   - **Short Buildup**: $\Delta P < 0, \Delta \text{OI} > 0$ (fresh short contracts initiated; sellers aggressive).
   - **Short Covering**: $\Delta P > 0, \Delta \text{OI} < 0$ (short sellers buying back to close positions).
   - **Long Unwinding**: $\Delta P < 0, \Delta \text{OI} < 0$ (long holders selling to liquidate positions).

2. **FII & DII Rolling Cumulative Cash Flows**:
   Daily net cash purchases by Foreign Institutional Investors ($F_t^{\text{FII}}$) and Domestic Institutional Investors ($F_t^{\text{DII}}$) in INR Crores are aggregated across multiple rolling horizons:
   $$S_{5D} = \sum_{k=0}^4 F_{t-k}, \quad S_{20D} = \sum_{k=0}^{19} F_{t-k}$$
   The 20-day cumulative net flow is ranked against its historical expanding distribution to yield the percentile rank $P_{20D} \in [0, 100]$, contextualizing institutional accumulation or distribution intensity.

3. **Option Put-Call Ratio (PCR)**:
   We evaluate the index-level total open interest ratio:
   $$\text{PCR} = \frac{\sum \text{Put Open Interest}}{\sum \text{Call Open Interest}}$$
   A ratio above $1.0$ indicates put-heavy positioning (often reflecting institutional downside hedging demand), while a ratio below $0.8$ reflects call dominance.

4. **Volume-Weighted Delivery Percentage**:
   To distinguish between intraday speculative churn and genuine delivery-backed institutional capital allocation, we compute:
   $$\text{Delivery \%} = \frac{\sum \text{Delivered Quantity}}{\sum \text{Traded Quantity}} \times 100$$
   aggregated across all Nifty 50 constituents.

5. **India VIX Historical Percentile Ribbon**:
   We compute the empirical expanding historical percentile of India VIX:
   $$\text{Percentile}_t = \frac{1}{N} \sum_{s=1}^N \mathbf{1}_{\{V_s \le V_t\}} \times 100$$
   exposing implied volatility extremes without making speculative market direction claims.

## Consequences
- The `/india` dashboard provides institutional clarity on Indian equity positioning without relying on opaque retail indicators.
- Hand-checked fixtures in `test_india_flows.py` verify that regimes align with known historical volatility episodes (e.g. June 4, 2024 election shock, November 2024 consolidation pullback).
