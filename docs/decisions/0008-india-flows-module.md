# ADR 0008: India Flows and Derivatives Positioning Module

## Status
Accepted

## Context
Indian financial markets possess distinctive microstructure characteristics driven by two primary forces:
1. **The FII vs DII Cash Market Absorption Dynamic**: Foreign Institutional Investors (FIIs) trade synchronously with global macro dollar and liquidity cycles, whereas Domestic Institutional Investors (DIIs) receive structural domestic retail SIP (Systematic Investment Plan) inflows, providing strong countercyclical absorption.
2. **Derivatives Positioning & Delivery Quality**: The National Stock Exchange of India (NSE) is the world's largest derivative exchange by volume. Understanding true spot capital commitment requires distinguishing leveraged intraday speculative volume from physical spot delivery percentage, and classifying open interest expansion by underlying price action into the four classical buildup quadrants.

## Decisions
1. **Four-Quadrant F&O Buildup Classification**:
   For any contract with daily price change $\Delta P$ and open interest change $\Delta OI$:
   - $\Delta P > 0, \Delta OI > 0 \implies$ **Long Buildup** (Bullish accumulation)
   - $\Delta P < 0, \Delta OI > 0 \implies$ **Short Buildup** (Bearish accumulation)
   - $\Delta P > 0, \Delta OI < 0 \implies$ **Short Covering** (Bearish exit/bounce)
   - $\Delta P < 0, \Delta OI < 0 \implies$ **Long Unwinding** (Bullish exit/stop loss)

2. **Multi-Horizon Rolling Sums & Percentiles for Cash Flows**:
   Both daily and rolling 5-day and 20-day sums are computed for FII and DII net cash purchases, with expanding historical percentiles to identify institutional exhaustion or extreme positioning without lookahead bias.

3. **Put-Call Ratio (PCR)**:
   Calculated as Total Put OI / Total Call OI with standard regime thresholds ($< 0.7$ oversold, $> 1.3$ overbought).

4. **Traceability**:
   Every figure (India VIX, FII flows, PCR, delivery percentage) links directly to the Traceability Audit Drawer documenting exact formula, sources (NSE/SEBI), $N$, and effective sample size $N_{\text{eff}}$.

## Consequences
- Comprehensive institutional intelligence for the Indian market.
- Clean contract validation via Pydantic and Zod schemas.
- Full traceability on `/india` page.
