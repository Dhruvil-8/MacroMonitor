# 0026. P3 Options-Implied Volatility and Sentiment Layer

Date: 2026-11-22
Status: Accepted

## Context
Module P3 of `PUSH_PACK_PLAN.md` deepens the market risk and derivatives layer using public Cboe volatility indices (30D VIX, 9D VIX, 3M VIX, 6M VIX, VVIX, SKEW) and India VIX without requiring expensive raw options tick feeds.

## Decisions Made
1. **Term Structure Slope Formulation**:
   - 3-Month Slope: $VIX_{3M} / VIX - 1$
   - 9-Day Slope: $VIX / VIX_{9D} - 1$
   - State classification uses a $\pm 2\%$ deadband:
     - `contango` if slope $> +0.02$
     - `backwardation` if slope $< -0.02$
     - `flat` otherwise.
2. **Expanding Percentiles**: VIX, VVIX, SKEW, and slopes are ranked strictly against expanding historical observations (no look-ahead bias).
3. **Variance Risk Premium (VRP) Proxy**:
   - Historical live display: $VRP = VIX^2 - \sigma_{21,\text{trailing}}^2$
   - Explicitly labeled as "Trailing proxy" to clearly distinguish from unobservable forward-looking expectation.
4. **Implied vs. Realized Volatility**:
   - S&P 500 realized 21-day volatility vs VIX (US).
   - Nifty 50 realized 21-day volatility vs India VIX (India).
5. **UI `/volatility`**:
   - `TermStructureChart.tsx` plots current curve against 1-month ago, 1-year median, and 10th-90th percentile envelope.
   - Interactive toggle to view the historical slope and backwardation regimes.

## Verification
- Unit tests in `pipeline/tests/unit/test_implied_vol.py` verify term-structure curve points, slope states, and VRP calculation.
