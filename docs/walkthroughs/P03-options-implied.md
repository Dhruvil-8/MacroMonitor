# Walkthrough: P3 Options-Implied Layer

## Overview
Module P3 extracts market implied expectations and tail risk from publicly published volatility indices and ratios without requiring proprietary tick-level options databases:
1. **VIX Term Structure:** Cboe term structure across 9-day (VIX9D), 30-day (VIX), 3-month (VIX3M), 6-month (VIX6M), and 1-year tenors.
2. **Curve Slope & Regimes:** Slope metric $\frac{\text{VIX3M}}{\text{VIX}} - 1$ classified into Contango, Flat, or Backwardation regimes with dead-bands and hysteresis.
3. **Tail Risk & Vol-of-Vol:** Cboe SKEW index tracking out-of-the-money put skew demand, and VVIX tracking the volatility of volatility.
4. **Variance Risk Premium (VRP) Proxy:** Trailing proxy $\text{VIX}^2 - \text{RV}_{21}^2$ comparing implied volatility against trailing 21-day realized volatility.
5. **Implied Volatility UI (`/volatility`):** Interactive term structure curve, slope history with backwardation shading, and fear/tail gauge panels.

## Key Files
- `pipeline/src/radar/metrics/volatility.py`: Options-implied metrics and term-structure engine.
- `web/src/app/volatility/page.tsx` & `web/src/components/charts/TermStructureChart.tsx`: UI analytics.
- `pipeline/tests/unit/test_implied_vol.py`: Unit tests for slope classification and VRP calculations.
- `docs/decisions/0026-p3-options-implied-layer.md`: Architecture Decision Record.
