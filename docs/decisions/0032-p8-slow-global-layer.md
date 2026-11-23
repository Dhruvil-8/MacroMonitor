# 0032. P8 Slow Global Layer (BIS REER, TIC Foreign Holdings, Central Banks)

Date: 2026-11-22
Status: Accepted

## Context
Module P8 of `PUSH_PACK_PLAN.md` introduces low-frequency, structural macroeconomic and reserve context explaining long-term capital flows and currency regimes.

## Decisions Made
1. **BIS Real Effective Exchange Rates (REER)**:
   - Tracks valuation percentiles for 6 major currencies (USD, INR, EUR, JPY, GBP, CNY).
   - Classified as `overvalued` ($>75\text{th}$ pct), `undervalued` ($<25\text{th}$ pct), or `neutral`.
2. **US Treasury Foreign Custody (TIC)**:
   - Tracks top foreign sovereign and institutional holders (Japan, China, UK, Luxembourg, Canada, India) with total holdings, shares, and 12-month net changes.
3. **Global Central Bank Balance Sheets**:
   - Total assets in USD trillions for Fed, ECB, and BOJ to monitor global quantitative tightening (QT) / easing trajectory.
4. **India Macro Reserve Adequacy**:
   - Foreign exchange reserves ($B) and import cover (months) alongside 12-month rolling FPI equity vs debt flows.
5. **Strict Lag Disclosure**:
   - Every metric explicitly notes publication lags (4 to 8 weeks).

## Verification
- Unit tests in `pipeline/tests/unit/test_global_slow.py` confirm REER expanding percentiles, valuation classifications, and payload structure.
