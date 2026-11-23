# Walkthrough: P2 Futures Positioning (CFTC COT)

## Overview
Module P2 provides institutional futures positioning and crowding analytics without look-ahead bias across 16 major contracts:
1. **Contract Coverage (`pipeline/config/cot_contracts.yaml`):** 16 contracts spanning Equities (E-mini S&P 500, Nasdaq 100), Rates (2Y, 10Y, 30Y Treasuries), Currencies (EUR, JPY, GBP, AUD, CAD, CHF), and Commodities (WTI Crude, Natural Gas, Gold, Copper, Silver).
2. **Point-in-Time Discipline:** Net speculative positioning as % of open interest:
   $$\text{NetPctOI}_t = \frac{\text{NonCommLong}_t - \text{NonCommShort}_t}{\text{OI}_t}$$
   Expanding 156-week percentiles and Z-scores strictly conditioning on prior weekly reports.
3. **Crowding & Hysteresis Buffer:** 2-week consecutive extreme requirement ($>95$th or $<5$th percentile) preventing false-alarm flip-flopping.
4. **Positioning UI (`/positioning`):**
   - **Positioning Ladder:** Cross-asset horizontal percentile ladder with Now, -4W, and -13W dots.
   - **Contract Detail Chart:** Historical Net % of Open Interest bars with expanding percentile line.
   - **Lag Label:** Clear reporting timestamps ("Positions as of Tuesday, published Friday").

## Key Files
- `pipeline/src/radar/ingest/cftc.py`: CFTC Socrata public API ingestion.
- `pipeline/src/radar/metrics/positioning.py`: COT metrics and crowding detection engine.
- `pipeline/config/cot_contracts.yaml`: 16-contract registry with CFTC codes and categories.
- `web/src/app/positioning/page.tsx`, `PositioningLadder.tsx`, `PositioningChart.tsx`: Frontend interfaces.
- `pipeline/tests/unit/test_positioning.py` & `test_positioning_pack.py`: Unit tests.
- `docs/decisions/0025-p2-cftc-positioning.md`: Architecture Decision Record.
