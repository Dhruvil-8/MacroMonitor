# 0025. P2 CFTC Commitments of Traders (COT) Positioning & Crowding Ladder

Date: 2026-11-22
Status: Accepted

## Context
Module P2 of `PUSH_PACK_PLAN.md` deepens the macro flows & derivatives layer using CFTC public legacy and disaggregated futures reports (via publicreporting.cftc.gov API). 

## Decisions Made
1. **Contract Universe Expansion**: Registered 16 liquid futures contracts across Equities, Rates, Currencies, and Commodities in `pipeline/config/cot_contracts.yaml`.
2. **Expanding Historical Horizon**: Percentiles and Z-scores of net speculative position as % of open interest are calculated using an expanding lookback window of 156 weeks (minimum 52 weeks to publish). No future observations are included.
3. **Multi-Horizon Percentile Strips**: Included 4-week (`percentile_4w_ago`) and 13-week (`percentile_13w_ago`) percentiles and changes in `net_pct_oi` to visualize positioning velocity.
4. **2-Week Hysteresis Rule**: Extreme crowding flags (`is_extreme`, `extreme_type`) require the net speculative percentile to remain strictly above 95% or below 5% for at least two consecutive weekly reporting dates, filtering single-week reporting anomalies.
5. **UI Positioning Ladder**: Built `PositioningLadder.tsx` with horizontal percentile tracks, asset-class filters (Equities, Rates, Currencies, Commodities), and clear lag disclosures ("Positions as of Tuesday, published Friday").

## Verification
- Unit tests in `pipeline/tests/unit/test_positioning_pack.py` confirm 100% adherence to expanding percentiles and 2-week hysteresis.
