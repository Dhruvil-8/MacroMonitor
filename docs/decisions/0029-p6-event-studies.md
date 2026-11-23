# 0029. P6 Event Studies on Central Bank Decision Days

Date: 2026-11-22
Status: Accepted

## Context
Module P6 of `PUSH_PACK_PLAN.md` introduces rigorous empirical event studies analyzing financial market behavior around central bank policy decisions (FOMC, RBI MPC).

## Decisions Made
1. **Pre-Event Estimation Window Discipline**:
   - Market model: $R_{i,t} = \alpha_i + \beta_i R_{m,t} + \epsilon_{i,t}$
   - Estimated strictly over a 250-trading-day window ending 30 trading days *prior* to the announcement.
   - The event window $[-5, +10]$ is strictly excluded from beta estimation to eliminate look-ahead contamination.
2. **Abnormal Return (AR) & Cumulative Abnormal Return (CAR)**:
   - $AR_{i,t} = R_{i,t} - (\hat{\alpha}_i + \hat{\beta}_i R_{m,t})$
   - $CAR_{i}[-5, \tau] = \sum_{t=-5}^{\tau} AR_{i,t}$
3. **Block Bootstrap Uncertainty Bands**:
   - Generates 50% and 90% confidence bands across event paths via block bootstrap resampling (500 iterations).
4. **Decision Conditioning**:
   - Groups outcomes into discrete policy categories ('hike', 'hold', 'cut').
   - Calculates event-day move magnitude ratio relative to unconditional trading days.
5. **UI `/events`**:
   - `ReactionRibbon.tsx` presents the multi-day CAR path with confidence envelopes, decision breakdown cards, and a detailed decision log.

## Verification
- Unit tests in `pipeline/tests/unit/test_events.py` verify parameter recovery, CAR calculation, and bootstrap ribbon generation.
