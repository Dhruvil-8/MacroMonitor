# Walkthrough: P6 Event Studies (Central Bank Decisions)

## Overview
Module P6 measures asset market reactions around policy decisions from major central banks without requiring expensive intraday surprise feeds:
1. **Decision Calendars (`pipeline/config/events.yaml`):** Curates announcement dates, times, and post-hoc classifications (hike, hold, cut) for the Federal Reserve (FOMC) and Reserve Bank of India (RBI MPC).
2. **Market-Model Abnormal Returns:** Estimates pre-event market beta over window $[-60, -10]$ and computes daily abnormal returns:
   $$AR_{i,t} = R_{i,t} - \hat{\beta}_i R_{m,t}$$
3. **Cumulative Abnormal Return (CAR) Ribbon:** Traces cumulative paths over $[-5, +20]$ days with stationary block-bootstrap 50% and 90% confidence bands.
4. **Event Studies UI (`/events`):** Central bank switcher, CAR reaction ribbon, magnitude comparison bars, and historical decisions ledger.

## Key Files
- `pipeline/src/radar/metrics/events.py`: Event study market model and block-bootstrap CAR engine.
- `pipeline/config/events.yaml`: Policy decision calendars.
- `web/src/app/events/page.tsx` & `web/src/components/charts/ReactionRibbon.tsx`: Frontend interfaces.
- `pipeline/tests/unit/test_events.py`: Unit tests verifying abnormal return recovery and time-alignment.
- `docs/decisions/0029-p6-event-studies.md`: Architecture Decision Record.
- Learning notes: `docs/learn/event-study.md` and `docs/learn/block-bootstrap.md`.
