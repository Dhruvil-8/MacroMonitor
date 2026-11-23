# Walkthrough: P11 Track Record and Out-of-Sample Validation

## Overview
Module P11 serves as Global Market Radar's institutional credibility engine:
1. **Pre-Registration Discipline:**
   - Formal mathematical hypotheses locked in `docs/spec/track_record.md`.
   - Fixed 20-day forward evaluation horizon.
   - Zero look-ahead bias: active alerts under 20 days old remain pending.
2. **Alert Declustering:**
   - Grouping consecutive alerts within 10 trading days into independent episodes to prevent artificial statistical inflation during crises.
3. **Base-Rate Benchmarking & Block-Bootstrap Inference:**
   - Measuring empirical hit rates against unconditional base rates on random days.
   - 1,000-iteration stationary block-bootstrap confidence intervals ($b=20$ days) for net statistical edge.
   - Honest disclosure: rules without demonstrated edge are labeled **"Context Only"**.
4. **Calibration & Rule Changelog:**
   - Monotonic severity tiering (Info, Low, Medium, High, Severe).
   - Git-verified changelog tracking parameter refinements with commit hashes.

## Key Files
- `docs/spec/track_record.md`: Pre-registration specifications.
- `pipeline/src/radar/metrics/track_record.py`: Evaluation and bootstrap engine.
- `web/src/components/charts/TrackRecordBars.tsx`: Visual hit rate vs base rate comparison and calibration cards.
- `web/src/app/track-record/page.tsx`: Comprehensive track record dashboard.
- `pipeline/tests/unit/test_track_record.py`: Unit test suite (4/4 passed).
- `docs/decisions/0034-p11-track-record-validation.md`: ADR.
- Learning note: `docs/learn/track-record-validation.md`.
