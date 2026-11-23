# Decision 0023: P0 Source Resilience Layer & Data Health Registry

## Status
Accepted

## Context
External free financial and macroeconomic data feeds (FRED, Yahoo, NSE, CFTC) are vulnerable to transient network failures, schema changes, and upstream maintenance outages. As mandated by `docs/PUSH_PACK_PLAN.md` Section 3 (P0), the ingestion pipeline must not break silently or corrupt historical series when an external source fails.

## Decision
1. **Source Registry**: Created `pipeline/config/sources.yaml` documenting all free data feeds, licensing terms, update schedules, rate limits, and fallback sources.
2. **Adapter Architecture**: Enhanced `BaseIngestor` in `pipeline/src/radar/ingest/base.py` with:
   - Three-state `CircuitBreaker` (CLOSED, OPEN, HALF_OPEN) with cooldowns.
   - `validate_schema` for runtime schema drift detection against mandatory column sets.
   - `validate_sanity` to block impossible price values or extreme artificial jumps (>50%).
3. **Data Health Matrix**: Added `SourceStatusEnvelope` to export `source_status.json` and built `/data-health` with the `SourceStatusTable` component to provide complete operational transparency.
4. **Testing**: Implemented `pipeline/tests/unit/test_resilience.py` validating circuit transitions, drift detection, and sanity bounds.
