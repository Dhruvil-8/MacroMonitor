# Walkthrough: P0 Source Resilience Layer

## Overview
Module P0 hardens the Global Market Radar data ingestion infrastructure against external provider failures, schema drifts, rate limits, and network volatility across free data feeds:
1. **Adapter Architecture (`BaseIngestor`):** Standardized interface with `fetch(since) -> raw`, `parse(raw) -> tidy`, `validate(tidy) -> report`, and `describe() -> metadata`.
2. **Resilience Engineering:** Exponential backoff with jitter, on-disk caching, circuit breaker protection (cooling down failing feeds), and schema-drift detection.
3. **Source Registry (`pipeline/config/sources.yaml`):** Comprehensive registry tracking documentation URLs, licensing/terms URLs, rate limits, schedules, and last-verified timestamps for FRED, Yahoo Finance, NSE India, CFTC, Cboe, EIA, and BIS.
4. **Data Health Dashboard (`/data-health`):** Real-time monitoring UI showing adapter status, circuit states, staleness badges, and terms links.

## Key Files
- `pipeline/src/radar/ingest/base.py`: Resilient ingestion adapter base class.
- `pipeline/config/sources.yaml`: Source registry and terms metadata.
- `pipeline/src/radar/clean/health.py`: Series and pipeline health evaluators.
- `web/src/app/data-health/page.tsx` & `web/src/components/ui/SourceStatusTable.tsx`: UI data health monitor.
- `pipeline/tests/unit/test_resilience.py`: Unit tests for backoff, circuit breaker, and drift detection.
- `docs/decisions/0023-p0-source-resilience-and-data-health.md`: Architecture Decision Record.
