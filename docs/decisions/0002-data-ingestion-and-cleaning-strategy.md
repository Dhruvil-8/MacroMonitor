# 2. Data Ingestion, Cleaning, and Health Evaluation Strategy

Date: 2026-11-21
Status: Accepted

## Context
Milestone M1 establishes the ingestion and data foundation for 60 macroeconomic and cross-asset series spanning Global Equity, US Sectors, Indian Equities, Rates, FX, Commodities, Risk Gauges, and Liquidity.

The pipeline must support reliable daily execution, zero external dependency costs, and resilience against upstream API timeouts and rate limiting (such as Yahoo Finance connection drops and FRED rate throttling).

## Decisions
1. **Multi-Source Ingestion & Fallback Architecture**:
   - `YahooIngestor`: Downloader configured with batch chunking (`chunk_size=5`), delays, exponential backoff, and fallback symbol support (e.g., ETFs for cash indices).
   - `FREDIngestor`: Direct public CSV endpoint fetching (no API key required for core execution), with fallback support for official API keys when configured.
   - `NSEIngestor`: Official Indian index proxies (`^NSEI`, `^NSEBANK`, `^CNXIT`) via Yahoo Finance with caching.

2. **Cleaning & Alignment Discipline**:
   - Timezone-naive normalization to business day (`B`) frequency.
   - Forward-fill limited to `max_ffill_days` (default 3 business days) with explicit boolean tracking column `is_filled`.
   - Jump and outlier detection using rolling Median Absolute Deviation (MAD) scaled for normal consistency ($1.4826 \times \text{MAD}$) with threshold `jump_sigma` (default 8.0).
   - Non-positive price rejection on equity/index series.

3. **Storage Abstraction**:
   - Dual-mode `DataStore` that operates locally on fast Parquet tables in `data/processed/` for local and CI workflows.
   - Idempotent schema design mirror of Postgres/Supabase schema (`series_meta`, `series_points`, `series_health`, `pipeline_runs`).

4. **Health & Staleness SLA**:
   - Automated evaluation of data coverage, fill ratio, and business day staleness (`stale_after_hours = 36`).
   - Abort policy: If $> 20\%$ of series are stale or failed, pipeline execution fails before publishing to prevent corrupted public state.

## Consequences
- High resilience to single-source outages.
- Clear auditability on whether any data point was raw or forward-filled.
- Zero credential requirements to run the entire pipeline locally or in public CI.
