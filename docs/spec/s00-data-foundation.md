# Specification: Data Foundation (s00-data-foundation)

## 1. Purpose
Defines the ingestion, storage, cleaning, and health-monitoring foundation for the 60 core macroeconomic and cross-asset series in the Global Market Radar pipeline.

## 2. Universe Contract (`pipeline/config/universe.yaml`)
Each series definition must conform to:
- `id`: Unique lowercase slug (e.g., `spx`, `nifty50`, `us10y`). Published URLs and foreign keys depend on this ID.
- `name`: Human-readable display label.
- `source`: Ingestion provider (`yahoo`, `fred`, `nse`).
- `source_symbol`: Ticker or series identifier at the upstream source.
- `group`: Asset class or category (`equity_index`, `equity_sector_us`, `equity_in`, `rates`, `fx`, `commodity`, `risk_gauge`, `liquidity`).
- `region`: Geographical economic region (`US`, `IN`, `EU`, `UK`, `JP`, `HK`, `CN`, `Global`).
- `unit`: Measurement unit (`index`, `bps`, `pct`, `usd`, `inr`, `usd_bn`).
- `frequency`: Native frequency (`D` for daily, `W` for weekly).
- `transform_default`: Canonical transformation for stationarity analysis (`log_return`, `diff`, `level`).
- `start_date`: ISO date `YYYY-MM-DD` marking earliest relevant history.
- `fallback`: Optional fallback string in format `<source>:<symbol>`.
- `license_note`: Clear description of redistribution terms and licensing restrictions.

## 3. Storage Layer (`pipeline/src/radar/db/store.py`)
Provides an idempotent storage abstraction:
- Works locally with Parquet/DuckDB tables under `data/processed/` when offline or when Supabase credentials are not provided.
- Interfaces with Supabase Postgres when `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` environment variables are present.
- Entities:
  - `series_meta`: Metadata for each active series.
  - `series_points`: Daily time-series values `(series_id, d, value, is_filled)`.
  - `series_health`: Daily health evaluation per series `(series_id, last_point, coverage_pct, fill_ratio, status, checked_at)`.
  - `pipeline_runs`: Operational run logs `(id, started_at, finished_at, status, code_version, params_hash, stale_series, notes)`.

## 4. Ingestion Layer (`pipeline/src/radar/ingest/`)
- Abstract base `BaseIngestor` implementing disk caching to `data/raw/<source>/<date>/`.
- `YahooIngestor`: Batch chunked downloader (`chunk_size=5`, `delay=0.15s`) with exponential backoff retry and individual fallback.
- `FREDIngestor`: Rates, yield curves, breakevens, and Fed balance sheet metrics. Supports FRED API key or cached public series files.
- `NSEIngestor`: Indian equity and index metrics with fallback to Yahoo Finance index proxies (`^NSEI`, `^NSEBANK`, etc.).

## 5. Cleaning Layer (`pipeline/src/radar/clean/cleaner.py`)
- Business Calendar Alignment:
  - Standardizes raw dates to timezone-naive business day DatetimeIndex (`YYYY-MM-DD`).
- Forward-Fill Limitation:
  - If gap $\le \text{max\_ffill\_days}$ (default 3 business days), fill with prior value and flag `is_filled = True`.
  - If gap $> \text{max\_ffill\_days}$, remaining missing dates stay NaN (`is_filled = False`).
- Outlier & Jump Filter:
  - Compute daily absolute log-returns or differences $r_t = |x_t - x_{t-1}|$.
  - Using expanding/rolling median and MAD (Median Absolute Deviation):
    $$\text{score}_t = \frac{|x_t - \text{median}(x)|}{1.4826 \times \text{MAD}(x)}$$
  - Points with score $> \text{jump\_sigma}$ (default 8.0) are flagged and replaced with NaN (to be handled by forward-fill policy).
- Duplicate and Impossible Value Rejection:
  - Prices $\le 0$ for non-spread assets rejected.
  - Duplicate timestamps sorted and deduplicated keeping latest valid observation.

## 6. Health & Staleness Evaluation (`pipeline/src/radar/clean/health.py`)
- For each series:
  - `last_point`: Max date observed.
  - `coverage_pct`: Proportion of expected business days present in the sample window.
  - `fill_ratio`: Proportion of points where `is_filled == True`.
  - Staleness check: If calendar gap between `last_point` and `as_of` $> \text{stale\_after\_hours}$ (adjusted for weekends), mark status as `'stale'`.
  - If missing entirely or coverage $< 50\%$, mark as `'failed'`.
  - Otherwise, status is `'ok'`.
- Run abort policy:
  - If `stale_count / total_series > max_stale_pct` (default 20%), the pipeline run logs `'partial'` or `'failed'` and halts before publishing.
