# Milestone Walkthrough: M1 Data Foundation

**Milestone:** M1 (M1a Global + M1b India)  
**Title:** Zero-Key Data Ingestion, Health Checks, and Jump Filtering  
**Governing Plan:** `docs/MASTER_PLAN.md`

---

## 1. What was built and why

Market data from free public feeds often contains bad ticks, duplicate dates, missing trading holidays, and sudden unadjusted splits. We built an ingestion and cleaning pipeline:
1. **Zero-Key Ingestion**: Direct extractors for Yahoo Finance (`yfinance`), Federal Reserve Bank of St. Louis (`fredapi` / web), and National Stock Exchange of India (NSE).
2. **Health Monitoring**: Tracks freshness, staleness, coverage, and missing streaks for every series. Aborts pipeline if critical series fail.
3. **MAD Jump Filter**: Non-parametric outlier filter using Median Absolute Deviation to detect and quarantine extreme single-day data artifacts without distorting true market volatility.
4. **Calendar Alignment**: Normalizes business days across different national holidays with bounded forward-filling.

---

## 2. Reading order of files

1. [pipeline/config/universe.yaml](file:///d:/MacroMonitor/pipeline/config/universe.yaml): 60+ series definitions, tickers, categories, and fallbacks.
2. [pipeline/src/radar/ingest/base.py](file:///d:/MacroMonitor/pipeline/src/radar/ingest/base.py): Base class for data extractors.
3. [pipeline/src/radar/clean/health.py](file:///d:/MacroMonitor/pipeline/src/radar/clean/health.py): Automated health checks and series diagnostics.
4. [pipeline/src/radar/clean/cleaner.py](file:///d:/MacroMonitor/pipeline/src/radar/clean/cleaner.py): Bounded forward-fill and MAD jump filter.
5. [pipeline/tests/unit/test_cleaner.py](file:///d:/MacroMonitor/pipeline/tests/unit/test_cleaner.py): Tests proving outlier rejection and boundary handling.

---

## 3. The 5 most important functions

1. `clean_series(s, max_ffill=3, jump_threshold=10.0)` in `cleaner.py`: Chains forward-fill limits with the jump filter.
2. `apply_jump_filter(s, window=60, threshold=10.0)` in `cleaner.py`: Replaces MAD outliers exceeding threshold standard deviations with NaN.
3. `check_series_health(df, series_meta)` in `health.py`: Computes staleness, missing rate, and health status (`ok`, `stale`, `failed`).
4. `evaluate_pipeline_health(health_df, abort_threshold=0.20)` in `health.py`: Halts the run if >20% of series fail.
5. `YahooIngestor.fetch(symbol, start_date)` in `yahoo.py`: Fetches historical adjusted closes with local caching.

---

## 4. Key formulas in words and symbols

### Median Absolute Deviation (MAD) Scale Estimator
For a rolling window of prices or returns $\{x_1, \dots, x_w\}$:
1. Find rolling median: $M = \text{median}(x)$
2. Compute deviations: $d_i = |x_i - M|$
3. Compute MAD: $\text{MAD} = \text{median}(d)$
4. Scale to normal distribution equivalent:
   $$\hat{\sigma}_{\text{MAD}} = 1.4826 \times \text{MAD}$$
5. Standardized distance:
   $$z_{\text{MAD}} = \frac{|x_t - M|}{\hat{\sigma}_{\text{MAD}}}$$
If $z_{\text{MAD}} > 10.0$, the observation is flagged as a jump artifact.

---

## 5. Worked numeric example

Consider 5 daily prices: `[100, 101, 100, 250, 102]`. (Notice `250` is a bad tick).
1. Median $M = 101$.
2. Deviations:
   - $|100 - 101| = 1$
   - $|101 - 101| = 0$
   - $|100 - 101| = 1$
   - $|250 - 101| = 149$
   - $|102 - 101| = 1$
3. Sorted deviations: `[0, 1, 1, 1, 149]`. Median deviation $\text{MAD} = 1.0$.
4. Scale $\hat{\sigma} = 1.4826 \times 1.0 = 1.4826$.
5. Test price `250`:
   $$z = \frac{|250 - 101|}{1.4826} = \frac{149}{1.4826} \approx 100.5\sigma$$
   Since $100.5 > 10.0$, the value `250` is rejected as an extreme jump.

---

## 6. Three experiments for the owner

1. **Experiment 1 (Change jump threshold):** Set `jump_threshold=3.0` in `params.yaml`. Run the cleaner on a volatile asset like Natural Gas. Notice that true market gaps get clipped if the threshold is set too low (which is why we use $10\sigma$).
2. **Experiment 2 (Test holiday alignment):** Run `clean_series` on an Indian asset during Diwali when US markets were open. Verify that forward-fill holds the previous close for up to 3 days without inventing data.
3. **Experiment 3 (Trigger pipeline abort):** Mock an empty DataFrame for 15 series in `test_health.py` and run `evaluate_pipeline_health`. Observe the system aborting gracefully rather than exporting corrupt artifacts.
