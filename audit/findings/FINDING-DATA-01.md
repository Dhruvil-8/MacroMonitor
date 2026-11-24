# Audit Finding Detail: FINDING-DATA-01

**ID:** FINDING-DATA-01  
**Severity:** High  
**Phase:** Phase B (Data Layer Audit)  
**Area:** B2 Data Cleaning (Jump & Outlier Filter)  
**Location:** `pipeline/src/radar/clean/cleaner.py:44-67`  
**Status:** FIXED  
**Verified By:** Independent Auditor  

---

## 1. Description & Expected Behavior
According to `docs/spec/s00-data-foundation.md`, data cleaning must adhere to strict anti-lookahead rules:
> "Strict anti-lookahead: no computation at time t may use data from time > t."

Furthermore, jump filtering is intended to remove corrupted vendor feed artifacts (e.g. 10x multiplier fat-finger errors, bad zeroes, unadjusted splits), while preserving genuine historical market crisis moves.

## 2. Actual Behavior & Impact
The `filter_jumps` implementation in `cleaner.py` was utilizing a centered rolling Median Absolute Deviation (`rolling(..., center=True)`). Centered rolling computations violate the strict anti-lookahead requirement by using future observations $t+1 \dots t+30$.

Furthermore, evaluating jumps directly on price levels against a 60-day centered median caused the genuine market shock of **August 5, 2024** (Yen carry trade crash) to be flagged as an outlier:
- On August 2, 2024, VIX closed at 23.39.
- On August 5, 2024, VIX spiked to 38.57.
- Because pre-crisis summer 2024 volatility was calm (VIX 12–14), the level deviation generated a Z-score of $z = 15.74 > 8.0$.
- Consequently, August 5 was replaced with `NaN` and forward-filled with Friday's close (23.39), erasing one of the most significant macro volatility events of 2024.

## 3. Reproduction Steps
1. Inspect raw VIX Yahoo Finance parquet at `data/raw/yahoo/2026-11-21/VIX.parquet` on `2024-08-05`: value is `38.57`.
2. Run baseline `clean_series` with `jump_sigma=8.0` and `center=True`.
3. Inspect `data/processed/series/vix.parquet` on `2024-08-05`: value was `23.39` with `is_filled=True`.

## 4. Root Cause
1. `center=True` in `rolling` window calculations introduced look-ahead bias.
2. Inherent positive skew and regime switching in volatility assets (`vix`, `indiavix`, `move`) was not accounted for by a rigid 8-sigma price-level threshold calibrated for equities.

## 5. Remediation Applied
1. Changed `filter_jumps` to use strictly trailing rolling window (`center=False`), enforcing 100% anti-lookahead compliance across all series.
2. Added an adaptive jump threshold (`jump_sigma=max(jump_sigma, 20.0)`) for volatility and crisis-sensitive gauges in `clean_series`. This preserves genuine market spikes (August 5 VIX $z \approx 15.25 \le 20.0$) while decisively catching vendor corruption errors (Canary 10x jump $z = 221.09 > 20.0$).
3. Re-ran `clean_series` on VIX and re-exported processed parquet and downstream analytical bundles.

## 6. Verification & Regression Tests
- **Regression Test:** `pipeline/tests/unit/test_audit_regressions.py::test_vix_august_2024_spike_preserved` PASSED.
- **Look-Ahead Test:** `pipeline/tests/unit/test_audit_regressions.py::test_jump_filter_strictly_trailing_anti_lookahead` PASSED.
- **Evidence:** `data/processed/series/vix.parquet` at `2024-08-05` confirms `value = 38.570000`, `is_filled = False`.
