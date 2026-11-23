# 0031. P9 Physical Energy Inventories & 5-Year Seasonal Bands

Date: 2026-11-22
Status: Accepted

## Context
Module P9 of `PUSH_PACK_PLAN.md` deepens the commodities layer by tracking physical supply and storage levels against historical 5-year seasonal bands for Crude Oil, Gasoline, Distillates, and Natural Gas.

## Decisions Made
1. **Strict Exclusion of Current Year**:
   - For any active year $Y$, the 5-year seasonal envelope (min, max, mean) is computed strictly over completed prior calendar years $[Y-5 \dots Y-1]$.
   - The current year $Y$ is plotted against the band to detect physical inventory excesses or deficits without contaminating the historical benchmark.
2. **Standardized Week-of-Year Alignment**:
   - Uses ISO calendar weeks $1 \dots 52$.
   - Evaluates weekly build/draw ($\Delta 1W$) alongside deviation from the 5-year seasonal normal.
3. **Lag Disclosure**:
   - Discloses EIA publication schedules: "Published Wednesdays, data as of previous Friday".
4. **UI `/fundamentals`**:
   - `SeasonalBandChart.tsx` renders 5-year min/max confidence area, 5-year mean, prior year, and current year curve.

## Verification
- Unit tests in `pipeline/tests/unit/test_fundamentals.py` verify that current-year spikes are strictly excluded from the 5-year benchmark band.
