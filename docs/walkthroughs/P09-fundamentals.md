# Walkthrough: P9 Energy Fundamentals & Seasonal Inventory Bands

## Overview
Module P9 links commodity driver moves (Crude Oil, Natural Gas) to underlying physical storage dynamics using official U.S. Energy Information Administration (EIA) data:
1. **EIA Weekly Series:** Tracks U.S. Commercial Crude Oil Stocks, Total Motor Gasoline Stocks, Distillate Fuel Oil Stocks, and Working Natural Gas in Underground Storage.
2. **5-Year Seasonal Bands:** For each calendar week, computes the 5-year historical min, max, and mean strictly excluding the active observation year to prevent look-ahead bias.
3. **Deviation Metrics:** Calculates storage deviation from the 5-year seasonal norm and expanding historical percentile.
4. **Fundamentals UI (`/fundamentals`):** Institutional seasonal band charts (plotted against the shaded 5-year envelope) with release-date lag badges.

## Key Files
- `pipeline/src/radar/metrics/fundamentals.py`: Seasonal inventory band computation engine.
- `web/src/app/fundamentals/page.tsx` & `web/src/components/charts/SeasonalBandChart.tsx`: UI analytics.
- `pipeline/tests/unit/test_fundamentals.py`: Unit tests verifying seasonal band exclusion and calendar week alignment.
- `docs/decisions/0031-p9-eia-fundamentals.md`: Architecture Decision Record.
- Learning note: `docs/learn/seasonal-inventory-bands.md`.
