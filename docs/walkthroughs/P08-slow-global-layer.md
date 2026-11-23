# Walkthrough: P8 Slow Global Layer

## Overview
Module P8 provides structural macro backdrop indicators operating at low frequencies (weekly, monthly, quarterly) to ground daily market moves:
1. **BIS Real Effective Exchange Rates (REER):** Long-term currency valuation percentiles across launch countries.
2. **U.S. Treasury TIC Foreign Holdings:** Monthly tracking of foreign official and private holdings of US Treasuries.
3. **Central Bank Balance Sheets:** Tracks balance sheet expansions across the Federal Reserve, ECB, Bank of Japan, and Reserve Bank of India in local currency and USD.
4. **India Macro Profile:** RBI foreign exchange reserves and rolling institutional FPI cash market flows.
5. **Slow Global UI (`/global`):** Small multiples with expanding percentile strips, currency valuation ranking, and explicit lag labels ("Q2 data, published 15 Nov").

## Key Files
- `pipeline/src/radar/metrics/global_slow.py`: Slow global macro metrics and alignment engine.
- `web/src/app/global/page.tsx`: Global structural macro dashboard.
- `pipeline/tests/unit/test_global_slow.py`: Unit tests for frequency alignment and lag labels.
- `docs/decisions/0032-p8-slow-global-layer.md`: Architecture Decision Record.
- Learning note: `docs/learn/reer-and-global-liquidity.md`.
