# Specification: Export and Contracts (s04-export-contracts)

## 1. Purpose
Defines the standard data contracts, serialization schemas, and manifest-driven delivery model between the quantitative Python pipeline and the Next.js static web frontend.

## 2. Universal Envelope
Every exported JSON file must be wrapped in a versioned, reproducible envelope:
```json
{
  "schema_version": "1.0",
  "as_of": "YYYY-MM-DD",
  "code_version": "v1.0.0",
  "params_hash": "<12-char sha256>",
  "data": { ... }
}
```

## 3. Core Export Endpoints (`/v1/`)
- `manifest.json`: Root inventory of all exported datasets, their file hashes, record counts, schema versions, and `as_of` timestamp.
- `pulse.json`: Top-level macro summary, market state overview, and top observations.
- `state_map.json`: Quadrant positions (trend strength t-stat, volatility percentile, momentum change) for all active universe assets.
- `vol_landscape.json`: Realized volatility across 10D, 20D, 60D windows, historical percentile, and volatility trend ($vol_{20}/vol_{60} - 1$).
- `divergences.json`: Z-score divergences across pre-registered asset pairs.
- `liquidity.json`: Federal Reserve balance sheet metrics, TGA, RRP, Net Liquidity, and 4W/13W changes.
- `series/<id>.json`: Compact columnar time-series `{"d": ["YYYY-MM-DD", ...], "v": [100.0, ...]}` with 4 significant figures or 2 price decimals.

## 4. Size & Performance Constraints
- Columnar arrays (`{"d": [...], "v": [...]}`) for time series rather than repeated object keys.
- Number precision clamped to domain significance (2 decimals for prices, 1 decimal for returns/percentages, 4 digits for ratios).
- Target size: each individual endpoint payload $< 300$ kB gzipped.

## 5. Dual-Language Enforcement
- Python schemas: Defined with `pydantic.BaseModel` in `pipeline/src/radar/export/schemas.py`.
- TypeScript schemas: Defined with `zod` in `web/src/lib/data/schemas.ts`.
- Contract validation test guarantees parity.
