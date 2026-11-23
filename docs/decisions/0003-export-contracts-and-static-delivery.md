# 3. Export Contracts and Static Delivery Strategy

Date: 2026-11-21
Status: Accepted

## Context
Milestone M3 establishes the data contracts and export architecture delivering computed statistics from the Python pipeline to the Next.js static frontend with zero runtime servers.

## Decisions
1. **Universal Versioned Envelope**:
   - Every JSON endpoint payload is wrapped in a strict metadata envelope containing `schema_version`, `as_of`, `code_version`, and `params_hash`.
   - The frontend validates `schema_version` and halts on breaking major version mismatches.

2. **Dual-Language Contract Parity**:
   - Python export models are defined in `pipeline/src/radar/export/schemas.py` using `pydantic.BaseModel`.
   - TypeScript frontend models are defined in `web/src/lib/data/schemas.ts` using `zod`.
   - Contract test suites in both Python (`test_contracts.py`) and Node (`test_contracts.mjs`) validate exact field parity.

3. **Size Optimization & Columnar Encoding**:
   - Time-series datasets are encoded in columnar arrays `{"d": [...], "v": [...]}` rather than repeating object keys per row.
   - Values are rounded to meaningful precision (2 decimals for prices, 4 significant digits for rates/ratios).
   - All exported files remain under 20 KB uncompressed, well below the 300 KB budget.

4. **Atomic Manifest**:
   - `manifest.json` is generated last as an atomic index of all datasets and their byte sizes, enabling the client to discover and load resources deterministically.
