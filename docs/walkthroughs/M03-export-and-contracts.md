# Milestone Walkthrough: M3 Export and Contracts

**Milestone:** M3  
**Title:** Strongly-Typed Data Contracts and Atomic Manifest Export  
**Governing Plan:** `docs/MASTER_PLAN.md`

---

## 1. What was built and why

To ensure the web application never crashes from missing keys, null fields, or type mismatches, the boundary between Python and TypeScript is governed by two mirroring schema libraries:
1. **Pydantic v2 in Python**: Validates every numerical array, timestamp, and metadata dictionary before it is written to disk.
2. **Zod v3 in TypeScript**: Validates every imported JSON contract at build time and runtime.
3. **Atomic Manifest**: A root `manifest.json` containing cryptographic hash, schema version, generation timestamp, and asset index.

---

## 2. Reading order of files

1. [pipeline/src/radar/export/schemas.py](file:///d:/MacroMonitor/pipeline/src/radar/export/schemas.py): Pydantic contract definitions.
2. [web/src/lib/data/schemas.ts](file:///d:/MacroMonitor/web/src/lib/data/schemas.ts): Zod schema equivalents in TypeScript.
3. [pipeline/src/radar/export/writers.py](file:///d:/MacroMonitor/pipeline/src/radar/export/writers.py): JSON serialization and atomic file export logic.
4. [web/tests/test_contracts.mjs](file:///d:/MacroMonitor/web/tests/test_contracts.mjs): Standalone Node.js test runner validating exported artifacts.

---

## 3. The 5 most important functions

1. `export_manifest(processed_dir, out_dir)` in `writers.py`: Creates `manifest.json` with metadata and series health status.
2. `export_pulse(processed_dir, out_dir)` in `writers.py`: Generates the market pulse summary contract.
3. `export_series_context(processed_dir, out_dir)` in `writers.py`: Exports 25-bin histograms and declustered analogs for all 60 series.
4. `ManifestSchema.model_validate(data)` in `schemas.py`: Pydantic validation of the manifest contract.
5. `ManifestSchema.parse(data)` in `schemas.ts`: Zod validation ensuring zero type drift in the frontend.

---

## 4. Key formulas in words and symbols

### Atomic File Write Guarantee
Directly overwriting an active production JSON file risks serving corrupted partial files to web visitors if a crash occurs mid-write. We use atomic filesystem renaming:
1. Write JSON data to temporary sibling file: `target.json.tmp.<pid>`
2. Flush to physical disk: `f.flush(); os.fsync(f.fileno())`
3. Atomic POSIX / Windows rename: `os.replace(temp_path, target_path)`
This ensures readers always observe either the complete previous version or the complete new version.

---

## 5. Worked numeric example: Contract Validation

Suppose Python exports an analog record:
```json
{
  "date": "2020-03-16",
  "distance": 0.421,
  "subsequent_return_20d": 0.125
}
```
If Python accidentally serializes `distance` as `"0.421"` (a string instead of float), Zod rejects it immediately:
```text
ZodError: Expected number, received string at "distance"
```
The build fails before any user can see an NaN chart.

---

## 6. Three experiments for the owner

1. **Experiment 1 (Run contract validator):** Run `node web/tests/test_contracts.mjs` in your terminal. Verify that all 129 generated JSON files pass 100% of their schema checks.
2. **Experiment 2 (Simulate a contract violation):** Open `web/public/data/v1/manifest.json` and change `"version": "1.0.0"` to a number `1.0`. Run `node web/tests/test_contracts.mjs` and see Zod catch the type mismatch.
3. **Experiment 3 (Inspect file hashes):** Look at `manifest.json`. Notice the list of series files with their last modified timestamps, allowing client-side caching to only fetch updated assets.
