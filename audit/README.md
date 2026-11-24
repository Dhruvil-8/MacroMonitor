# Global Market Radar - Independent Audit & Validation

This directory contains the independent audit harness, verification oracles, reconciliation scripts, Playwright UI crawlers, canary tests, external reference fixtures, and official audit reports for the Global Market Radar codebase.

Audited against:
- `docs/AUDIT_PLAN.md`
- `docs/MASTER_PLAN.md`
- `docs/AGENT_PLAN.md`
- `docs/PUSH_PACK_PLAN.md`
- `docs/spec/s*.md`

## Structure
- `snapshot/<date>/`: Frozen snapshot of database parquet tables and `/v1/` JSON bundles with SHA-256 checksums.
- `oracles/`: Independent, clean-room re-implementations of each mathematical metric built strictly from specifications without inspecting production internals.
- `data_checks/`: Source reconciliation and raw-to-clean data integrity scripts.
- `contract_checks/`: Pydantic & Zod schema consistency, envelope integrity, and null/precision checks.
- `ui_checks/`: Playwright headless crawlers inspecting DOM text, canvas/SVG charts, ECharts option trees, tooltips, and URL state round-trips.
- `canaries/`: Error-injection canary test suite verifying 100% error catch rate.
- `fixtures/`: Hand-calculated benchmarks, external reference values (closes, yields, COT reports), and synthetic test series.
- `results/`: Machine-readable audit check outputs and coverage matrix.
- `findings.csv`: Master register of all findings, severities, root causes, regression tests, and statuses.
- `traceability.csv`: Complete end-to-end lineage mapping visible UI elements to backend pipeline sources.
- `AUDIT_REPORT.md`: The definitive independent audit report with findings and certification.
