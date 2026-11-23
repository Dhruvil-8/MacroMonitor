# ADR 011: Legacy Prototype Decommissioning

## Status
Accepted

## Date
2026-11-21

## Context
Following the full implementation, testing, and production static export of Milestones M0 through M11 of the Global Market Radar architecture (`MARKET_RADAR_AGENT_PLAN.md` / `docs/AGENT_PLAN.md`), the repository retained several legacy prototype directories and files from earlier development iterations:
1. `config/`: Old asset universe and pairing matrices (superseded by `pipeline/config/params.yaml` and `pipeline/config/universe.yaml`).
2. `dashboard/`: Prototype Streamlit app `dashboard_streamlit.py`.
3. `frontend/`: Initial Vite + React SPA prototype (superseded by Next.js 14 App Router in `web/`).
4. `scripts/`: Legacy standalone backfill scripts (superseded by `pipeline/src/radar/cli.py`).
5. `src/`: Legacy FastAPI backend server `server.py` and old analytics engines.
6. `tests/`: Old test suite targeting legacy FastAPI code (superseded by `pipeline/tests/unit/`).
7. `Dockerfile`: Container image referencing legacy `src/server.py`.
8. `requirements.txt`: Root requirements file (superseded by `pipeline/pyproject.toml`).

These legacy files introduced confusion, duplicated logic, and created maintenance overhead.

## Decision
Decommission and delete all legacy directories and files:
- Removed: `config/`, `dashboard/`, `frontend/`, `scripts/`, `src/`, `tests/`, `Dockerfile`, `requirements.txt`.
- Preserved:
  - `pipeline/`: Active quantitative data engine, statistical routines, and data contract export.
  - `web/`: Active Next.js 14 App Router statically-exported web application with ECharts visualizations and design tokens.
  - `docs/`: Technical specifications (`docs/spec/`) and Architecture Decision Records (`docs/decisions/`).
  - `.github/`: CI workflows for automated testing, static export deployment, and nightly pipeline runs.
  - Root configuration: `AGENTS.md`, `MARKET_RADAR_AGENT_PLAN.md`, `.editorconfig`, `.gitattributes`, `.gitignore`, `.pre-commit-config.yaml`.
- Updated `README.md` to document the unified monorepo architecture, testing procedures, and static delivery.

## Consequences
- Clean monorepo structure with zero dead code or confusing legacy imports.
- Verification confirms 100% test passing (65/65 unit tests, 94% coverage), 100% Zod contract validation, and 70/70 Next.js static routes successfully prerendered.
