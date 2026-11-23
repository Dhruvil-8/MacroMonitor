# Milestone Walkthrough: M0 Foundation

**Milestone:** M0  
**Title:** Monorepo Foundation & Tooling Architecture  
**Governing Plan:** `docs/MASTER_PLAN.md`

---

## 1. What was built and why

A professional quantitative system requires absolute decoupling between data computation and presentation. We established a unified monorepo:
1. `pipeline/`: Python 3.11+ data engine with strict anti-lookahead mathematical guarantees, PyArrow Parquet storage, and Pydantic v2 data contract schemas.
2. `web/`: Next.js 14 App Router TypeScript frontend that pre-renders static HTML/JSON views with zero client-side server dependence.
3. `tokens.css`: A financial terminal design token architecture enforcing high-contrast, luminous dark-mode aesthetics, monospaced tabular figures, and AA accessibility.

---

## 2. Reading order of files

1. [pipeline/pyproject.toml](file:///d:/MacroMonitor/pipeline/pyproject.toml): Core Python package dependencies and build definition.
2. [web/package.json](file:///d:/MacroMonitor/web/package.json): Frontend dependencies, TypeScript configuration, and Next.js static build script.
3. [web/src/styles/tokens.css](file:///d:/MacroMonitor/web/src/styles/tokens.css): Visual discipline tokens (colors, typography, grid, spacing).
4. [.github/workflows/ci.yml](file:///d:/MacroMonitor/.github/workflows/ci.yml): Automated continuous integration pipeline (linting, tests, static export).

---

## 3. The 5 most important concepts & files

1. **Monorepo Separation of Concerns**: Python processes mathematical models and writes atomic JSON contracts; Next.js only draws figures.
2. **Deterministic Builds**: Static export (`output: 'export'`) means the frontend can be deployed on any static edge CDN without backend server costs.
3. **Tabular Numerals**: Financial figures use monospaced fonts (`JetBrains Mono`) with `font-variant-numeric: tabular-nums` to prevent column jittering when data updates.
4. **Design Token Palette**: Muted backgrounds (`#0a0b0e`), luminous accents (`#38bdf8`, `#34d399`), and semantic warning colors.
5. **CI Quality Gate**: Pull requests must pass Ruff linting, Pytest with coverage thresholds, Zod schema validation, and Next.js static compilation.

---

## 4. Worked numeric example: Tabular Layout Geometry

Consider a column of financial percentage returns: `+12.4%`, `-0.3%`, `+104.2%`.
In variable-width fonts (e.g. Arial or Times New Roman), digit `1` is narrower than `8`, causing right-aligned decimal points to misalign:
```text
Variable:
+12.4%
 -0.3%
+104.2%   <-- decimal points drift horizontally!

Tabular Monospace (JetBrains Mono):
 +12.4%
  -0.3%
+104.2%   <-- decimal points align on a strict vertical axis.
```

---

## 5. Three experiments for the owner

1. **Experiment 1 (Change a theme token):** Open `web/src/styles/tokens.css` and change `--bg-0` from `#0a0b0e` to `#050507`. Rebuild `web` to observe how subtle changes in background depth affect readability.
2. **Experiment 2 (Inspect bundle size):** Run `npm run build` in `web/` and inspect the First Load JS shared by all routes. Notice how keeping visualization libraries lazy keeps the initial page load under 100 kB.
3. **Experiment 3 (Trigger a CI failure):** Intentionally introduce a syntax error or unused import in `pipeline/src/radar/cli.py` and run `ruff check pipeline/src` to see the linter catch it.
