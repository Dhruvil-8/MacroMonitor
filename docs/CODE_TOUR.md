# Global Market Radar: Code Tour

This document maps user-facing product features to their implementation in the repository.

---

## Architecture at a Glance

```text
MacroMonitor/
├── pipeline/                  # Python 3.11+ Quantitative Engine
│   ├── config/                # Declarative parameters, universe, calendars, drivers
│   │   ├── params.yaml        # Statistical parameters (windows, thresholds, seeds)
│   │   ├── universe.yaml      # Asset definitions with source feeds and fallbacks
│   │   ├── calendars.yaml     # Time zone close rules for market synchronization
│   │   ├── drivers.yaml       # Global macro factor definitions for Move Decomposer
│   │   └── templates.yaml     # Neutral briefing sentence templates
│   ├── src/radar/
│   │   ├── ingest/            # Zero-key data extractors (Yahoo, FRED, NSE)
│   │   ├── clean/             # MAD jump filter, forward-fill limits, health validator
│   │   ├── metrics/           # Pure statistical functions (anti-lookahead)
│   │   │   ├── returns_vol.py # Log/linear returns, realized vol, EWMA vol
│   │   │   ├── stats.py       # Expanding z-scores, percentiles, Neff, ACF, KDE
│   │   │   ├── decomposition.py # Move Decomposer (OLS + HAC, Ridge, FX identity)
│   │   │   ├── correlation.py # Rolling Pearson, downside correlation, distance matrices
│   │   │   ├── liquidity.py   # US Net Liquidity & stablecoin metrics
│   │   │   ├── india_flows.py # FII/DII rolling flows, F&O 4-quadrant OI matrix
│   │   │   ├── stress.py      # Stress composite & historical crisis comparison
│   │   │   └── context.py     # 25-bin histograms & declustered analogs
│   │   ├── insights/          # Rule-based observation engine & daily briefing
│   │   └── export/            # Pydantic v2 schemas and atomic JSON contract writers
│   └── tests/unit/            # Pytest suite with property and known-answer tests
│
├── web/                       # Next.js 14 App Router Static Application
│   ├── src/
│   │   ├── app/               # Statically prerendered pages
│   │   │   ├── page.tsx       # / (Today: Move Decomposer hero, state map, briefing)
│   │   │   ├── countries/     # /countries & /countries/[code]
│   │   │   ├── sectors/       # /sectors (US & India rotation & contribution)
│   │   │   ├── drivers/       # /drivers (sensitivity heatmap & forest plots)
│   │   │   ├── risk/          # /risk (vol cone, QQ, tail risk, PCA, drawdowns)
│   │   │   ├── liquidity/     # /liquidity (US net liquidity & stablecoins)
│   │   │   ├── radar/         # /radar (change feed & divergence panels)
│   │   │   ├── india/         # /india (FII/DII flows, OI matrix, PCR)
│   │   │   ├── context/[id]/  # /context/[id] (empirical distributions & analogs)
│   │   │   ├── briefing/      # /briefing (daily briefing archive & share cards)
│   │   │   └── methodology/   # /methodology (mathematical formulas & sources)
│   │   ├── components/
│   │   │   ├── charts/        # Apache ECharts & D3 statistical components
│   │   │   ├── cards/         # ChartCard with Method Drawer and Audit Drawer
│   │   │   └── layout/        # TopBar, Navigation, Command Palette (Cmd-K)
│   │   ├── styles/tokens.css  # Luminous dark editorial theme tokens
│   │   └── lib/data/          # Static JSON fetchers & Zod runtime contract validators
│   └── tests/                 # Node.js Zod contract validation tests
│
├── docs/                      # Technical Documentation & Learning Layer
│   ├── MASTER_PLAN.md         # Governing product & engineering blueprint
│   ├── AGENT_PLAN.md          # Technical appendix
│   ├── decisions/             # Architecture Decision Records (ADRs 0001-0013)
│   ├── walkthroughs/          # Milestone walkthroughs for the owner
│   ├── learn/                 # Concise statistical learning notes (<700 words)
│   └── glossary.md            # Definition of terms and formulas
│
└── research/notebooks/        # Jupyter notebooks reproducing calculations step-by-step
```
