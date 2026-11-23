# MacroMonitor — Macro Statistics

[![CI](https://github.com/Dhruvil-8/MacroMonitor/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhruvil-8/MacroMonitor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 14](https://img.shields.io/badge/next.js-14-black.svg)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue.svg)](https://www.typescriptlang.org/)

A macro regime monitoring and market statistics dashboard across 60 multi-asset series (Equities, Sovereign Rates, Foreign Exchange, Commodities, and India). Built with a decoupled architecture featuring a Python calculation pipeline and a static Next.js frontend with Apache ECharts.

> [!NOTE]
> **Validation Status**: This application was generated with AI / LLM assistance as an analytical macro statistics platform with **further validation pending**. All quantitative metrics, regime classifications, and statistical indicators are strictly descriptive and provided for monitoring and research purposes. They do not constitute investment advice or trading recommendations.
---

## Architecture Overview

The system is organized as a decoupled monorepo:

```text
MacroMonitor/
├── pipeline/                  # Python 3.11+ Calculation Pipeline
│   ├── config/
│   │   ├── params.yaml        # Single source of truth for quantitative parameters
│   │   ├── universe.yaml      # 60-series asset dictionary with primary and fallback feeds
│   │   ├── sources.yaml       # Feed registry, rate limits, and circuit breaker rules
│   │   ├── curves.yaml        # Sovereign yield curve maturities and Nelson-Siegel anchors
│   │   ├── cot_contracts.yaml # 16 CFTC futures contracts and market codes
│   │   └── events.yaml        # Central bank decision calendar dates (FOMC, RBI MPC)
│   ├── src/radar/             # Data engine, models, and export writers
│   └── tests/unit/            # Unit test suite (157 unit tests)
├── web/                       # Next.js 14 App Router Frontend (Static Export)
│   ├── src/
│   │   ├── app/               # Application routes
│   │   ├── components/charts/ # 51 interactive chart components (Apache ECharts)
│   │   └── lib/data/          # Static data loaders & Zod runtime contract validators
│   └── public/data/v1/        # Pre-calculated static JSON data contracts
└── data/                      # Local raw caches and processed parquet storage
```

---

## Core Views & Capabilities

1. **Market Pulse (`/`)**
   - **Market State Map**: Multi-asset overview with 60-day trend t-statistics, 20-day realized volatility percentiles, and quadrant classification.
   - **Volatility Landscape**: Cross-asset realized volatility term structure (10D, 20D, 60D).

2. **Chart Gallery (`/charts`)**
   - 51 interactive chart components covering returns decomposition, rolling macro betas, yield curves, positioning ladders, connectedness networks, and risk distributions.

3. **Yield Curve Lab (`/rates`)**
   - **Nelson-Siegel 3-Factor Fit**: Diebold-Li parametric decomposition into Level ($L$), Slope ($S$), and Curvature ($C$).
   - **Term Structure Heatmap & Anatomy**: Par yield curve history and 10Y real vs. breakeven inflation breakdown.

4. **Futures Positioning (`/positioning`)**
   - **CFTC Positioning Ladder**: 156-week expanding percentile rankings across 16 futures contracts (equities, rates, commodities, FX).
   - **Time Series Dynamics**: Non-commercial net positioning as a percentage of total open interest.

5. **Risk Lab (`/risk`)**
   - **Extreme Value Theory (EVT)**: Peaks-Over-Threshold (POT) Generalized Pareto tail fitting, shape parameter $\xi$, and 99%/99.5% VaR & Expected Shortfall.
   - **Markov-Switching Regimes**: Hamilton 2-state latent stress probability ribbon and duration estimates.
   - **External Benchmarks**: Cross-correlation checks against Chicago Fed NFCI and St. Louis Fed FSI.

6. **Central Bank Events (`/events`)**
   - **Event Studies**: Cumulative Abnormal Return (CAR) ribbons for FOMC and RBI MPC rate announcements with block-bootstrap confidence bands.

7. **Connectedness & Spillovers (`/connectedness`)**
   - **Diebold-Yilmaz GFEVD**: Directional shock spillover matrix, Total Connectedness Index (TCI), and circular network graph.

8. **Energy Fundamentals (`/fundamentals`)**
   - **Seasonal Inventory Envelopes**: Weekly EIA commercial crude, gasoline, distillate, and natural gas inventories compared against 5-year historical ranges.

9. **Liquidity & Macro Flows (`/liquidity`, `/india`)**
   - **US Net Liquidity**: Federal Reserve balance sheet liquidity ($L_t = \text{WALCL} - \text{TGA} - \text{RRP}$).
   - **India Cash & F&O**: FII/DII institutional cash flow tracking, Put-Call Ratio (PCR), and open interest buildup classification.

10. **Track Record & Validation (`/track-record`)**
    - Out-of-sample forward evaluation of macro alerts against unconditioned base rates over 20-day horizons.

11. **Source Data Health (`/data-health`)**
    - Health checks, rate limit tracking, schema verification, and circuit breaker status across all upstream data feeds.

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 20+ and npm

### Environment Configuration

Copy the example environment file and add any optional keys:
```bash
cp .env.example .env
```

| Key | Status | Description |
|---|---|---|
| `FRED_API_KEY` | Optional | FRED API key (free from St. Louis Fed) |
| `EIA_API_KEY` | Optional | Energy Information Administration API key (free from EIA) |
| `CFTC_APP_TOKEN` | Optional | Socrata app token (CFTC works without token via public limits) |
| `ALPHAVANTAGE_API_KEY` | Optional | Fallback market data provider |

### 1. Python Pipeline

Install the pipeline in editable mode:
```bash
pip install -e pipeline/
pip install pytest pytest-cov
```

Run test suite:
```bash
$env:PYTHONPATH="pipeline/src"; pytest pipeline/tests/unit/
```

Run pipeline tasks:
```bash
# Ingest data
python -m radar.cli ingest --universe pipeline/config/universe.yaml

# Clean and validate
python -m radar.cli clean

# Compute rolling metrics & models
python -m radar.cli compute

# Export JSON files for web dashboard
python -m radar.cli export --out-dir web/public/data/v1/
```

### 2. Web Application

Navigate to the `web` workspace and install dependencies:
```bash
cd web
npm install
```

Run the development server:
```bash
npm run dev
```
Open `http://localhost:3000` in your browser.

Build the static distribution:
```bash
npm run build
```

---

## Configuration

All model parameters and asset universes are declarative and version-controlled:
- `pipeline/config/params.yaml`: Window lengths, thresholds, degrees of freedom, and Nelson-Siegel parameters.
- `pipeline/config/universe.yaml`: Tracked asset tickers, descriptions, and primary/fallback data sources.
- `pipeline/config/sources.yaml`: Source feed registry, rate limits, and circuit breaker thresholds.
- `pipeline/config/cot_contracts.yaml`: CFTC contract market codes and asset classes.
- `pipeline/config/events.yaml`: Policy meeting dates and event study windows.

---