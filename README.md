# Global Macro Transmission Monitor

A system for detecting structural breaks in historical relationships (transmission mechanisms) between global macro drivers and asset classes.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Repository:** [github.com/Dhruvil-8/MacroMonitor](https://github.com/Dhruvil-8/MacroMonitor)
**Live Demo:** [huggingface.co/spaces/Dhruvil8/MacroMonitor](https://huggingface.co/spaces/Dhruvil8/MacroMonitor)

## Development Story

This project represents a collaborative evolution in AI-assisted software engineering, leveraging next-generation models for planning, implementation, and refinement.

### Phase 1: Inception & Core Architecture
*   **Planning Strategy**: Developed with **Gemini** to outline the "Transmission Monitor" thesis—focusing on structural breaks rather than simple price alerts.
*   **v1.0 Implementation**: The initial codebase was architected and coded by **ChatGPT 5.2** paired with **Gemini 3 Pro**. This phase established the core "Engine" pattern, separation of concerns (Data/Analysis/Alerts), and the configuration-driven design (YAML).

### Phase 2: Refactoring & Advanced Analytics (Current)
*   **Environment**: Refactoring and modernization were conducted within the **Antigravity IDE** by Google.
*   **Engineering Lead**: **Claude Opus 4.5** worked in tandem with **Gemini 3 Pro** to upgrade the system infrastructure.
 
---

## Overview

The system continually monitors the correlation stability between macro drivers (e.g., US 10Y Yield, DXY, VIX, Oil) and downstream assets (Equities, Credit, FX). When a historical relationship significantly decouples or when a macro driver exhibits a transmission shock, the system generates a structured alert.

### Key Features

*   **Deterministic Logic**: Pure statistical analysis using rolling correlations, Z-scores, and Beta deviation.
*   **Data Pipeline**:
    *   **Stateless**: Fetches 10-year rolling history on startup.
    *   **Robust**: Handles missing data, survival bias, and weekend gaps via forward-fill and drop logic.
    *   **Hybrid Storage**: Supports committing historical baselines for instant startup.
*   **Interactive Analytics**: A "Quant Lab" dashboard allows real-time regime exploration and parameter stress-testing.

## Quantitative Methodology

### Alert Logic (The "Transmission Gate")
An alert is triggered only when the system detects a significant anomaly. This is governed by a multi-factor gate:

1.  **Correlation Break**: The rolling 30-day correlation z-score exceeds the threshold (default 2.0σ).
2.  **Macro Shock Override**: If a macro driver moves >4.0σ (Standard Deviations), an alert is forced regardless of correlation stability (capturing "Black Swan" events).
3.  **Beta Deviation**: The target asset moves in a direction opposite to what its historical Beta would predict.

## Deployment

### Docker Local
```bash
docker build -t macro-monitor .
docker run -p 7860:7860 macro-monitor
```

## Configuration

The system behavior is fully controlled via configuration files in `config/`:

*   **`asset_universe.yaml`**: Dictionary of all trackable assets (Tickers, Categories).
*   **`pairing_matrix.yaml`**: The logic map. Defines *which* macro driver is expected to influence *which* target asset (e.g., `DXY -> Emerging Markets`).
*   **`thresholds.yaml`**: Risk sensitivity parameters (Z-Score limits, Window sizes).

## Project Structure

```text
macro-monitor/
├── config/
│   ├── asset_universe.yaml    # Asset definitions (Tickers, Categories)
│   ├── pairing_matrix.yaml    # Macro-target pairings logic
│   └── thresholds.yaml        # Alert thresholds & window params
├── src/
│   ├── macro_monitor.py       # Main CLI orchestrator
│   ├── dashboard_api.py       # Dashboard analytics layer
│   ├── utils.py               # Logging & validation helpers
│   ├── engine/                # Core Quantitative Library
│   │   ├── data_engine.py     # Data ingestion & preparation
│   │   ├── returns.py         # Return methodology (Mixed: Log/Linear)
│   │   ├── correlation.py     # Rolling Z-scores
│   │   ├── beta.py            # Beta & deviation analysis
│   │   └── regime_detector.py # Alert gating logic & Shock detection
│   └── alerts/
│       └── formatter.py       # JSON formatting & persistence
├── dashboard/
│   └── dashboard_streamlit.py # Interactive Quant Lab (v2.1)
├── scripts/
│   ├── validate_scenarios.py  # Historical crash validation suite
│   └── backfill.py            # Data management utility
├── tests/                     # Pytest suite
├── data/processed/            # Output data (Local Cache)
├── Dockerfile                 # HF Spaces Deployment config
└── requirements.txt           # Python dependencies
```
Configuration
The system behavior is fully controlled via configuration files in config/:

asset_universe.yaml: Dictionary of all trackable assets.
pairing_matrix.yaml: The logic map. Defines which macro driver is expected to influence which target asset (e.g., DXY -> Emerging Markets).
thresholds.yaml: Risk sensitivity parameters (Z-Score limits, Window sizes).