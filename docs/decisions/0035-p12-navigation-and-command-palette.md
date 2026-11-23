# 0035. P12 Unified Navigation Hierarchy and Command Palette

Date: 2026-11-22
Status: Accepted

## Context
As Global Market Radar expanded from core market state monitors to include institutional depth modules (Yield Curve Lab, CFTC Positioning, Options Implied Volatility, Connectedness Spillovers, Event Studies, Seasonal Fundamentals, Slow Global Layers, EVT Tails, and Track Record Validation), the top navigation bar became cluttered with 16+ horizontal links. This created navigation friction and visual bloat.

## Decision
1. **5-Group Functional Categorization:**
   Organized the entire platform architecture into 5 coherent domain groupings:
   - **Markets:** Today (`/`), Charts (27) (`/charts`), Country Profiles (`/countries`), Sector Breakdown (`/sectors`), Historical Analogues (`/analogues`), Cross-Asset Radar (`/radar`), Cross-Asset Matrix (`/cross-asset`).
   - **Drivers & Rates:** Macro Drivers (`/drivers`), Yield Curve Lab (`/rates`), Futures Positioning (`/positioning`), Central Bank Events (`/events`), Energy Fundamentals (`/fundamentals`).
   - **Risk & Vol:** Risk Lab (`/risk`), Implied Volatility (`/volatility`), Cross-Market Connectedness (`/connectedness`).
   - **Liquidity & Global:** Net Liquidity (`/liquidity`), Global Credit & Reserves (`/global`), India Flows (`/india`).
   - **Evidence & Audit:** Pre-Registered Track Record (`/track-record`), Mathematical Methodology (`/methodology`), Source Data Health (`/data-health`), Daily Briefing (`/briefing`), Corrections & Errata (`/corrections`).

2. **Universal Command Palette (`⌘K` / `Ctrl+K`):**
   - Transformed the search modal into a full keyboard-driven Command Palette (`AssetSearchModal.tsx`).
   - Indexes all 20+ analytical pages, 8 launch countries, 11 GICS sectors, key macro drivers, CFTC futures contracts, and individual market series.
   - Provides full keyboard accessibility (Arrow Up/Down, Enter to activate, Escape to close).

## Consequences
- Clean, uncluttered header adhering to visual discipline tokens and WCAG AA contrast standards.
- Instant keyboard navigation across the entire institutional suite.
