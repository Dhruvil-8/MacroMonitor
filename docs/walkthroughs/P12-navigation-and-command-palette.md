# Walkthrough: P12 Unified Navigation Hierarchy and Command Palette

## Overview
Module P12 consolidates the 20+ analytical views of Global Market Radar into a cohesive, highly accessible navigation structure:
1. **5-Group Functional Categorization:**
   - **Markets:** Today, Charts (27), Countries, Sectors, Analogues, Radar, Cross-Asset.
   - **Drivers & Rates:** Macro Drivers, Yield Curve Lab (P4), Futures Positioning (P2), Central Bank Events (P6), Energy Fundamentals (P9).
   - **Risk & Vol:** Risk Lab (P10), Implied Volatility (P3), Connectedness (P7).
   - **Liquidity & Global:** Net Liquidity, Global Credit & Reserves (P8), India Flows.
   - **Evidence & Audit:** Track Record (P11), Methodology, Source Data Health (P0), Daily Briefing, Corrections.
2. **Universal Command Palette (`⌘K` / `Ctrl+K`):**
   - Instant search across all 20+ pages, 8 launch countries, 11 sectors, macro drivers, CFTC contracts, and individual market series.
   - Full keyboard navigation with arrow selection and Enter activation.

## Key Files
- `web/src/components/layout/HeaderNav.tsx`: Dropdown menu system and mobile accordion.
- `web/src/components/ui/AssetSearchModal.tsx`: Upgraded universal Command Palette.
- `web/src/app/layout.tsx`: Root layout integrating the navigation hierarchy.
- `docs/decisions/0035-p12-navigation-and-command-palette.md`: ADR.
