# ADR 012: Scope: Macro-Only, No Individual Stocks

## Status
Accepted (Governed by `docs/MASTER_PLAN.md`)

## Date
2026-11-22

## Context
Initial planning discussions explored tracking stock-level data, insider filings, screeners, watchlists, or trade ideas. However, retail stock sites are abundant and noisy, whereas institutional-grade macro research tools that deconstruct market moves with statistical rigor are rare.

## Decision
Permanent out-of-scope boundaries are established:
1. **No individual stocks or tickers:** The platform tracks index, sector, country, currency, commodity, and interest rate series only. Fund proxies (e.g. SPY for S&P 500) may be used as stand-in data sources when primary index feeds are inaccessible, but are explicitly labeled as `proxy: true` in metadata and never presented as tradeable tickers in the UI.
2. **No trading or advisory language:** No "buy", "sell", price targets, forecasts, or algorithmic recommendations.
3. **No stock-level screeners, filings, or watchlists:** FII/DII and F&O data are tracked at aggregate and index levels only.
4. **Learning layer is in the repo, not the product:** Pedagogical walkthroughs, statistical learning notes, and exploratory notebooks live in `docs/` and `research/` for the owner, while the web UI maintains a clean, professional research publication tone with standardized Method and Audit drawers.

## Consequences
- The product remains focused, calm, and trustworthy.
- Simplifies the data pipeline and eliminates the overhead of tracking thousands of equity tickers and corporate actions.
- Avoids regulatory exposure under SEBI / global financial advisory rules by adhering strictly to descriptive statistics.
