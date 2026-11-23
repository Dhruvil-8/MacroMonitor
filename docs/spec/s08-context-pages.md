# Specification: Context Pages and Distribution Traceability (s08-context-pages)

## 1. Purpose
Defines the deep-dive historical context page (`/context/[id]`) available for every active series in the 60-series universe. Discloses full empirical distributions, peer rankings, and declustered similar past readings.

## 2. Statistical Components (Section 8.10)
- **Distribution Histogram**:
  - Unconditional empirical distribution of daily stationary returns or metric values.
  - Current reading annotated with a pinned vertical marker line and exact percentile callout.
- **Percentile History Ribbon**:
  - Multi-year rolling ribbon showing expanding percentile trajectory across regime cycles.
- **Peer Rank**:
  - Ranking within the asset's peer group (e.g. S&P 500 among global equity indices, Brent among commodities).
- **Declustered Similar Past Readings**:
  - Identifies the 5 historical dates with the closest statistical distance $|x_t - x_{\text{past}}|$.
  - **Declustering constraint**: Minimum gap of $\ge 20$ trading business days between selected analogue dates to avoid reporting multiple consecutive days from a single crisis episode.
  - Reports subsequent 20-day and 60-day path following each analogue.
- **Structured Data (SEO / Open Data)**:
  - JSON-LD `Dataset` schema markup on every Context page disclosing variable name, description, temporal coverage, spatial coverage, and source provider.
- **Static Route Generation**:
  - `generateStaticParams()` pre-rendering routes for all 60 series slugs (`spx`, `ndx`, `nifty50`, `us10y`, etc.).
