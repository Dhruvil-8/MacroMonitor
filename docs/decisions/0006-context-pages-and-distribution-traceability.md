# 6. Context Pages and Distribution Traceability Strategy

Date: 2026-11-21
Status: Accepted

## Context
Milestone M7 provides deep-dive historical context (`/context/[id]`) for every active series across the 60-series universe, grounding current values against multi-year empirical distributions and declustered historical episodes.

## Decisions
1. **Unconditional Empirical Distribution**:
   - Computes return and level histograms across 25 dynamic bins.
   - Current observation is marked with a pinned vertical reference line and percentile annotation.

2. **Declustered Analogue Retrieval**:
   - Searches for the 5 closest historical readings $|x_t - x_{\text{past}}|$.
   - Enforces a strict minimum spacing of $\ge 20$ business days between selected dates to avoid clustering multiple days from a single crisis episode.

3. **SEO and Open Science Dataset Markup**:
   - Embeds schema.org `Dataset` JSON-LD metadata on every Context route.
   - Provides public metadata on spatial and temporal coverage, licensing terms, and upstream data provider provenance.

4. **Static Path Generation**:
   - `generateStaticParams()` pre-renders static HTML pages for all 60 asset IDs at build time.
