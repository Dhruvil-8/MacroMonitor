# ADR 0001: Monorepo Architecture and Static Delivery Pattern

## Context
Global Market Radar requires a decoupled architecture that combines heavy statistical computations in Python with a zero-cost, high-performance web delivery layer.

## Decision
1. **Monorepo Structure**:
   - `pipeline/`: Python 3.11+ analytics pipeline (ingest, clean, metrics, insights, export).
   - `web/`: Next.js static export (`output: 'export'`) served over CDN (Cloudflare Pages).
   - `docs/`: Specs, ADRs, methodology, runbooks.

2. **Decoupled Static Delivery**:
   - Browser never connects to a stateful database or live Python process.
   - All analytics are pre-computed on a schedule (GitHub Actions) into compact JSON bundles validated by Pydantic $\leftrightarrow$ Zod contracts.
   - Zero-cost scalability, sub-second global latency, and high resilience against cloud service outages.

## Status
Accepted.
