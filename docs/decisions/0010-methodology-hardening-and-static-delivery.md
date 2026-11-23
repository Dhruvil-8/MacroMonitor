# ADR 0010: Methodology Documentation, Security Hardening, and Static Delivery

## Status
Accepted

## Context
As the final foundation of the platform launch, Milestone M11 establishes:
1. Complete transparency of all mathematical definitions, sources, and parameters on `/methodology`.
2. Static-first delivery (`output: 'export'`) ensuring zero server dependencies, instant static caching on Cloudflare Pages, and high resilience against traffic spikes.
3. Strict regulatory non-advisory compliance, automated security headers, and XML sitemaps.

## Decisions
1. **Interactive Methodology Hub**:
   - Organized into four structured tabs: Mathematical Specifications, Data Sources & Lineage, Hyperparameters (`params.yaml`), and Regulatory Compliance.
   - Every metric clearly lists its underlying spec file (`docs/spec/s00` through `s11`).

2. **Security & Caching Headers**:
   - `_headers` configured with Content-Security-Policy (CSP), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, and `Permissions-Policy`.
   - Data endpoints (`/data/v1/*.json`) assigned `Cache-Control: public, max-age=300, stale-while-revalidate=3600`.

3. **Static Prerendering**:
   - 70 distinct routes (including all 60 series context pages) statically prerendered with zero client-server runtime coupling.
   - Canonical XML sitemap and `robots.txt` generated for search engine indexing.

## Consequences
- 100% adherence to Section 19 Definition of Done.
- Zero server maintenance overhead.
- High performance, traceable, and regulatory-safe macro platform.
