# 0034. P11 Pre-Registered Track Record and Out-of-Sample Validation

Date: 2026-11-22
Status: Accepted

## Context
Financial dashboards routinely suffer from post-hoc hindsight bias: rules and indicators are adjusted retrospectively after market crashes to make past signals look prescient ("p-hacking" and "overfitting"). When market conditions change, these unanchored rules collapse. To serve as a credible institutional intelligence platform, Global Market Radar must publish its empirical track record with unbending scientific discipline.

## Decision
1. **Pre-Registration:**
   - Every observation rule family is tied to an explicit, pre-registered mathematical hypothesis locked in `docs/spec/track_record.md` before results are computed.
   - Forward evaluation horizon is fixed at 20 trading days.
   - Point-in-time discipline: no outcome is computed until the 20-day forward horizon has fully elapsed. Alerts triggered within the trailing 20 days remain in "Active - Horizon Pending" state.

2. **Episode Declustering:**
   - Consecutive alerts for the same rule within 10 business days are clustered into a single independent episode. This prevents double-counting correlated stress episodes (e.g. 5 alerts during a single 1-week crisis).

3. **Base-Rate Benchmarking & Block-Bootstrap Inference:**
   - For every rule, we calculate the unconditional probability of the target outcome on randomly chosen unalerted dates (the base rate).
   - We compute stationary block-bootstrap 95% confidence intervals (block length = 20 days, $K=1,000$ iterations) over the net statistical edge $(\text{Hit Rate} - \text{Base Rate})$.
   - Rules where the confidence interval includes zero or where edge is negligible ($\le 5\%$) are openly classified as **"Context Only"** and published without sugarcoating.

4. **Calibration & Rule Changelog:**
   - Monotonicity check across severity bands (Info, Low, Medium, High, Severe) to ensure alert severity correlates with empirical shock probability.
   - Git-verified rule changelog documenting any parameter refinements with commit hashes and version stamps.

## Consequences
- Creates an unassailable credibility moat: users can audit exactly how well each observation performed historically out-of-sample.
- Eliminates confirmation bias and holds the platform accountable.
