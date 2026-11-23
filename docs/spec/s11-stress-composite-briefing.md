# Technical Specification: s11-stress-composite-briefing

## 1. Objective
Specify the data aggregation, mathematical formula, crisis episode validation, automated daily briefing generation, and export contracts for the **Financial Stress Composite & Daily Market Briefing** (`/briefing`) in accordance with `MARKET_RADAR_AGENT_PLAN.md` Section 8.7 and Section 13.

---

## 2. Mathematical Definition

### 2.1 Constituent Components
The Global Financial Stress Composite aggregates six distinct dimensions of market friction:

| Component ($k$) | Constituent Proxies | Metric Transform | Target Unit |
|---|---|---|---|
| **Equity Stress** ($c_{\text{equity}}$) | `vix` (level) & `spx` (drawdown) | Mean of empirical expanding percentiles | $[0, 100]$ |
| **Rates Stress** ($c_{\text{rates}}$) | `us10y` (10-day yield volatility) | Empirical expanding percentile | $[0, 100]$ |
| **FX Stress** ($c_{\text{fx}}$) | `eurusd`, `usdjpy` (20-day realized vol) | Mean of empirical expanding percentiles | $[0, 100]$ |
| **Commodity Stress** ($c_{\text{comm}}$) | `brent`, `gold` (20-day realized vol) | Mean of empirical expanding percentiles | $[0, 100]$ |
| **Correlation Stress** ($c_{\text{corr}}$) | Average pairwise 60-day correlation across benchmark assets | Empirical expanding percentile | $[0, 100]$ |
| **Credit Stress** ($c_{\text{credit}}$) | High-yield OAS / corporate spread (if available) | Empirical expanding percentile | $[0, 100]$ |

### 2.2 Composite Aggregation Rule (Strictly Zero Imputation)
Let $K_t \subseteq \{1, 2, 3, 4, 5, 6\}$ be the set of components with valid, non-stale observations at date $t$.
$$S_t = \frac{1}{|K_t|} \sum_{k \in K_t} c_{k, t}$$

- **Zero Imputation**: If a component is missing at date $t$, it is dropped from the denominator $|K_t|$ and explicitly listed in the metadata `missing_components`. No synthetic values, means, or forward-fills across missing series are ever imputed.
- **Minimum Component Count**: $|K_t| \ge 3$ required to publish a valid composite score; otherwise status is marked `insufficient_data`.

### 2.3 Regime Bands
| Stress Score Range ($S_t$) | Regime Label | Visual Token |
|---|---|---|
| $S_t < 40.0$ | **Low Stress / Calm** | Emerald / `--pos` |
| $40.0 \le S_t < 70.0$ | **Normal Market Conditions** | Slate / `--accent` |
| $70.0 \le S_t < 85.0$ | **Elevated Stress** | Amber / `--warn` |
| $S_t \ge 85.0$ | **High Macro Stress / Crisis** | Rose / `--neg` |

---

## 3. Historical Episode Validation

The composite score must align with verified historical market episodes:
1. **Global Financial Crisis (October 2008)**: $S_t \ge 90.0$ (Peak cross-asset liquidation)
2. **US Sovereign Debt Downgrade / Eurozone Crisis (August 2011)**: $S_t \ge 80.0$
3. **COVID-19 Global Liquidity Shock (March 2020)**: $S_t \ge 92.0$ (Historic volatility spike)
4. **Global Yen Carry Trade Unwind (August 5, 2024)**: $S_t \ge 68.0$ (Violent FX and equity vol jump)

---

## 4. Daily Briefing Generation Rules
The Daily Briefing is an automated, rules-based synthesis generated without manual editing:
1. **Headline & Executive Summary**: Concise 2-sentence summary stating current Stress Composite score, dominant regime, and primary driver.
2. **Top Market Observations**: The top 3-5 verified observations from the Insight Engine.
3. **Cross-Asset Signals**: Active divergences (e.g. SPX vs Copper, Gold vs Real Yields) and sectoral rotation phases.
4. **Liquidity & Flows Pulse**: US Net Liquidity 4W delta, FII/DII cash market net flows, and total stablecoins supply.
5. **Safety Compliance**: Automated rejection of forbidden terms (`buy`, `sell`, `target`, `will rise`, `accumulate`, `short`, `recommendation`). Only descriptive statistics are generated.

---

## 5. Export Contracts

### `/v1/stress.json`
```json
{
  "schema_version": "1.0",
  "as_of": "2026-11-20",
  "code_version": "v1.0.0",
  "params_hash": "...",
  "data": {
    "current_stress": 52.4,
    "regime": "Normal Market Conditions",
    "components": {
      "equity": 58.2,
      "rates": 44.1,
      "fx": 51.0,
      "commodity": 49.5,
      "correlation": 59.2
    },
    "missing_components": ["credit"],
    "dates": ["2024-01-02", ...],
    "history": [48.2, ...],
    "episodes": [
      {"name": "COVID-19 Shock", "date": "2020-03-16", "stress": 94.2},
      {"name": "August 2024 Volatility Spike", "date": "2024-08-05", "stress": 72.5}
    ],
    "audit": {
      "formula": "S_t = mean(available_component_percentiles)",
      "parameters": {"min_components": 3},
      "series_used": ["vix", "spx", "us10y", "eurusd", "usdjpy", "brent", "gold"],
      "n": 750,
      "n_eff": 240.5
    }
  }
}
```
