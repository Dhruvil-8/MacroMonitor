# Technical Specification: s09-liquidity-module

## 1. Objective
Specify the data ingestion, unit alignment, mathematical calculation, equity overlay, and JSON export contract for the **Global Liquidity & Funding Monitor** (`/liquidity`) in accordance with `MARKET_RADAR_AGENT_PLAN.md` Section 8.8 and Section 13.

---

## 2. Mathematical Definition

### 2.1 Constituent Series & Source Units
| Identifier | Series Name | Source | FRED Symbol | Native Frequency | Native Unit | Target Unit | Conversion |
|---|---|---|---|---|---|---|---|
| `fed_assets` | Federal Reserve Total Assets | FRED (H.4.1) | `WALCL` | Weekly (Wed) | Millions USD (`usd_mn`) | Billions USD (`usd_bn`) | $\div 1,000$ |
| `tga` | Treasury General Account | FRED (Financial Stmt) | `WTREGEN` | Weekly (Wed) | Millions USD (`usd_mn`) | Billions USD (`usd_bn`) | $\div 1,000$ |
| `rrp` | Overnight Reverse Repurchase Agreements | FRED / NY Fed | `RRPONTSYD` | Daily (Business) | Billions USD (`usd_bn`) | Billions USD (`usd_bn`) | $\times 1.0$ |
| `spx` | S&P 500 Index | Yahoo Finance | `^GSPC` | Daily (Business) | Index Level | Index Level | N/A |
| `stablecoins` | Total Stablecoins Supply | DefiLlama / Aggregator | `peggedUSD` | Daily | USD (`usd`) | Billions USD (`usd_bn`) | $\div 1,000,000,000$ |

### 2.2 Unit & Frequency Alignment
1. **Unit Normalization**:
   All balance sheet series are normalized to **Billions of USD ($B)**:
   $$A_t = \frac{\text{WALCL}_t}{1000}$$
   $$T_t = \frac{\text{WTREGEN}_t}{1000}$$
   $$R_t = \text{RRPONTSYD}_t$$

2. **Frequency Alignment**:
   `WALCL` and `WTREGEN` are reported weekly on Wednesdays. `RRPONTSYD` is reported daily.
   All series are aligned to standard business calendar dates using forward-filling up to `max_ffill_days = 7` (to bridge weekend/holiday weekly release lags).

3. **Net Liquidity Formula**:
   $$\text{Net Liquidity}_t = A_t - T_t - R_t$$
   Where:
   - $A_t$: Total assets of the Federal Reserve (adds liquidity when expanded via QE/facilities).
   - $T_t$: Treasury operating cash balance at the Fed (drains commercial bank reserves when Treasury builds cash).
   - $R_t$: Cash parked by money market funds at the Fed overnight repo facility (drains active interbank liquidity).

### 2.3 Velocity & Momentum Changes
- **4-Week Change ($\sim 20$ business days)**:
  $$\Delta_{4W, t} = \text{Net Liquidity}_t - \text{Net Liquidity}_{t-20}$$
- **13-Week Change ($\sim 65$ business days / 1 quarter)**:
  $$\Delta_{13W, t} = \text{Net Liquidity}_t - \text{Net Liquidity}_{t-65}$$
- **Percentiles**:
  Empirical expanding percentiles $P_{4W, t} \in [0, 100]$ and $P_{13W, t} \in [0, 100]$ computed against expanding history strictly up to time $t$.

### 2.4 Equity Index Overlay & Correlation
- Net Liquidity and S&P 500 normalized to common date index.
- Rolling 60-day Pearson correlation:
  $$\rho_{60, t} = \text{Corr}(L_{t-59..t}, P^{\text{spx}}_{t-59..t})$$
- **Causality Disclaimer**:
  "Association, not causation: Changes in central bank reserves and repo facilities correlate with financial conditions and risk appetite across market regimes, but do not imply mechanical equity price causation."

### 2.5 Stablecoin Total Supply
- Level: $S_t$ in Billions USD.
- 30-Day Change:
  $$\Delta_{30D, t}^{\%} = \frac{S_t - S_{t-30}}{S_{t-30}} \times 100$$

---

## 3. Hand-Checked Arithmetic Verification on Three Dates

### Date 1: 2024-01-03 (Baseline)
- $A_t$ (`WALCL`): \$7,686,674 M $\rightarrow$ **\$7,686.67 B**
- $T_t$ (`WTREGEN`): \$759,045 M $\rightarrow$ **\$759.05 B**
- $R_t$ (`RRPONTSYD`): **\$701.82 B**
- Expected Net Liquidity:
  $$7,686.67 - 759.05 - 701.82 = \mathbf{\$6,225.80\text{ B}}$$

### Date 2: 2025-01-08 (Mid-Cycle Qt Drawdown)
- $A_t$ (`WALCL`): \$6,980,120 M $\rightarrow$ **\$6,980.12 B**
- $T_t$ (`WTREGEN`): \$720,500 M $\rightarrow$ **\$720.50 B**
- $R_t$ (`RRPONTSYD`): **\$185.30 B**
- Expected Net Liquidity:
  $$6,980.12 - 720.50 - 185.30 = \mathbf{\$6,074.32\text{ B}}$$

### Date 3: 2026-11-18 (Current Snapshot)
- $A_t$ (`WALCL`): \$6,741,452 M $\rightarrow$ **\$6,741.45 B**
- $T_t$ (`WTREGEN`): \$916,613 M $\rightarrow$ **\$916.61 B**
- $R_t$ (`RRPONTSYD`): **\$0.40 B**
- Expected Net Liquidity:
  $$6,741.45 - 916.61 - 0.40 = \mathbf{\$5,824.44\text{ B}}$$

---

## 4. Export Contract (`/v1/liquidity.json`)
```json
{
  "schema_version": "1.0",
  "as_of": "2026-11-20",
  "code_version": "v1.0.0",
  "params_hash": "...",
  "data": {
    "current_net_liquidity": 5824.44,
    "change_4w": -45.2,
    "change_4w_pct": 34.5,
    "change_13w": -120.4,
    "change_13w_pct": 28.2,
    "stablecoins_total_bn": 282.5,
    "stablecoins_30d_pct": 4.8,
    "dates": ["2024-01-02", ...],
    "net_liquidity": [6225.8, ...],
    "fed_assets": [7686.7, ...],
    "tga": [759.0, ...],
    "rrp": [701.8, ...],
    "spx_overlay": [4742.8, ...],
    "rolling_corr_60d": [0.45, ...],
    "audit": {
      "formula": "Net Liquidity = (WALCL / 1000) - (WTREGEN / 1000) - RRPONTSYD",
      "parameters": {"change_windows": [20, 65], "stablecoin_window": 30, "rolling_corr_window": 60},
      "series_used": ["fed_assets (WALCL)", "tga (WTREGEN)", "rrp (RRPONTSYD)", "spx (^GSPC)"],
      "n": 750,
      "n_eff": 128.4
    }
  }
}
```
