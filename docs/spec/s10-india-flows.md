# Technical Specification: s10-india-flows

## 1. Objective
Specify the data ingestion, calculation methods, schemas, and export contract for the **India Flows & Positioning Module** (`/india`) in accordance with `MARKET_RADAR_AGENT_PLAN.md` Section 8.9 and Section 13.

---

## 2. Mathematical Definition

### 2.1 FII & DII Institutional Net Flows
- **Daily Net Flow** (in INR Crores, ₹ Cr):
  - $F_t^{\text{FII}} = \text{Gross Purchases}_{\text{FII}} - \text{Gross Sales}_{\text{FII}}$
  - $F_t^{\text{DII}} = \text{Gross Purchases}_{\text{DII}} - \text{Gross Sales}_{\text{DII}}$
- **Rolling Multi-Horizon Sums**:
  - 5-Day Rolling Sum:
    $$S_{5D, t} = \sum_{k=0}^4 F_{t-k}$$
  - 20-Day Rolling Sum:
    $$S_{20D, t} = \sum_{k=0}^{19} F_{t-k}$$
- **Expanding Percentile**:
  Empirical expanding percentile $P_{20D, t} \in [0, 100]$ of the 20-day cumulative flow relative to all available historical 20-day windows strictly prior to or up to time $t$.

### 2.2 Delivery Percentage
$$\text{Delivery \%} = \frac{\text{Delivered Quantity}}{\text{Traded Quantity}} \times 100$$
- Volume-weighted aggregate for Nifty 50 constituents:
  $$\text{Agg Delivery \%}_t = \frac{\sum_{i=1}^{50} \text{Delivered Quantity}_{i, t}}{\sum_{i=1}^{50} \text{Traded Quantity}_{i, t}} \times 100$$
- Percentile rank computed against the asset's own history over a minimum of 60 trading days.

### 2.3 F&O Open Interest (OI) Buildup Classification
For any derivative instrument (index or stock future) with daily price change $\Delta P_t = P_t - P_{t-1}$ and open interest change $\Delta OI_t = OI_t - OI_{t-1}$:

| Price Change ($\Delta P$) | OI Change ($\Delta OI$) | Classification | Market Interpretation |
|---|---|---|---|
| $\Delta P > 0$ | $\Delta OI > 0$ | **Long Buildup** | New bullish positioning / capital entering long |
| $\Delta P < 0$ | $\Delta OI > 0$ | **Short Buildup** | New bearish positioning / aggressive short sellers |
| $\Delta P > 0$ | $\Delta OI < 0$ | **Short Covering** | Bearish exhaustion / shorts closing for profit/loss |
| $\Delta P < 0$ | $\Delta OI < 0$ | **Long Unwinding** | Bullish exhaustion / longs exiting positions |

### 2.4 Put-Call Ratio (PCR)
$$\text{PCR}_t = \frac{\sum \text{Open Interest}_{\text{Puts}}}{\sum \text{Open Interest}_{\text{Calls}}}$$
- Computed across near-month and total option open interest for Nifty 50.
- Normal baseline range: $[0.8, 1.2]$.
- $\text{PCR} < 0.7$: Extreme call dominance / potential oversold bounce.
- $\text{PCR} > 1.3$: Extreme put dominance / potential overbought resistance.

### 2.5 India VIX Ribbon
- Current implied volatility level $V_t$.
- Expanding historical percentiles (10th, 25th, 50th, 75th, 90th) providing stationary regime context.

---

## 3. Hand-Checked Arithmetic Verification on Three Dates

### Date 1: 2024-06-04 (Election Results Volatility Day)
- Nifty 50: Price dropped $-5.93\%$ ($\Delta P < 0$), OI increased $+8.4\%$ ($\Delta OI > 0$) $\rightarrow$ **Short Buildup**
- FII Net Cash: **$-12,436.22\text{ \u20b9 Cr}$**
- DII Net Cash: **$+3,318.98\text{ \u20b9 Cr}$**
- India VIX: Jumped from $20.9$ to **$26.75$** ($> 95\text{th percentile}$)

### Date 2: 2024-11-04 (Consolidation Pullback)
- Nifty 50: Price $-1.27\%$ ($\Delta P < 0$), OI decreased $-2.1\%$ ($\Delta OI < 0$) $\rightarrow$ **Long Unwinding**
- FII Net Cash: **$-4,329.79\text{ \u20b9 Cr}$**
- DII Net Cash: **$+2,936.08\text{ \u20b9 Cr}$**
- PCR: **$0.78$**

### Date 3: 2026-11-20 (Current Baseline)
- Nifty 50: Price $+0.42\%$ ($\Delta P > 0$), OI increased $+1.8\%$ ($\Delta OI > 0$) $\rightarrow$ **Long Buildup**
- FII Net Cash: **$+1,250.40\text{ \u20b9 Cr}$**
- DII Net Cash: **$+1,840.15\text{ \u20b9 Cr}$**
- 5D FII Sum: $1,250.40 + (-450.2) + 820.1 + (-120.0) + 650.0 = \mathbf{+2,150.3\text{ \u20b9 Cr}}$
- PCR: **$1.08$**
- India VIX: **$17.32$** ($48\text{th percentile}$)

---

## 4. Export Contract (`/v1/india_flows.json`)
```json
{
  "schema_version": "1.0",
  "as_of": "2026-11-20",
  "code_version": "v1.0.0",
  "params_hash": "...",
  "data": {
    "india_vix": 17.32,
    "india_vix_percentile": 48.0,
    "pcr": 1.08,
    "fii_net_1d": 1250.40,
    "dii_net_1d": 1840.15,
    "fii_net_5d": 2150.30,
    "dii_net_5d": 6420.50,
    "fii_net_20d": -4210.80,
    "dii_net_20d": 18950.20,
    "fii_20d_percentile": 38.5,
    "dii_20d_percentile": 82.0,
    "delivery_pct_index": 54.2,
    "delivery_pct_percentile": 62.0,
    "flow_dates": ["2026-11-01", ...],
    "fii_daily": [1250.4, ...],
    "dii_daily": [1840.15, ...],
    "fii_cum_20d": [-4210.8, ...],
    "dii_cum_20d": [18950.2, ...],
    "oi_buildup": [
      {
        "symbol": "NIFTY",
        "price_change_pct": 0.42,
        "oi_change_pct": 1.80,
        "buildup": "Long Buildup",
        "pcr": 1.08
      }
    ],
    "audit": {
      "formula": "OI Buildup = f(sign(ΔP), sign(ΔOI)), PCR = ΣPut_OI / ΣCall_OI",
      "parameters": {"rolling_flow_windows": [5, 20]},
      "series_used": ["nifty50", "indiavix", "fii_cash", "dii_cash", "nifty_fo"],
      "n": 750,
      "n_eff": 210.5
    }
  }
}
```
