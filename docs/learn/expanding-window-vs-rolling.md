# Learning Note: Expanding Window vs. Rolling Window

**Topic:** Lookahead Bias and Historical Standardization  
**Reading Time:** 3 minutes (~450 words)

---

## 1. Intuition in Plain Language

Suppose you want to know: "How extreme was the market selloff on March 16, 2020 (the COVID shock)?"

If you calculate the average volatility of the S&P 500 across all data from 2000 through 2025, you are using data from 2021, 2022, 2023, and 2024 to evaluate a move that happened in March 2020. This is **lookahead bias**.

In real life, on March 16, 2020, an investor had no idea what the market would do in 2021. To evaluate how unusual that day was honestly, you must freeze time and use only the information available up to that exact afternoon.

---

## 2. Definitions and Difference

```text
Expanding Window (t' <= t):
Day 1:   [D1]
Day 2:   [D1, D2]
Day 100: [D1, D2, ..., D100]
Day 500: [D1, D2, ..., D100, ..., D500]   <-- Window grows; strictly anti-lookahead!

Rolling Window (fixed width W = 60):
Day 60:  [D1, ..., D60]
Day 61:       [D2, ..., D61]
Day 100:           [D41, ..., D100]       <-- Drops old history; tracks recent regime!
```

---

## 3. When to Use Which

| Metric | Window Type | Why? |
|---|---|---|
| **Historical Percentile** | Expanding | Measures: "Where does today stand relative to all past history experienced so far?" |
| **Expanding Z-Score** | Expanding | Measures: "How many standard deviations from the historical anchor observed to date?" |
| **Realized Volatility** | Rolling (20D, 60D) | Volatility is non-stationary and clusters in regimes. We want current market turbulence. |
| **Rolling Correlation** | Rolling (60D, 252D) | Relationships drift over time (e.g. stock-bond correlation shifts). |
| **Beta Estimation** | Prior Rolling (250D) | Captures current sensitivity, but estimated ending *before* the horizon to avoid lookahead. |

---

## 4. The Expanding Z-Score Formula

For any date $t$ with minimum required history $T_{\text{min}}$ (default 60 days):

$$\mu_t = \frac{1}{t} \sum_{i=1}^t x_i, \quad \sigma_t = \sqrt{\frac{1}{t-1} \sum_{i=1}^t (x_i - \mu_t)^2}$$

$$z_t = \frac{x_t - \mu_t}{\sigma_t}$$

Notice the subscript on $\mu_t$ and $\sigma_t$: they are dynamic historical anchors, not constants!

---

## 5. Where It Lives in the Code

- Expanding Z-Score: `compute_expanding_zscore` in [pipeline/src/radar/metrics/stats.py](file:///d:/MacroMonitor/pipeline/src/radar/metrics/stats.py).
- Expanding Percentile: `compute_expanding_percentile` in [pipeline/src/radar/metrics/stats.py](file:///d:/MacroMonitor/pipeline/src/radar/metrics/stats.py).
- Lookahead Verification Test: `test_expanding_zscore_no_lookahead` in [pipeline/tests/unit/test_stats.py](file:///d:/MacroMonitor/pipeline/tests/unit/test_stats.py).
