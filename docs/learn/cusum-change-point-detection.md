# CUSUM Change-Point Detection for Macro Relationships

## 1. Problem Formulation
In financial macro intelligence, a critical challenge is distinguishing between **transient volatility noise** (e.g., an idiosyncratic 2-day earnings gap between the S&P 500 and Nasdaq 100) and a **structural regime break** (e.g., a lasting shift in discount rate expectations that alters tech duration sensitivity for months).

Traditional Z-score thresholding ($|Z| \ge 2.0$) suffers from two key failure modes:
1. **False Positives**: Fat-tailed daily noise can easily breach $\pm 2\sigma$ on a single session without any persistent shift in the underlying distribution.
2. **Detection Lag**: A subtle but persistent shift (e.g., $+0.8\sigma$ drift over 20 sessions) may never breach a high threshold like $2.5\sigma$, leaving the regime break undetected until major portfolio damage occurs.

To resolve this, we employ Page's **Two-Sided Cumulative Sum (CUSUM) Control Chart** on standardized momentum innovations.

---

## 2. Mathematical Definition

Let $x_t$ be the momentum divergence spread between two pre-registered assets:
$$D_t = Z(\text{mom}_{20, A}) - Z(\text{mom}_{20, B})$$

We standardize innovations against their historical baseline:
$$z_t = \frac{D_t - \bar{D}}{\hat{\sigma}_D}$$

The two-sided CUSUM recursion maintains two separate accumulators—one tracking positive drift ($S_t^+$) and one tracking negative drift ($S_t^-$):

$$S_0^+ = 0, \quad S_t^+ = \max(0, S_{t-1}^+ + z_t - k)$$
$$S_0^- = 0, \quad S_t^- = \max(0, S_{t-1}^- - z_t - k)$$

Where:
- $k > 0$ is the **slack allowance** (reference value).
- $h > 0$ is the **decision threshold** (control limit).

### Condition for Regime Shift
A structural regime shift is registered when either accumulator breaches $h$:
- If $S_t^+ \ge h$: **Upward Regime Shift** detected on date $t$.
- If $S_t^- \ge h$: **Downward Regime Shift** detected on date $t$.

Upon registering a breach, the triggered accumulator is reset to 0:
$$S_t^+ \leftarrow 0 \quad \text{or} \quad S_t^- \leftarrow 0$$

---

## 3. Parameter Tuning & Average Run Length (ARL)

### Choice of Slack $k = 0.5$
According to statistical quality control theory (Page, 1954; Hawkins & Olwell, 1998), to optimally detect a mean shift of size $\delta$ standard deviations, the reference value should be set to:
$$k = \frac{\delta}{2}$$
Setting $k = 0.5$ optimally tunes the detector for shifts of $\delta = 1.0\sigma$. Any innovation smaller than $k = 0.5$ is absorbed as noise and steadily drives $S_t$ towards 0.

### Choice of Threshold $h = 4.0$
The decision threshold $h$ governs the trade-off between the in-control Average Run Length ($\text{ARL}_0$, average time between false alarms under pure white noise) and the out-of-control Average Run Length ($\text{ARL}_1$, average time to detect a real shift):
- Under pure Gaussian white noise $\mathcal{N}(0, 1)$ with $k = 0.5$ and $h = 4.0$, $\text{ARL}_0 \approx 168$ sessions (~8 months of daily data without a false alarm).
- When a sustained $+1.5\sigma$ mean shift occurs, $\text{ARL}_1 \approx 3.3$ sessions, guaranteeing detection within 3 to 4 business days of regime break onset.

---

## 4. Architectural Implementation

In `Global Market Radar`, CUSUM is precomputed directly in the data export pipeline (`pipeline/src/radar/metrics/divergence.py`):
```python
def compute_cusum_series(
    series: pd.Series,
    slack_k: float = 0.5,
    threshold_h: float = 4.0,
) -> dict[str, Any]:
    # Standardize
    z = (series - series.mean()) / (series.std(ddof=1) or 1.0)
    n = len(z)
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    shifts = []

    for t in range(1, n):
        s_pos[t] = max(0.0, s_pos[t - 1] + z.iloc[t] - slack_k)
        s_neg[t] = max(0.0, s_neg[t - 1] - z.iloc[t] - slack_k)

        if s_pos[t] >= threshold_h:
            shifts.append({"date": str(z.index[t].date()), "direction": "upward", "value": round(float(s_pos[t]), 2)})
            s_pos[t] = 0.0

        if s_neg[t] >= threshold_h:
            shifts.append({"date": str(z.index[t].date()), "direction": "downward", "value": round(float(s_neg[t]), 2)})
            s_neg[t] = 0.0

    return {
        "dates": [str(d.date()) for d in z.index],
        "cusum_pos": [round(float(v), 2) for v in s_pos],
        "cusum_neg": [round(float(v), 2) for v in s_neg],
        "shifts": shifts,
        "threshold": threshold_h,
    }
```

On the frontend, `CusumChangeChart.tsx` (Chart C24) renders the dual-colored trajectories:
- Emerald green for $S_t^+$ (upward decoupling)
- Rose red for $S_t^-$ (downward decoupling)
- Dashed red threshold line at $y = h$
- Interactive tooltips showing exact $S^+, S^-$ values, threshold $h$, and discrete shift event dates.
