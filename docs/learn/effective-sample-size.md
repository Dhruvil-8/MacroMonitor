# Learning Note: Effective Sample Size ($N_{\text{eff}}$)

**Topic:** Autocorrelation Adjustment in Financial Time Series  
**Reading Time:** 3 minutes (~450 words)

---

## 1. Intuition in Plain Language

When you toss a fair coin 100 times, each toss is completely independent of the previous toss. You have 100 genuinely independent pieces of information.

Financial markets are not like coin tosses. Tomorrow's volatility is heavily influenced by today's volatility. If the VIX was at 35 today, it is very likely to be close to 35 tomorrow. When consecutive observations are correlated across time (**autocorrelation**), observing 100 consecutive days of high volatility does not give you 100 independent proofs of high volatility. It might only be worth 15 or 20 independent observations.

If an analyst ignores autocorrelation and calculates standard errors using the raw sample size $N = 100$, they will drastically underestimate the true uncertainty, resulting in falsely high confidence and misleading z-scores.

---

## 2. The Formula

To correct for first-order autocorrelation ($\rho_1$), we compute the **effective sample size** $N_{\text{eff}}$:

$$N_{\text{eff}} = N \frac{1 - \rho_1}{1 + \rho_1}$$

Where:
- $N$ is the raw number of observations.
- $\rho_1 = \text{Corr}(x_t, x_{t-1})$ is the lag-1 autocorrelation coefficient.

### Boundary Behaviors
- **Zero persistence ($\rho_1 = 0$):**  
  $$N_{\text{eff}} = N \frac{1 - 0}{1 + 0} = N$$
  Observations are independent; no correction needed.
- **Positive persistence ($\rho_1 > 0$):**  
  $$N_{\text{eff}} < N$$
  Observations cluster; true degrees of freedom are lower than the row count.
- **Negative persistence ($\rho_1 < 0$):**  
  $$N_{\text{eff}} > N$$
  Mean-reverting series oscillate rapidly, effectively providing extra crossings around the mean.

---

## 3. A Tiny Worked Example

Suppose you measure a bond yield spread over $N = 250$ trading days (one year).
You compute the lag-1 autocorrelation of the spread and find $\rho_1 = 0.80$.

$$N_{\text{eff}} = 250 \times \frac{1 - 0.80}{1 + 0.80} = 250 \times \frac{0.20}{1.80} = 250 \times 0.1111 \approx 28$$

Although the database has 250 daily data points, the series only has about **28 independent pieces of statistical information**! The effective degrees of freedom for hypothesis testing is 28, not 250.

---

## 4. Common Mistakes

1. **Confusing returns with price levels:** Daily stock returns typically have very low autocorrelation ($\rho_1 \approx 0$). However, realized volatility, credit spreads, P/E ratios, and macroeconomic yields have very high autocorrelation ($\rho_1 > 0.80$). Always test the series you are actually standardizing.
2. **Dividing by zero or negative values:** If sample estimation noise yields $\rho_1 \ge 1.0$, the formula breaks down. In code, clip $\rho_1 \in [-0.99, 0.99]$ and ensure $N_{\text{eff}} \ge 1$.

---

## 5. Where It Lives in the Code

- Implementation: `compute_effective_sample_size` in [pipeline/src/radar/metrics/stats.py](file:///d:/MacroMonitor/pipeline/src/radar/metrics/stats.py).
- Tests: `test_effective_sample_size_ar1` in [pipeline/tests/unit/test_stats.py](file:///d:/MacroMonitor/pipeline/tests/unit/test_stats.py).
