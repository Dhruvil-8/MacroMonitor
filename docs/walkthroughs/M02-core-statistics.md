# Milestone Walkthrough: M2 Core Statistics

**Milestone:** M2  
**Title:** Strictly Anti-Lookahead Metrics & Effective Degrees of Freedom  
**Governing Plan:** `docs/MASTER_PLAN.md`

---

## 1. What was built and why

In finance, lookahead bias is fatal: using full-sample means or standard deviations to normalize past data leaks future information into the past. We built a library of core statistical metrics engineered with mathematical integrity:
1. **Expanding Window Normalization**: Historical percentiles and z-scores computed using only information available up to date $t$.
2. **Realized Volatility**: Annualized 20D, 60D, and 252D rolling standard deviation of log returns.
3. **Effective Sample Size ($N_{\text{eff}}$)**: Autocorrelation-adjusted sample size preventing false statistical significance.
4. **Drawdown Analytics**: Historical peak-to-trough drawdowns, durations, and recovery periods.

---

## 2. Reading order of files

1. [pipeline/src/radar/metrics/returns_vol.py](file:///d:/MacroMonitor/pipeline/src/radar/metrics/returns_vol.py): Log returns, diffs, and realized volatility.
2. [pipeline/src/radar/metrics/stats.py](file:///d:/MacroMonitor/pipeline/src/radar/metrics/stats.py): Expanding z-scores, percentiles, and $N_{\text{eff}}$.
3. [pipeline/src/radar/metrics/drawdown.py](file:///d:/MacroMonitor/pipeline/src/radar/metrics/drawdown.py): High-water marks, drawdowns, and recovery days.
4. [pipeline/tests/unit/test_returns_vol.py](file:///d:/MacroMonitor/pipeline/tests/unit/test_returns_vol.py): Known-answer and lookahead guarantee tests.
5. [pipeline/tests/unit/test_stats.py](file:///d:/MacroMonitor/pipeline/tests/unit/test_stats.py): Expanding statistics tests and $N_{\text{eff}}$ validation.

---

## 3. The 5 most important functions

1. `compute_expanding_zscore(series, min_obs=60)` in `stats.py`: Strictly anti-lookahead z-score:
   $$z_t = \frac{x_t - \mu_t}{\sigma_t}, \quad \mu_t = \text{mean}(x_{1..t}), \quad \sigma_t = \text{std}(x_{1..t})$$
2. `compute_effective_sample_size(series, nlags=1)` in `stats.py`: Adjusts $N$ for persistence using lag-1 autocorrelation $\rho_1$.
3. `compute_expanding_percentile(series, min_obs=60)` in `stats.py`: Computes historical percentile rank on $[0, 100]$ using past observations only.
4. `compute_realized_volatility(returns, window=20, annualize_factor=252)` in `returns_vol.py`: Rolling sample standard deviation scaled by $\sqrt{252}$.
5. `compute_drawdown(series)` in `drawdown.py`: Computes drawdown from the running cumulative maximum: $(P_t - \max_{s \le t} P_s) / \max_{s \le t} P_s$.

---

## 4. Key formulas in words and symbols

### Effective Sample Size
When daily returns or volatility have serial correlation $\rho_1 \ne 0$:
$$N_{\text{eff}} = N \frac{1 - \rho_1}{1 + \rho_1}$$
- If $\rho_1 = 0$ (independent): $N_{\text{eff}} = N$.
- If $\rho_1 = 0.5$ (positive persistence): $N_{\text{eff}} = N \frac{0.5}{1.5} = \frac{N}{3}$.
Each observation provides only one-third of an independent observation's information.

---

## 5. Worked numeric example: Effective Sample Size

Suppose we observe $N = 100$ daily values with lag-1 autocorrelation $\rho_1 = 0.3333$:
$$N_{\text{eff}} = 100 \times \frac{1 - 0.3333}{1 + 0.3333} = 100 \times \frac{0.6667}{1.3333} = 100 \times 0.50 = 50.0$$
Even though there are 100 rows in the database, the degrees of freedom available for hypothesis testing is only 50.

---

## 6. Three experiments for the owner

1. **Experiment 1 (Verify anti-lookahead guarantee):** Pick a historical date $t = 2020-03-16$. Compute $z_t$ using data up to that day. Now append extreme fake data for 2021-2022 to the series and recompute $z_t$. Notice that $z_t$ does not change by even $10^{-12}$.
2. **Experiment 2 (Autocorrelation in VIX vs. Equities):** Compute $\rho_1$ and $N_{\text{eff}}$ on S&P 500 returns versus VIX levels. Notice that equity returns have $\rho_1 \approx 0$ ($N_{\text{eff}} \approx N$), while VIX levels have $\rho_1 \approx 0.95$ ($N_{\text{eff}} \ll N$).
3. **Experiment 3 (Drawdown recovery):** Test `compute_drawdown` on a synthetic path that crashes 50% and recovers to a new all-time high. Verify that drawdown returns to `0.0%` exactly at the recovery point.
