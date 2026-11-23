# Learning Note: The Median Absolute Deviation (MAD) Jump Filter

**Topic:** Robust Outlier Rejection in Quantitative Pipelines  
**Reading Time:** 3 minutes (~500 words)

---

## 1. Intuition in Plain Language

Imagine five people in a coffee shop with bank balances:
`$5,000`, `$6,000`, `$4,500`, `$7,000`, `$5,500`.

The average is `$5,600`, and the standard deviation is about `$960`.
Now, Elon Musk walks in with `$250,000,000,000`.

If you calculate the standard deviation now, it explodes to `$102,000,000,000`.
Notice the paradox: because Musk is such an extreme outlier, he inflates the standard deviation so much that Musk himself only looks like he is about $2.4\sigma$ away from the mean!

In classical statistics, **the outlier masks itself** by blowing up the very scale ($\sigma$) used to detect it. This is why traditional mean-and-standard-deviation outlier filters fail on market data ticks. The **Median Absolute Deviation (MAD)** solves this by using the median, which is immune to extreme values.

---

## 2. The Formula

For a rolling window of prices or returns $\{x_1, \dots, x_w\}$:

1. **Calculate the rolling median:**
   $$M = \text{median}(x)$$
2. **Calculate absolute deviations from the median:**
   $$d_i = |x_i - M|$$
3. **Calculate the median of those deviations (the MAD):**
   $$\text{MAD} = \text{median}(d)$$
4. **Convert MAD to a standard deviation equivalent for a normal distribution:**
   $$\hat{\sigma}_{\text{MAD}} = 1.4826 \times \text{MAD}$$
5. **Score the candidate point:**
   $$z_{\text{MAD}} = \frac{|x_t - M|}{\hat{\sigma}_{\text{MAD}}}$$

If $z_{\text{MAD}} > 10.0$, the point is flagged as a data error and quarantined.

---

## 3. Why the Multiplier 1.4826?

For a perfectly bell-shaped standard normal distribution $\mathcal{N}(0, 1)$, the median deviation happens at the 75th percentile of the half-normal distribution:
$$\Phi^{-1}(0.75) \approx 0.67449$$
Taking the reciprocal:
$$\frac{1}{0.67449} \approx 1.4826$$
Multiplying the raw MAD by $1.4826$ makes it an unbiased estimator of standard deviation $\sigma$ when the underlying data is normally distributed.

---

## 4. A Tiny Worked Example

Suppose the last 5 prices of an index are:
`[100, 101, 102, 100, 350]` (where 350 is a bad feed tick).

1. Median $M = 101$.
2. Deviations from median:
   - $|100 - 101| = 1$
   - $|101 - 101| = 0$
   - $|102 - 101| = 1$
   - $|100 - 101| = 1$
   - $|350 - 101| = 249$
3. Sorted deviations: `[0, 1, 1, 1, 249]`.
4. Median of deviations = `1.0`.
5. Robust scale estimate $\hat{\sigma}_{\text{MAD}} = 1.4826 \times 1.0 = 1.4826$.
6. Distance of 350:
   $$\frac{|350 - 101|}{1.4826} = \frac{249}{1.4826} \approx 167.9\sigma$$

The bad tick is detected as a staggering $168\sigma$ anomaly and quarantined immediately.

---

## 5. Where It Lives in the Code

- Implementation: `apply_jump_filter` in [pipeline/src/radar/clean/cleaner.py](file:///d:/MacroMonitor/pipeline/src/radar/clean/cleaner.py).
- Tests: `test_jump_filter_rejects_extreme_outliers` in [pipeline/tests/unit/test_cleaner.py](file:///d:/MacroMonitor/pipeline/tests/unit/test_cleaner.py).
