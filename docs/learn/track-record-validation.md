# Learning Note: Pre-Registered Track Records and Statistical Validation

## 1. The Perils of Hindsight Bias in Financial Analytics
In quantitative finance, the easiest thing to do is find a parameter combination that "predicted" past crashes. If you test 100 different indicator combinations on historical data, approximately 5 will appear statistically significant at the 5% level by pure chance. When financial platforms selectively display only winning indicators without pre-registration, users are deceived by narrative fallacy.

## 2. Why Base Rates Matter: The Prosecutor's Fallacy
Suppose an indicator alerts before a market selloff 70% of the time (70% hit rate). Does that mean the indicator is valuable?
Not necessarily. If the market naturally experiences the defined outcome 65% of the time on completely random, unalerted days (the unconditional base rate), then the indicator's true edge is only:
$$\text{Edge} = \text{Hit Rate} - \text{Base Rate} = 70\% - 65\% = +5\%$$
Without measuring the base rate, one cannot distinguish genuine informational edge from high baseline frequency.

## 3. The Need for Alert Declustering
Financial volatility clusters tightly in time. If a market experiences a major crisis, an alert rule might fire every single day for two weeks. If an analyst counts all 10 alerts as 10 independent successes, the sample size $N$ is artificially inflated by a factor of 10, dramatically understating standard errors.
- **Declustering Solution:** Group alerts occurring within a 10-business-day window into a single **independent episode**. The evaluation is performed once per episode from the initial trigger date.

## 4. Stationary Block Bootstrap for Dependent Time Series
Standard i.i.d. bootstrap resampling breaks the temporal dependency and autocorrelation structure of financial returns. 
To construct valid confidence intervals:
- We divide the time series into blocks of length $b = 20$ days (matching the outcome horizon).
- We resample entire blocks with replacement to preserve short-term serial autocorrelation.
- The 95% confidence interval of the difference $(\text{Hit Rate} - \text{Base Rate})$ tells us whether the observed outperformance could reasonably have arisen from sampling error.

If the 95% confidence interval includes zero, we designate the rule as **"Context Only"**—informative for qualitative awareness, but not statistically demonstrated as an independent directional edge.
