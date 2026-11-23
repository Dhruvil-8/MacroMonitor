# Learning Note: Event Studies on Central Bank Announcements

## The Classic Event Study Methodology
Event studies measure the economic impact of an unanticipated policy announcement by examining asset prices in a narrow window around the event.

### 1. Market Model Benchmark
We isolate market-wide co-movement from idiosyncratic event impact:
$$R_{i,t} = \alpha_i + \beta_i R_{m,t} + \epsilon_{i,t}$$

- **Estimation Window**: Trailing 250 trading days ending 30 days before the decision day. The 30-day buffer ensures pre-announcement positioning does not distort beta estimation.
- **Event Window**: Typically $[-5, +10]$ trading days relative to Day 0 (the announcement).

### 2. Abnormal Returns
The abnormal return (AR) is the return unexplained by general market movements:
$$AR_{i,t} = R_{i,t} - (\hat{\alpha}_i + \hat{\beta}_i R_{m,t})$$

The Cumulative Abnormal Return (CAR) aggregates abnormal returns over time:
$$CAR_i(\tau_1, \tau_2) = \sum_{t=\tau_1}^{\tau_2} AR_{i,t}$$

### 3. Block Bootstrap for Non-Normal Financial Returns
Because daily returns exhibit fat tails, volatility clustering, and skewness, standard Gaussian confidence intervals understate uncertainty. We employ block bootstrap resampling across historical events to establish empirical 50% and 90% confidence bands without restrictive distributional assumptions.
