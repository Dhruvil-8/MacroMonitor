# Learning Note: Stationary Block-Bootstrap for Dependent Time Series

## Why Standard IID Bootstrap Fails on Financial Data
Efron's classical bootstrap draws observations randomly with replacement from an empirical sample. While powerful for independent and identically distributed (IID) cross-sectional data, it completely breaks down when applied to financial time series.

Financial market returns exhibit:
1. **Autocorrelation**: Consecutive returns or cumulative abnormal returns share temporal memory.
2. **Volatility Clustering**: Periods of turbulence are followed by turbulence, while calm is followed by calm (heteroskedasticity).
3. **Serial Dependence**: Drawing individual days independently destroys serial correlation structure, artificially narrowing confidence intervals and producing misleading statistical inferences.

## The Block-Bootstrap Solution
The **Block-Bootstrap** (Künsch 1989; Politis & Romano 1994) solves this problem by resampling contiguous blocks of consecutive observations rather than single data points. By keeping contiguous blocks intact, the local autocorrelation and conditional heteroskedasticity within each block are fully preserved.

### Moving Block Bootstrap (MBB)
- Divide sample of length $T$ into overlapping blocks of fixed length $b$:
  $$B_i = \{x_i, x_{i+1}, \dots, x_{i+b-1}\}$$
- Randomly sample $\lceil T / b \rceil$ blocks with replacement and concatenate them to form synthetic bootstrap paths.

### Stationary Bootstrap
- Instead of a fixed block size, block length $L$ is drawn from a geometric distribution with mean $\mathbb{E}[L] = b$:
  $$\mathbb{P}(L = k) = (1 - p)^{k-1} p, \quad p = \frac{1}{b}$$
- This guarantees that the resampled bootstrap series is strictly stationary.

## Application in Event Studies (Module P6)
In central bank policy event studies (FOMC and RBI MPC), we calculate Cumulative Abnormal Returns (CAR) across event windows $[-5, +20]$. We apply stationary block-bootstrap ($B = 1000$ iterations, block length $b = 5$) to construct robust 50% and 90% confidence bands around the CAR trajectory without imposing Gaussian distribution assumptions.
