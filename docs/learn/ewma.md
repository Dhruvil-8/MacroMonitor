# Learning Note: Exponentially Weighted Moving Average (EWMA) Volatility & Correlation

## Formulation
Rolling window statistics assign equal weight $\frac{1}{W}$ to all observations within a window of length $W$, and abruptly drop observations older than $W$ days (the "ghost feature" effect). In contrast, J.P. Morgan's RiskMetrics framework introduced **Exponentially Weighted Moving Average (EWMA)**, which assigns exponentially decaying weights to historical observations:

$$\sigma_{t}^2 = (1 - \lambda) \sum_{k=0}^{\infty} \lambda^k r_{t-k}^2$$

This formulation can be written recursively:
$$\sigma_{t}^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_t^2$$

where $\lambda \in (0, 1)$ is the decay factor (smoothing parameter). In RiskMetrics standard daily practice, $\lambda = 0.94$.

## Effective Half-Life
The effective half-life $H_{1/2}$ of an observation represents how many days it takes for an observation's weight to decay by 50%:
$$H_{1/2} = \frac{\ln(0.5)}{\ln(\lambda)} \approx \frac{0.693}{1 - \lambda}$$
For $\lambda = 0.94$:
$$H_{1/2} \approx \frac{0.693}{0.06} \approx 11.55 \text{ trading days}$$

## EWMA Cross-Asset Correlation
The bivariate covariance between assets $i$ and $j$ updates recursively:
$$\sigma_{ij, t} = \lambda \sigma_{ij, t-1} + (1 - \lambda) r_{i,t} r_{j,t}$$
The dynamic EWMA correlation is then:
$$\rho_{ij, t} = \frac{\sigma_{ij, t}}{\sqrt{\sigma_{i, t}^2 \sigma_{j, t}^2}}$$

## Key Advantages
1. **Instantaneous Shock Response**: Reacts immediately to sudden market jumps without waiting for an entire rolling window to absorb the move.
2. **Smooth Decay**: As shocks age, their influence smoothly fades without the artificial step changes caused by observations rolling out of a boxcar window.
3. **No Arbitrary Horizon Choice**: Eliminates the subjective choice between 20-day, 60-day, or 252-day windows by calibrating a single continuous decay factor.
