# Learning Note: 2-Regime Markov-Switching Volatility Models

## 1. The Multi-Regime Nature of Financial Markets
Asset markets do not evolve under constant statistical laws. Rather, financial returns alternate between distinct unobserved "regimes":
1. **Regime 0 (Calm / Low Volatility):** Dominated by steady compounding, narrow bid-ask spreads, balanced price discovery, and low dispersion.
2. **Regime 1 (Stress / High Volatility):** Characterized by liquidity withdrawal, rapid deleveraging, high cross-asset correlations, and frequent extreme outliers.

Classical linear models (like simple rolling standard deviations) assume a single continuous state. In contrast, James Hamilton (1989) introduced **Markov-switching models**, which model the economy as jumping between discrete latent states $S_t \in \{0, 1\}$.

## 2. Model Specification
Consider a daily return process $y_t$:
$$y_t = \mu_{S_t} + \sigma_{S_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1)$$

The state $S_t$ follows a first-order Markov chain governed by the transition probability matrix $\mathbf{P}$:
$$\mathbf{P} = \begin{pmatrix} p_{00} & p_{01} \\ p_{10} & p_{11} \end{pmatrix}$$
Where:
- $p_{00} = P(S_t = 0 \mid S_{t-1} = 0)$: Probability that the calm regime persists.
- $p_{11} = P(S_t = 1 \mid S_{t-1} = 1)$: Probability that the stress regime persists.
- $p_{01} = 1 - p_{00}$: Probability of transitioning from calm into stress.
- $p_{10} = 1 - p_{11}$: Probability of transitioning from stress back into calm.

## 3. Expected Regime Durations
Because the time spent in each state follows a geometric distribution, the expected duration $E[D_i]$ (in trading days) of regime $i$ is:
$$E[D_0] = \frac{1}{1 - p_{00}} = \frac{1}{p_{01}}, \quad E[D_1] = \frac{1}{1 - p_{11}} = \frac{1}{p_{10}}$$
For instance, if $p_{00} = 0.96$, the calm regime typically persists for $\frac{1}{0.04} = 25$ trading days. If $p_{11} = 0.90$, the stress regime persists for $\frac{1}{0.10} = 10$ trading days.

## 4. Point-in-Time Discipline: Filtered vs. Smoothed Probabilities
It is vital to distinguish between two probabilities:
1. **Filtered Probability $P(S_t = j \mid Y_{1..t})$:**
   Computed using the forward Hamilton recursion using data available **up to time $t$ only**. This is strictly point-in-time and represents what an observer could know on day $t$ without look-ahead bias.
2. **Smoothed Probability $P(S_t = j \mid Y_{1..T})$:**
   Computed using Kim's backward smoothing pass over the full historical sample $T$. This incorporates subsequent information and provides an optimal historical reconstruction of past crises.

Global Market Radar reports the **filtered** probability for current risk monitoring and the **smoothed** probability for historical visual shading.
