# Specification: Returns and Volatility (s01-returns-vol)

## 1. Purpose
Computes stationary returns and annualized realized volatility across multiple lookback windows without look-ahead bias.

## 2. Mathematical Definition
### 2.1 Returns
- For prices, equities, commodities, FX, and indices (`transform_default == 'log_return'`):
  $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
- For yields, interest rates, and credit spreads (`transform_default == 'diff'`):
  $$r_t = (y_t - y_{t-1}) \times 100 \quad \text{(in basis points)}$$

### 2.2 Realized Volatility
For rolling windows $w \in \{10, 20, 60\}$:
$$\sigma_{w, t} = \text{std}(r_{t-w+1 \dots t}) \times \sqrt{252}$$
Where $\text{std}$ uses unbiased sample standard deviation ($N-1$ degrees of freedom, `ddof=1`).

### 2.3 Volatility Trend
$$\text{trend}_t = \frac{\sigma_{20, t}}{\sigma_{60, t}} - 1$$
- $\text{trend}_t > 0$: Short-term volatility expanding relative to intermediate baseline.
- $\text{trend}_t < 0$: Short-term volatility compressing relative to intermediate baseline.

## 3. Anti-Lookahead Guarantee
All calculations are strictly backward-looking (rolling over trailing $w$ observations up to and including $t$). Future observations $t+1 \dots$ have zero impact on $\sigma_{w, t}$.
