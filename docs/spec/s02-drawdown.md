# Specification: Drawdown (s02-drawdown)

## 1. Purpose
Measures peak-to-trough decline over trailing horizons without look-ahead bias.

## 2. Mathematical Definition
For price series $P_t$ and rolling window $w = 252$ business days:
$$\text{Peak}_t = \max_{s \in [t-w+1, t]} P_s$$
$$\text{Drawdown}_t = \frac{P_t}{\text{Peak}_t} - 1$$

## 3. Properties & Invariants
- $\text{Drawdown}_t \le 0$ for all $t$.
- $\text{Drawdown}_t = 0$ if and only if $P_t$ is at its $w$-day high.
- Units: expressed as a decimal ratio $\in [-1.0, 0.0]$, rendered as percentage in UI.
- Anti-lookahead: Trailing rolling maximum only.
