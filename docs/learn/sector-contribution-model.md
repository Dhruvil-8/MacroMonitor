# Pedagogical Learning Note: The Approximate Sector Contribution Model

## 1. Problem Statement

Investors and macro analysts regularly ask:
> *"The S&P 500 fell 3.7% today. Which sectors drove that sell-off?"*

In a professional research context, calculating exact constituent attribution requires minute-by-minute tick data for all 503 stocks, real-time dividend adjustments, float shares, and intraday index reweighting algorithms. For an open macro research instrument, this approach is both computationally impractical and introduces proprietary corporate data boundaries that contradict our macro-first design philosophy.

## 2. Mathematical Definition

Instead, we use a transparent **Approximate Linear Contribution Model** built on published index factsheets:

For each constituent macro sector $s \in \{1, \dots, K\}$:
$$\text{contribution}_{s, h} = w_s \times r_{s, h}$$

where:
- $w_s$ is the published sector weight from the official monthly index factsheet (e.g., S&P Dow Jones Indices or NSE India).
- $r_{s, h} = \frac{P_{s, t}}{P_{s, t-h}} - 1$ is the cumulative return of sector ETF or index $s$ over horizon $h$.

## 3. The Explicit Residual Guarantee

Because index weights drift daily between monthly rebalancings (as outperforming sectors grow in relative capitalization and underperforming sectors shrink), and because indices may contain miscellaneous unclassified securities, the raw sum $\sum_{s} \text{contribution}_{s, h}$ will not equal the published index return $r_{\text{index}, h}$.

Rather than hiding this difference or normalizing it away, we define the **Residual**:
$$\text{residual}_h = r_{\text{index}, h} - \sum_{s=1}^K \text{contribution}_{s, h}$$

This guarantees the fundamental arithmetic identity:
$$\sum_{s=1}^K \text{contribution}_{s, h} + \text{residual}_h \equiv r_{\text{index}, h}$$

### Interpretation
- When the residual is small ($\le \pm 20$ bps), the published factsheet weights accurately capture the market's sectoral attribution.
- When the residual is larger, it signals significant intra-month rebalancing drift, non-sector factors, or idiosyncratic corporate actions.
- The visual waterfall explicitly includes the residual as a distinct category, ensuring complete transparency for institutional macro researchers.
