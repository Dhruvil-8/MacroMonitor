# Learning Note: Nelson-Siegel Yield Curve Modeling

## The Diebold-Li Form
The Nelson-Siegel model fits the term structure of interest rates with three economically interpretable factors:

$$y(\tau) = L_t + S_t \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau}\right) + C_t \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}\right)$$

Where $\tau$ is maturity in months.

### Economic Interpretation of the Factors:
1. **Level ($L_t$)**: The loading is $1.0$ across all maturities. It represents the long-term asymptotic rate or macroeconomic policy anchor.
2. **Slope ($S_t$)**: The loading starts at $1.0$ at $\tau = 0$ and decays monotonically to $0$. A negative $S_t$ indicates an inverted curve.
3. **Curvature ($C_t$)**: The loading starts at $0$, peaks at medium maturities ($\tau \approx 30$ months when $\lambda = 0.0609$), and decays back toward $0$. It measures hump or belly dynamics relative to wings.

## Real Rates vs. Inflation Breakevens
Nominal 10-year yields decompose into real yield and inflation expectation:
$$y_{10Y} = r_{10Y,\text{real}} + \pi_{10Y,\text{breakeven}} + \phi_{10Y,\text{term premium}}$$

By isolating TIPS from nominal Treasuries, we determine whether rate shifts stem from real growth/tightening or changing inflation risk premia.
