# Learning Note: Sovereign Yield Curve Basics & Spread Anatomy

## What is a Yield Curve?
A sovereign yield curve graphs the interest rates paid on government debt obligations of varying maturities—from short-term cash instruments (1-month, 3-month Treasury bills) to medium-term notes (2-year, 5-year, 10-year) and long-term bonds (20-year, 30-year).

Because governments with sovereign currency issuance carry virtually zero nominal default risk, sovereign yield curves represent the fundamental **risk-free benchmark** against which all private sector credit, mortgages, corporate debt, and equity discount rates are priced.

## Classic Term Structure Shapes
1. **Normal Upward-Sloping**: Long-term yields exceed short-term yields ($y_{10\text{Y}} > y_{2\text{Y}}$). Investors demand a positive term premium to compensate for the duration risk and inflation uncertainty of locking up capital over decades.
2. **Flat**: Yields across short, medium, and long maturities converge. Often observed during monetary tightening transitions when the central bank raises policy rates to match long-term equilibrium.
3. **Inverted**: Short-term yields exceed long-term yields ($y_{10\text{Y}} < y_{2\text{Y}}$ or $y_{10\text{Y}} < y_{3\text{M}}$). Historically, sustained yield curve inversions have been among the most reliable empirical leading indicators of economic recessions, reflecting expectations of future rate cuts in response to economic deceleration.

## Key Benchmark Spreads
- **$10\text{Y} - 2\text{Y}$ Spread**: The bellwether curve slope measure reflecting monetary policy expectations versus structural long-term growth.
- **$10\text{Y} - 3\text{M}$ Spread**: The short-end funding spread preferred by academic Fed researchers (e.g., Estrella & Mishkin) for recession probability modeling.
- **$30\text{Y} - 5\text{Y}$ Spread**: The long-duration curve slope reflecting fiscal supply dynamics and long-term term premium pricing.

## 10-Year Yield Anatomy
The nominal 10-year Treasury yield decomposes via the Fisher identity:
$$y_{10\text{Y}} = r_{10\text{Y},\text{TIPS}} + \pi_{10\text{Y},\text{Breakeven}}$$
- **TIPS Real Yield ($r_{\text{real}}$)**: The real rate of return above consumer inflation, reflecting economic capital productivity and monetary tightness.
- **Breakeven Inflation Rate ($\pi_{\text{breakeven}}$)**: The market's implied 10-year CPI expectation, derived directly from the yield differential between nominal Treasuries and inflation-indexed bonds.
