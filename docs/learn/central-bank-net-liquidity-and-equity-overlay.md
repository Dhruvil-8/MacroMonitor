# Federal Reserve Net Liquidity & Cross-Asset Dynamics

## 1. The Central Bank Balance Sheet Identity
To understand central bank liquidity, one must examine the Federal Reserve's balance sheet structure:

$$\text{Total Assets} = \text{Total Liabilities} + \text{Capital}$$

On the liability side, the primary components are:
1. **Commercial Bank Reserves**: Funds held by commercial banks at the Federal Reserve. This represents the primary high-powered money base available to support private credit intermediation and market making.
2. **Treasury General Account (TGA)**: The operating cash balance of the US Treasury maintained directly at the Federal Reserve. When the Treasury collects taxes or issues debt, cash moves out of commercial bank reserves and into the TGA, draining private system liquidity.
3. **Overnight Reverse Repurchase Facility (RRP)**: Cash parked by non-bank financial institutions (primarily money market mutual funds) directly at the New York Fed. Cash in the RRP facility is effectively sterilized from circulating in the financial system.
4. **Currency in Circulation**: Physical Federal Reserve notes held by the public.

By rearranging the balance sheet equation, **Net Liquidity** serves as a direct proxy for active commercial bank reserve availability:

$$\text{Net Liquidity} \approx \text{Fed Total Assets} - \text{TGA} - \text{Overnight RRP}$$

---

## 2. Unit Conversions & Release Mechanics

The Federal Reserve publishes balance sheet figures across different sources and unit conventions:
- **Fed Total Assets (`WALCL`)**: Reported weekly on Thursdays as of Wednesday close on the H.4.1 release in **Millions of Dollars**. Converted to Billions:
  $$\text{Assets}_{\$B} = \frac{\text{WALCL}}{1000}$$
- **Treasury General Account (`WTREGEN`)**: Reported weekly on Thursdays as of Wednesday close in **Millions of Dollars**. Converted to Billions:
  $$\text{TGA}_{\$B} = \frac{\text{WTREGEN}}{1000}$$
- **Overnight Reverse Repo (`RRPONTSYD`)**: Reported daily by the New York Fed at 1:15 PM ET in **Billions of Dollars**.

### Handling Heterogeneous Frequencies
Because `RRPONTSYD` updates daily while `WALCL` and `WTREGEN` update weekly, naive subtraction produces jagged discontinuities. In the pipeline, we union timestamps and forward-fill weekly Wednesday figures up to 7 business days, creating a synchronized daily series.

---

## 3. Rate of Change vs Levels

Raw Net Liquidity has expanded from under \$1,000B prior to 2008 to over \$6,000B following post-pandemic quantitative easing. Because of this non-stationary structural drift, comparing raw levels across different decades is statistically misleading.

Instead, we examine rolling rates of change:
- **4-Week Change ($\Delta_{20}$)**: Captures immediate month-to-month central bank cash injections or drains.
- **13-Week Change ($\Delta_{65}$)**: Captures quarterly funding conditions (e.g., quarterly tax dates, debt ceiling resolution cycles).

By evaluating the expanding-window percentile of these changes, we contextualize whether current reserve fluctuations represent normal cyclical ebbs or extreme historical shocks.

---

## 4. Empirical Realities vs Market Mythologies

Retail financial narratives often treat Net Liquidity as a deterministic "cheat code" for equity indices ($R^2 \approx 0.8$ during quantitative easing regimes). However, rigorous institutional macro research reveals important nuances:

1. **Association, Not Mechanical Transmission**:
   Commercial bank reserves do not directly buy equities. Reserve expansion lowers short-term funding costs, compresses term premia, and encourages institutional balance sheet leverage, which secondarily elevates equity valuation multiples.
2. **Reverse Repo Exhaustion**:
   During 2023-2024, as the Fed engaged in Quantitative Tightening (shrinking `WALCL`), the Treasury's heavy bill issuance was absorbed by money market funds pulling cash out of `RRPONTSYD` (falling from \$2,500B to nearly \$0). This RRP drainage offset Fed balance sheet contraction, keeping Net Liquidity remarkably flat and insulating equities from monetary tightening.
3. **Global Multi-Central Bank Offsets**:
   Domestic US liquidity represents only one component of global liquidity. Balance sheet expansions by the Bank of Japan (BoJ) or People's Bank of China (PBoC), currency swap lines, and offshore Eurodollar credit can override domestic US contractions.
