# Specification: Cross-Asset Correlation & Rotation Map (s07-cross-asset-correlation-rotation)

## 1. Purpose
Defines the cross-asset correlation matrix, temporal correlation breakdown detector, and relative rotation cycle tracker.

## 2. Correlation Evolution (Section 8.4)
- **Rolling Correlation**:
  For returns of pairs in `pipeline/config/pairs.yaml`:
  $$\rho_{60, t} = \text{Corr}(r_A, r_B \text{ over } 60\text{D}), \quad \rho_{250, t} = \text{Corr}(r_A, r_B \text{ over } 250\text{D})$$
- **Correlation Shift**:
  $$\Delta \rho_t = \rho_{60, t} - \rho_{250, t}$$
  Evaluated against historical expanding percentile. A shift $\ge 2.0\sigma$ represents a breakdown of normal multi-month co-movement.
- **Heatmap Trio**:
  Snapshots of pairwise Pearson correlation across a fixed benchmark set (e.g. SPX, NDX, Nifty 50, US 10Y, DXY, Brent, Gold, Copper) evaluated at:
  1. Current (`now`)
  2. 3 Months Ago (`3M`)
  3. 1 Year Ago (`1Y`)

## 3. Rotation Map (Section 8.6)
Cross-asset and sector momentum cycle visualizer:
- **Relative Strength (RS)**:
  $$\text{RS}_t = \frac{P_{t, \text{asset}}}{P_{t, \text{benchmark}}}$$
  Normalized against 60-day moving average:
  $$\text{RS\_Ratio}_t = 100 \times \left(1 + \frac{\text{RS}_t - \text{MA}_{60}(\text{RS})}{\text{MA}_{60}(\text{RS})}\right)$$
- **Ratio Momentum (RM)**:
  $$\text{RS\_Mom}_t = 100 \times \left(1 + \frac{\text{RS\_Ratio}_t - \text{RS\_Ratio}_{t-10}}{\text{RS\_Ratio}_{t-10}}\right)$$
- **Four Phases**:
  1. *Leading* ($\text{RS\_Ratio} \ge 100, \text{RS\_Mom} \ge 100$): Outperforming with expanding momentum.
  2. *Fading* ($\text{RS\_Ratio} \ge 100, \text{RS\_Mom} < 100$): Outperforming with decelerating momentum.
  3. *Lagging* ($\text{RS\_Ratio} < 100, \text{RS\_Mom} < 100$): Underperforming with decelerating momentum.
  4. *Recovering* ($\text{RS\_Ratio} < 100, \text{RS\_Mom} \ge 100$): Underperforming with accelerating momentum.
- **Animation**: 10-day tail showing trajectory path.
