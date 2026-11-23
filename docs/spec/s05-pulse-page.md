# Specification: Pulse Page and Traceable Visual Components (s05-pulse-page)

## 1. Purpose
Defines the visual, interaction, and data-binding contract for the primary Pulse route (`/`), including signature charts (Market State Map, Volatility Landscape) and the non-negotiable Audit Drawer component.

## 2. Components
### 2.1 AuditDrawer (`AuditDrawer.tsx`)
Traceability side-drawer accessible from every chart card and metric stat block:
- **Header**: Metric display title and category.
- **Traceability Matrix**:
  - Exact mathematical formula (rendered with KaTeX / readable mathematical notation).
  - Hyperparameter values from `params.yaml` (lookback window, z-score threshold, degrees of freedom).
  - Input series identifiers, source attribution, and native measurement unit.
  - Sample sizes: raw observation count ($N$) and autocorrelation-adjusted effective sample size ($N_{\text{eff}}$).
  - Empirical percentile rank against strictly expanding history.
  - Reproducibility stamp: `as_of`, `code_version`, `params_hash`.
  - Direct deep-link to the technical specification page in `/methodology`.

### 2.2 ChartCard (`ChartCard.tsx`)
Card container ensuring layout and design token consistency:
- Eyebrow category tag (small uppercase with letter-spacing 0.08em).
- Title and descriptive neutral subtitle.
- Audit trigger button (icon).
- Chart viewport slot with error boundary and skeleton state.
- `InsightStrip` at card footer explaining current state in plain quantitative language.
- Source and date attribution footer.

### 2.3 Market State Map (`MarketStateMap.tsx`)
2D Cross-Asset Regime Visualizer:
- **X-Axis**: Signed $t$-statistic of slope of log-price over 60 days ($\pm 4$).
- **Y-Axis**: 20-Day realized volatility historical percentile ($0 \dots 100\%$).
- **Quadrants**:
  - Upper-Right / Upper-Left ($|t| \ge 2.0$, Vol $\ge 50\%$): *Violent trend*
  - Lower-Right / Lower-Left ($|t| \ge 2.0$, Vol $< 50\%$): *Orderly trend*
  - Lower-Center ($|t| < 2.0$, Vol $< 50\%$): *Quiet range*
  - Upper-Center ($|t| < 2.0$, Vol $\ge 50\%$): *Choppy*
- Soft quadrant bounding shading using `--bg-2` and `--line`.
- Hover shows asset tooltip with 1D change, volatility trend, and quadrant label.

### 2.4 Volatility Landscape (`VolatilityLandscape.tsx`)
Multi-horizon volatility matrix:
- Rows: Core universe assets grouped by asset class.
- Columns: 10-day annualized realized vol, 20-day vol, 60-day vol, Vol Trend ($vol_{20}/vol_{60} - 1$), and 20-day historical percentile.
- Percentile cells encoded with CVD-safe diverging heatmap backgrounds.
- Right-aligned tabular numbers in mono font.
