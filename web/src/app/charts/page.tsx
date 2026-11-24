// web/src/app/charts/page.tsx
"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { LayoutGrid, Layers, ShieldCheck, Activity, BarChart2, Compass, Cpu, ExternalLink, Search } from "lucide-react";

// Baseline Chart Components
import { MoveDecomposerWaterfall } from "@/components/charts/MoveDecomposerWaterfall";
import { WorldChoropleth } from "@/components/charts/WorldChoropleth";
import { MarketStateMap } from "@/components/charts/MarketStateMap";
import { RotationMap } from "@/components/charts/RotationMap";
import { DriverSensitivityHeatmap } from "@/components/charts/DriverSensitivityHeatmap";
import { RollingBetaChart } from "@/components/charts/RollingBetaChart";
import { DriverForestPlot } from "@/components/charts/DriverForestPlot";
import { DriverBivariateScatter } from "@/components/charts/DriverBivariateScatter";
import { VolatilityConeChart } from "@/components/charts/VolatilityConeChart";
import { DistributionHistogram } from "@/components/charts/DistributionHistogram";
import { ContextQqPlot } from "@/components/charts/ContextQqPlot";
import { TailRiskBarChart } from "@/components/charts/TailRiskBarChart";
import { AutocorrelationBarChart } from "@/components/charts/AutocorrelationBarChart";
import { UnderwaterDrawdownChart } from "@/components/charts/UnderwaterDrawdownChart";
import { PcaSynchronizationChart } from "@/components/charts/PcaSynchronizationChart";
import { DownsideCorrelationChart } from "@/components/charts/DownsideCorrelationChart";
import { HexbinJointDensityChart } from "@/components/charts/HexbinJointDensityChart";
import { RidgelineDensityChart } from "@/components/charts/RidgelineDensityChart";
import { PercentileStrip } from "@/components/charts/PercentileStrip";
import { ZScoreBandChart } from "@/components/charts/ZScoreBandChart";
import { CorrelationHeatmap } from "@/components/charts/CorrelationHeatmap";
import { DivergencePanel } from "@/components/charts/DivergencePanel";
import { CusumChangeChart } from "@/components/charts/CusumChangeChart";
import { CalendarHeatmap } from "@/components/charts/CalendarHeatmap";
import { FanChart } from "@/components/charts/FanChart";
import { NetLiquidityChart } from "@/components/charts/NetLiquidityChart";
import { SectorContributionWaterfall } from "@/components/charts/SectorContributionWaterfall";

// Push Pack & Advanced Analytical Chart Components
import { PositioningLadder } from "@/components/charts/PositioningLadder";
import { PositioningChart } from "@/components/charts/PositioningChart";
import { TermStructureChart } from "@/components/charts/TermStructureChart";
import { CurveRibbon } from "@/components/charts/CurveRibbon";
import { FactorPanel } from "@/components/charts/FactorPanel";
import { YieldHeatmap } from "@/components/charts/YieldHeatmap";
import { ReactionRibbon } from "@/components/charts/ReactionRibbon";
import { ConnectednessMatrix } from "@/components/charts/ConnectednessMatrix";
import { NetSpilloverBars } from "@/components/charts/NetSpilloverBars";
import { ChordView } from "@/components/charts/ChordView";
import { SeasonalBandChart } from "@/components/charts/SeasonalBandChart";
import { EvtRegimePanel } from "@/components/charts/EvtRegimePanel";
import { RegimeRibbon } from "@/components/charts/RegimeRibbon";
import { TrackRecordBars } from "@/components/charts/TrackRecordBars";
import { StressCrossCheckPanel } from "@/components/charts/StressCrossCheckPanel";
import { SectorHeatmap } from "@/components/charts/SectorHeatmap";
import { SectorRotationMap } from "@/components/charts/SectorRotationMap";
import { FiiDiiFlowChart } from "@/components/charts/FiiDiiFlowChart";
import { OiBuildupMap } from "@/components/charts/OiBuildupMap";
import { VolatilityLandscape } from "@/components/charts/VolatilityLandscape";
import { StressStackChart } from "@/components/charts/StressStackChart";
import { StressHistoryChart } from "@/components/charts/StressHistoryChart";
import { SourceStatusTable } from "@/components/charts/SourceStatusTable";

export type ChartCategory =
  | "attribution"
  | "drivers"
  | "rates_curves"
  | "positioning"
  | "spillover"
  | "risk"
  | "monitoring";

interface ChartMeta {
  id: string;
  code: string;
  name: string;
  category: ChartCategory;
  idea: string;
  insight: string;
  deployedPage: string;
  renderComponent: () => React.ReactNode;
}

class ChartErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("Chart preview render exception:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="h-64 rounded bg-[#0c1017] border border-amber-500/20 p-4 flex flex-col items-center justify-center text-center">
          <div className="text-amber-400 font-mono text-xs font-semibold mb-1">
            Preview Unavailable
          </div>
          <div className="text-[#8c9bb0] text-[11px] font-mono max-w-md">
            {this.state.error?.message || "Data formatting exception in chart preview."}
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function LazyChartItem({ render }: { render: () => React.ReactNode }) {
  const [isVisible, setIsVisible] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "300px" }
    );
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={containerRef} className="min-h-[260px] flex flex-col justify-center">
      {isVisible ? (
        <ChartErrorBoundary>
          {render()}
        </ChartErrorBoundary>
      ) : (
        <div className="h-64 rounded bg-[#07090d]/60 border border-[#1d2633]/40 flex flex-col items-center justify-center text-xs font-mono text-[#6f7c90] animate-pulse">
          <span>Loading chart model...</span>
        </div>
      )}
    </div>
  );
}

export default function ChartsCataloguePage() {
  const [activeCategory, setActiveCategory] = useState<"all" | ChartCategory>("all");
  const [searchQuery, setSearchQuery] = useState("");

  const catalogue: ChartMeta[] = [
    {
      id: "c01",
      code: "C01",
      name: "Waterfall with Whiskers (Move Decomposer)",
      category: "attribution",
      idea: "Econometric Decomposition & Estimation Uncertainty",
      insight: "Splits daily or multi-day moves into macro factors and an explicit residual with Newey-West HAC intervals.",
      deployedPage: "/",
      renderComponent: () => (
        <MoveDecomposerWaterfall
          decomposition={{
            target: "India (Nifty 50)",
            target_id: "featured",
            horizon_days: 5,
            total_return: 0.0261,
            contributions: {
              global_equities_row: 0.0142,
              us_10y_yield: -0.0031,
              crude_oil_brent: -0.0018,
              copper_comex: 0.0024,
              gold_comex: 0.0009,
              fx_currency: 0.0079,
            },
            intervals: {
              global_equities_row: [0.008, 0.020],
              us_10y_yield: [-0.006, -0.001],
              crude_oil_brent: [-0.004, 0.001],
              copper_comex: [0.001, 0.004],
              gold_comex: [-0.001, 0.002],
              fx_currency: [0.005, 0.011],
            },
            residual: 0.0056,
            betas: {
              global_equities_row: 0.65,
              us_10y_yield: -0.15,
              crude_oil_brent: -0.12,
              copper_comex: 0.18,
              gold_comex: 0.06,
              fx_currency: 0.98,
            },
            r2_in_sample: 0.68,
            r2_out_sample: 0.42,
            mostly_unexplained: false,
            regularized: false,
            narrative: "Nifty 50 advanced +2.61% over 5D, led by global ROW equity strength and rupee stability.",
            lens: "usd",
            n_obs: 250,
            window_dates: { prior_start: "2025-11-15", prior_end: "2026-11-15", horizon_start: "2026-11-16", horizon_end: "2026-11-24" },
          }}
        />
      ),
    },
    {
      id: "c02",
      code: "C02",
      name: "Performance Matrix with Lens Toggle",
      category: "attribution",
      idea: "Multi-Horizon Returns & FX Identity: (1+r_usd) = (1+r_loc)(1+r_fx)",
      insight: "Compares 10 global economies across Local, USD, and Currency impact lenses simultaneously.",
      deployedPage: "/countries",
      renderComponent: () => (
        <div className="p-4 rounded bg-[var(--bg-0)] border border-[var(--line)] font-mono text-xs space-y-2">
          <div className="flex justify-between border-b border-[var(--line)] pb-2 text-[var(--tx-2)]">
            <span>Economy</span>
            <span>Local Return</span>
            <span>FX Impact</span>
            <span>USD Return</span>
          </div>
          {[
            { name: "United States (S&P 500)", loc: "-3.70%", fx: "0.00%", usd: "-3.70%" },
            { name: "India (Nifty 50)", loc: "+1.12%", fx: "+0.18%", usd: "+1.30%" },
            { name: "Germany (DAX)", loc: "-2.60%", fx: "-0.02%", usd: "-2.62%" },
            { name: "Japan (Nikkei 225)", loc: "-5.06%", fx: "-0.20%", usd: "-5.25%" },
          ].map((r, i) => (
            <div key={i} className="flex justify-between text-[var(--tx-1)]">
              <span className="font-semibold text-[var(--tx-0)]">{r.name}</span>
              <span className={r.loc.startsWith("+") ? "text-[var(--pos)]" : "text-[var(--neg)]"}>{r.loc}</span>
              <span className="text-[var(--tx-2)]">{r.fx}</span>
              <span className={r.usd.startsWith("+") ? "text-[var(--pos)] font-bold" : "text-[var(--neg)] font-bold"}>{r.usd}</span>
            </div>
          ))}
        </div>
      ),
    },
    {
      id: "c03",
      code: "C03",
      name: "World Spatial Choropleth / Matrix",
      category: "drivers",
      idea: "Spatial Cross-Country Return & Realized Volatility Comparison",
      insight: "Highlights geographical hot spots and stress clustering across the 10 launch economies.",
      deployedPage: "/countries",
      renderComponent: () => <WorldChoropleth />,
    },
    {
      id: "c04",
      code: "C04",
      name: "Market State Map (Trend vs Volatility)",
      category: "attribution",
      idea: "4-Quadrant Cross-Asset Classification",
      insight: "Maps assets into Orderly Trend, Violent Trend, Quiet Range, and Choppy regimes using 60D t-stat and 20D vol %tile.",
      deployedPage: "/",
      renderComponent: () => (
        <MarketStateMap
          assets={[
            { asset: "spx", name: "S&P 500", group: "equity_index", region: "US", trend_t: 2.8, vol_percentile: 35.0, quadrant: "Orderly trend", momentum_20d: 0.042 },
            { asset: "nifty50", name: "Nifty 50", group: "india_equity", region: "IN", trend_t: 3.1, vol_percentile: 28.0, quadrant: "Orderly trend", momentum_20d: 0.038 },
            { asset: "brent", name: "Brent Oil", group: "commodity", region: "Global", trend_t: -1.2, vol_percentile: 85.0, quadrant: "Choppy", momentum_20d: -0.065 },
            { asset: "gold", name: "Gold", group: "commodity", region: "Global", trend_t: 2.4, vol_percentile: 42.0, quadrant: "Orderly trend", momentum_20d: 0.021 },
            { asset: "us10y", name: "US 10Y Yield", group: "rate", region: "US", trend_t: -2.3, vol_percentile: 78.0, quadrant: "Violent trend", momentum_20d: -0.015 },
          ]}
          onSelectAsset={() => { }}
        />
      ),
    },
    {
      id: "c05",
      code: "C05",
      name: "Rotation Map with 10D Momentum Tails",
      category: "attribution",
      idea: "Relative Strength vs Momentum Trajectory",
      insight: "Tracks leading, weakening, lagging, and improving sector or asset trajectories over 10 trading sessions.",
      deployedPage: "/sectors",
      renderComponent: () => (
        <RotationMap
          benchmark="SPY"
          assets={[
            { asset: "tech", name: "Technology", group: "sector", rs_ratio: 104.5, rs_mom: 101.2, phase: "Leading", tail_dates: ["2026-11-10", "2026-11-15", "2026-11-24"], tail_x: [102.1, 103.4, 104.5], tail_y: [100.2, 100.8, 101.2] },
            { asset: "fin", name: "Financials", group: "sector", rs_ratio: 101.2, rs_mom: 98.4, phase: "Fading", tail_dates: ["2026-11-10", "2026-11-15", "2026-11-24"], tail_x: [102.5, 101.8, 101.2], tail_y: [99.5, 98.9, 98.4] },
            { asset: "energy", name: "Energy", group: "sector", rs_ratio: 96.5, rs_mom: 97.2, phase: "Lagging", tail_dates: ["2026-11-10", "2026-11-15", "2026-11-24"], tail_x: [97.8, 97.0, 96.5], tail_y: [98.0, 97.5, 97.2] },
            { asset: "util", name: "Utilities", group: "sector", rs_ratio: 98.2, rs_mom: 102.5, phase: "Recovering", tail_dates: ["2026-11-10", "2026-11-15", "2026-11-24"], tail_x: [96.5, 97.3, 98.2], tail_y: [101.0, 101.8, 102.5] },
          ]}
        />
      ),
    },
    {
      id: "c06",
      code: "C06",
      name: "Sensitivity Heatmap with Significance Hatching",
      category: "drivers",
      idea: "Multi-Factor Rolling Betas with HAC Errors",
      insight: "Color-coded exposure of tracked assets to Oil, Dollar, Yields, and ROW equity with stability marks.",
      deployedPage: "/drivers",
      renderComponent: () => (
        <DriverSensitivityHeatmap
          rows={[
            {
              target_id: "spx",
              target_name: "S&P 500",
              target_code: "SPX",
              window: 250,
              r2: 0.65,
              regularized: false,
              n_obs: 250,
              drivers: {
                row_equity: { driver_id: "row_equity", beta: 0.85, ci_lower: 0.75, ci_upper: 0.95, t_stat: 17.0, p_value: 0.0001, significant: true, stability: 0.88, prior: "positive", prior_aligned: true, history: [] },
                brent: { driver_id: "brent", beta: 0.12, ci_lower: 0.04, ci_upper: 0.20, t_stat: 3.0, p_value: 0.003, significant: true, stability: 0.75, prior: "positive", prior_aligned: true, history: [] },
                copper: { driver_id: "copper", beta: 0.22, ci_lower: 0.12, ci_upper: 0.32, t_stat: 4.4, p_value: 0.001, significant: true, stability: 0.82, prior: "positive", prior_aligned: true, history: [] },
                gold: { driver_id: "gold", beta: -0.05, ci_lower: -0.13, ci_upper: 0.03, t_stat: -1.25, p_value: 0.21, significant: false, stability: 0.65, prior: "positive", prior_aligned: true, history: [] },
                dxy: { driver_id: "dxy", beta: -0.45, ci_lower: -0.59, ci_upper: -0.31, t_stat: -6.4, p_value: 0.0001, significant: true, stability: 0.91, prior: "negative", prior_aligned: true, history: [] },
                us10y: { driver_id: "us10y", beta: -0.25, ci_lower: -0.37, ci_upper: -0.13, t_stat: -4.1, p_value: 0.0001, significant: true, stability: 0.84, prior: "negative", prior_aligned: true, history: [] },
              },
            },
          ]}
          selectedTargetId="spx"
          onSelectTarget={() => { }}
          selectedDriverId="row_equity"
          onSelectDriver={() => { }}
        />
      ),
    },
    {
      id: "c07",
      code: "C07",
      name: "Rolling Beta with Confidence Band",
      category: "drivers",
      idea: "Estimation Uncertainty & Structural Break Detection",
      insight: "Shows 250D rolling beta alongside &plusmn;1.96 SE confidence envelope to distinguish drift from noise.",
      deployedPage: "/drivers",
      renderComponent: () => (
        <RollingBetaChart
          driverItem={{
            driver_id: "row_equity",
            beta: 0.65,
            ci_lower: 0.53,
            ci_upper: 0.77,
            t_stat: 10.8,
            p_value: 0.0001,
            significant: true,
            stability: 0.85,
            prior: "positive",
            prior_aligned: true,
            history: [
              { date: "2026-05", beta: 0.58, ci_lower: 0.44, ci_upper: 0.72 },
              { date: "2026-06", beta: 0.62, ci_lower: 0.48, ci_upper: 0.76 },
              { date: "2026-07", beta: 0.60, ci_lower: 0.46, ci_upper: 0.74 },
              { date: "2026-08", beta: 0.65, ci_lower: 0.51, ci_upper: 0.79 },
              { date: "2026-09", beta: 0.68, ci_lower: 0.54, ci_upper: 0.82 },
              { date: "2026-10", beta: 0.64, ci_lower: 0.50, ci_upper: 0.78 },
              { date: "2026-11", beta: 0.67, ci_lower: 0.53, ci_upper: 0.81 },
            ],
          }}
          targetName="Nifty 50"
          driverName="Global Equities (ROW)"
        />
      ),
    },
    {
      id: "c08",
      code: "C08",
      name: "Forest Plot (Driver Sensitivities)",
      category: "drivers",
      idea: "Simultaneous Multi-Driver Confidence Intervals",
      insight: "Displays point estimates and 95% HAC confidence intervals across all candidate drivers on one axis.",
      deployedPage: "/drivers",
      renderComponent: () => (
        <DriverForestPlot
          targetRow={{
            target_id: "spx",
            target_name: "S&P 500",
            target_code: "SPX",
            window: 250,
            r2: 0.65,
            regularized: false,
            n_obs: 250,
            drivers: {
              row_equity: { driver_id: "row_equity", beta: 0.85, ci_lower: 0.75, ci_upper: 0.95, t_stat: 17.0, p_value: 0.0001, significant: true, stability: 0.88, prior: "positive", prior_aligned: true, history: [] },
              us10y: { driver_id: "us10y", beta: -0.28, ci_lower: -0.40, ci_upper: -0.16, t_stat: -4.6, p_value: 0.0001, significant: true, stability: 0.84, prior: "negative", prior_aligned: true, history: [] },
              dxy: { driver_id: "dxy", beta: -0.42, ci_lower: -0.58, ci_upper: -0.26, t_stat: -5.25, p_value: 0.0001, significant: true, stability: 0.90, prior: "negative", prior_aligned: true, history: [] },
              brent: { driver_id: "brent", beta: 0.14, ci_lower: 0.04, ci_upper: 0.24, t_stat: 2.8, p_value: 0.005, significant: true, stability: 0.75, prior: "positive", prior_aligned: true, history: [] },
            },
          }}
          selectedDriverId="row_equity"
          onSelectDriver={() => { }}
        />
      ),
    },
    {
      id: "c09",
      code: "C09",
      name: "Bivariate Scatter with OLS & Confidence Bands",
      category: "drivers",
      idea: "Single-Factor Scatter & Residual Strip",
      insight: "Shows raw scatter points, linear regression slope line, and residual dispersion strip.",
      deployedPage: "/drivers",
      renderComponent: () => (
        <DriverBivariateScatter
          scatterData={{
            scatter: [
              [0.77, 0.19], [-1.99, -0.35], [-0.88, -0.09], [0.44, 0.67], [-1.51, 0.21],
              [-0.26, -1.07], [-0.92, -0.16], [-2.75, -0.24], [1.28, -1.17], [0.23, 0.79],
              [1.08, 0.88], [2.61, 0.64], [0.50, 0.45], [-0.23, 0.32], [0.0, 0.0],
              [-2.60, -0.03], [2.12, -0.35], [-0.03, -0.14], [-1.74, -0.74], [1.0, 1.0],
              [-1.23, -0.35], [3.33, 0.01], [2.15, 0.65], [0.83, 0.16], [2.47, -0.19],
              [1.59, -0.54], [-4.24, 0.26], [-0.55, 0.12], [0.89, -0.42], [1.14, 0.38]
            ],
            n_obs: 250,
            line: [[-4.0, -1.2], [4.0, 1.4]],
            r2: 0.62,
            beta: 0.34,
            t_stat: 4.8,
            band_upper: [[-4.0, -0.5], [4.0, 2.1]],
            band_lower: [[-4.0, -1.9], [4.0, 0.7]],
          }}
          targetName="S&P 500"
          driverName="Brent Crude Oil"
        />
      ),
    },
    {
      id: "c10",
      code: "C10",
      name: "Volatility Cone (Multi-Horizon Distribution)",
      category: "risk",
      idea: "Horizon-Dependent Volatility Term Structure",
      insight: "Compares current realized volatility across 10D to 250D against expanding historical percentiles (10th to 90th).",
      deployedPage: "/risk",
      renderComponent: () => (
        <VolatilityConeChart
          assetName="S&P 500"
          coneData={[
            { window: 10, current: 0.182, p10: 0.095, p25: 0.112, p50: 0.141, p75: 0.180, p90: 0.245, min: 0.08, max: 0.35, percentile: 72 },
            { window: 20, current: 0.156, p10: 0.101, p25: 0.120, p50: 0.148, p75: 0.185, p90: 0.238, min: 0.09, max: 0.32, percentile: 58 },
            { window: 60, current: 0.141, p10: 0.110, p25: 0.128, p50: 0.152, p75: 0.189, p90: 0.230, min: 0.10, max: 0.30, percentile: 46 },
            { window: 120, current: 0.138, p10: 0.115, p25: 0.132, p50: 0.156, p75: 0.191, p90: 0.225, min: 0.11, max: 0.28, percentile: 41 },
            { window: 250, current: 0.144, p10: 0.120, p25: 0.138, p50: 0.160, p75: 0.195, p90: 0.221, min: 0.115, max: 0.26, percentile: 43 },
          ]}
        />
      ),
    },
    {
      id: "c11",
      code: "C11",
      name: "Distribution Histogram with KDE & Normal Overlay",
      category: "risk",
      idea: "Empirical Density vs Gaussian Bell Curve (Vol Skew & Kurtosis)",
      insight: "Reveals fat tails, negative return skew, and excess kurtosis over Gaussian benchmark.",
      deployedPage: "/risk",
      renderComponent: () => (
        <DistributionHistogram
          bins={[-4.06, -3.72, -3.38, -3.04, -2.71, -2.37, -2.03, -1.69, -1.35, -1.02, -0.68, -0.34, 0.0, 0.34, 0.67, 1.01, 1.35, 1.69, 2.02, 2.36, 2.70, 3.04, 3.38, 3.71, 4.05]}
          counts={[1, 2, 4, 8, 15, 32, 65, 110, 142, 125, 95, 62, 41, 26, 14, 6, 3, 1, 1, 0, 0, 0, 0, 0, 1]}
          kdeGrid={[-0.76, -0.63, -0.50, -0.37, -0.25, -0.12, 0.0, 0.12, 0.25, 0.37, 0.50, 0.63, 0.76]}
          kdeDensity={[0.01, 0.05, 0.18, 0.52, 1.15, 1.72, 1.83, 1.72, 1.15, 0.52, 0.18, 0.05, 0.01]}
          normalDensity={[0.01, 0.04, 0.15, 0.45, 1.05, 1.68, 1.83, 1.68, 1.05, 0.45, 0.15, 0.04, 0.01]}
          currentValue={6910.39}
          percentile={75.5}
          unit="index"
        />
      ),
    },
    {
      id: "c12",
      code: "C12",
      name: "Quantile-Quantile (QQ) Normality Plot",
      category: "risk",
      idea: "Direct Tail Departure Diagnostic",
      insight: "Plots empirical quantiles against theoretical normal quantiles; S-shape reveals heavy fat tails.",
      deployedPage: "/risk",
      renderComponent: () => (
        <ContextQqPlot
          theoretical={[-2.33, -1.88, -1.65, -1.48, -1.34, -1.23, -1.13, -1.04, -0.95, -0.88, -0.81, -0.74, 0.0, 0.74, 0.81, 0.88, 0.95, 1.04, 1.13, 1.23, 1.34, 1.48, 1.65, 1.88, 2.33]}
          empirical={[-3.85, -2.52, -1.95, -1.62, -1.41, -1.25, -1.12, -1.01, -0.91, -0.82, -0.74, -0.66, 0.02, 0.68, 0.76, 0.85, 0.93, 1.03, 1.14, 1.28, 1.46, 1.68, 2.05, 2.65, 4.12]}
          assetName="S&P 500 Index"
        />
      ),
    },
    {
      id: "c13",
      code: "C13",
      name: "Tail-Risk Bar Chart (VaR & Expected Shortfall)",
      category: "risk",
      idea: "Historical vs Gaussian Tail Losses",
      insight: "Exposes shortfall gap: why real-world tail crashes exceed standard normal predictions.",
      deployedPage: "/risk",
      renderComponent: () => (
        <TailRiskBarChart
          assetName="S&P 500"
          tailMetrics={[
            { confidence: 0.95, hist_var: -0.0185, gauss_var: -0.0164, hist_es: -0.0265, gauss_es: -0.0206, tail_ratio: 1.45 },
            { confidence: 0.99, hist_var: -0.0320, gauss_var: -0.0233, hist_es: -0.0440, gauss_es: -0.0267, tail_ratio: 1.65 },
          ]}
        />
      ),
    },
    {
      id: "c14",
      code: "C14",
      name: "Autocorrelation Bars (Returns & Squared Volatility)",
      category: "risk",
      idea: "Volatility Clustering & Serial Correlation",
      insight: "Lags 1-20 autocorrelation reveals volatility memory and clustering while returns exhibit zero persistence.",
      deployedPage: "/risk",
      renderComponent: () => (
        <AutocorrelationBarChart
          assetName="S&P 500"
          acfData={{
            lags: [1, 2, 3, 4, 5],
            acf_returns: [-0.04, 0.01, 0.02, -0.01, 0.03],
            acf_squared: [0.28, 0.22, 0.19, 0.16, 0.14],
            band: 0.08,
            n_eff: 250,
          }}
        />
      ),
    },
    {
      id: "c15",
      code: "C15",
      name: "Underwater Drawdown Chart & Crisis Timeline",
      category: "risk",
      idea: "Historical Peak-to-Trough Drawdown Dynamics",
      insight: "Measures severity, duration, and recovery trajectories across historical crisis episodes.",
      deployedPage: "/risk",
      renderComponent: () => (
        <UnderwaterDrawdownChart
          assetName="S&P 500"
          series={[
            { date: "2026-06", drawdown: 0.0 },
            { date: "2026-07", drawdown: -0.04 },
            { date: "2026-08", drawdown: -0.08 },
            { date: "2026-09", drawdown: -0.03 },
            { date: "2026-10", drawdown: -0.01 },
            { date: "2026-11", drawdown: -0.098 },
          ]}
          episodes={[
            { status: "active", peak_date: "2026-10-18", trough_date: "2026-11-24", depth: -0.098, duration_days: 34, recovery_date: null, recovery_days: null },
          ]}
        />
      ),
    },
    {
      id: "c16",
      code: "C16",
      name: "PCA Synchronization Line & Scree Plot",
      category: "risk",
      idea: "Cross-Market Co-Movement & Systemic Factor Share",
      insight: "PC1 explained-variance share detects when global markets are moving together abnormally, with US sign alignment (v_{1,US} > 0).",
      deployedPage: "/risk",
      renderComponent: () => (
        <PcaSynchronizationChart
          pcaData={{
            scree: [64.0, 15.0, 9.4, 5.6, 3.5, 1.5, 0.6, 0.4],
            loadings_pc1: { spx: 0.42, dax: 0.38, ftse: 0.35, n225: 0.31, nifty50: 0.28 },
            loadings_pc2: { spx: -0.15, dax: -0.12, ftse: 0.05, n225: 0.45, nifty50: 0.52 },
            pc1_history: [
              { date: "2026-06", pc1_share: 44.0 },
              { date: "2026-07", pc1_share: 46.0 },
              { date: "2026-08", pc1_share: 52.0 },
              { date: "2026-09", pc1_share: 49.0 },
              { date: "2026-10", pc1_share: 58.0 },
              { date: "2026-11", pc1_share: 64.0 },
            ],
          }}
          latestShare={0.64}
        />
      ),
    },
    {
      id: "c17",
      code: "C17",
      name: "Downside vs All-Day Correlation (Stress Premium)",
      category: "risk",
      idea: "Asymmetric Tail Correlation & Diversification Failure",
      insight: "Evaluates correlation conditional on benchmark down-days to determine whether hedges breakdown under stress.",
      deployedPage: "/risk",
      renderComponent: () => (
        <DownsideCorrelationChart
          items={[
            { target_id: "dax", target_name: "DAX (Germany)", all_day_corr: 0.72, downside_corr: 0.88, stress_premium: 0.16, stress_threshold: -0.01, n_all: 250, n_stress: 32 },
            { target_id: "nifty50", target_name: "Nifty 50 (India)", all_day_corr: 0.45, downside_corr: 0.58, stress_premium: 0.13, stress_threshold: -0.01, n_all: 250, n_stress: 32 },
            { target_id: "gold", target_name: "Gold COMEX", all_day_corr: 0.05, downside_corr: -0.15, stress_premium: -0.20, stress_threshold: -0.01, n_all: 250, n_stress: 32 },
            { target_id: "brent", target_name: "Brent Crude", all_day_corr: 0.35, downside_corr: 0.52, stress_premium: 0.17, stress_threshold: -0.01, n_all: 250, n_stress: 32 },
          ]}
        />
      ),
    },
    {
      id: "c18",
      code: "C18",
      name: "Hexbin Joint Tail Density",
      category: "drivers",
      idea: "Non-Linear Joint Distribution & Bivariate Tail Clustering",
      insight: "Precomputed bivariate hexagonal binning showing joint tail co-dependence without assuming linearity.",
      deployedPage: "/risk",
      renderComponent: () => <HexbinJointDensityChart />,
    },
    {
      id: "c19",
      code: "C19",
      name: "Ridgeline Return Distributions",
      category: "risk",
      idea: "Cross-Market Distributional Shape Comparison",
      insight: "Overlapping KDE curves across 8 key asset classes highlighting asymmetry and current position marker.",
      deployedPage: "/risk",
      renderComponent: () => <RidgelineDensityChart />,
    },
    {
      id: "c20",
      code: "C20",
      name: "Percentile Strip with 'You Are Here' Marker",
      category: "monitoring",
      idea: "Immediate Contextual Historical Percentile Ranking",
      insight: "Gradient ribbon mapping current asset readings to expanding empirical history [0% to 100%].",
      deployedPage: "/",
      renderComponent: () => (
        <div className="space-y-3">
          <PercentileStrip label="S&P 500 Index" percentile={93.7} currentValue="6,910.39" />
          <PercentileStrip label="US 10-Year Treasury Yield" percentile={55.7} currentValue="5.04%" />
          <PercentileStrip label="Brent Crude Oil Futures" percentile={85.3} currentValue="$92.57/bbl" />
          <PercentileStrip label="Gold Futures" percentile={85.4} currentValue="$3,515.40/oz" />
        </div>
      ),
    },
    {
      id: "c21",
      code: "C21",
      name: "Standardized Z-Score Band Chart",
      category: "monitoring",
      idea: "Outlier Move Detection Against Normal Standard Deviations",
      insight: "Plots normalized innovations against &plusmn;2&sigma; and &plusmn;3&sigma; thresholds to isolate regime spikes.",
      deployedPage: "/radar",
      renderComponent: () => <ZScoreBandChart />,
    },
    {
      id: "c22",
      code: "C22",
      name: "Hierarchical Correlation Heatmap",
      category: "attribution",
      idea: "Cross-Asset Clustered Co-Movement Matrix",
      insight: "Pairwise correlation matrix ordered hierarchically to group co-moving market clusters.",
      deployedPage: "/cross-asset",
      renderComponent: () => (
        <CorrelationHeatmap
          benchmarkAssets={["SPX", "NDX", "DAX", "N225", "NIFTY50", "GOLD", "BRENT"]}
          heatmapNow={{
            SPX: { SPX: 1.0, NDX: 0.92, DAX: 0.74, N225: 0.55, NIFTY50: 0.46, GOLD: 0.05, BRENT: 0.32 },
            NDX: { SPX: 0.92, NDX: 1.0, DAX: 0.71, N225: 0.52, NIFTY50: 0.44, GOLD: 0.02, BRENT: 0.28 },
            DAX: { SPX: 0.74, NDX: 0.71, DAX: 1.0, N225: 0.48, NIFTY50: 0.41, GOLD: 0.08, BRENT: 0.35 },
            N225: { SPX: 0.55, NDX: 0.52, DAX: 0.48, N225: 1.0, NIFTY50: 0.35, GOLD: 0.01, BRENT: 0.22 },
            NIFTY50: { SPX: 0.46, NDX: 0.44, DAX: 0.41, N225: 0.35, NIFTY50: 1.0, GOLD: 0.18, BRENT: -0.15 },
            GOLD: { SPX: 0.05, NDX: 0.02, DAX: 0.08, N225: 0.01, NIFTY50: 0.18, GOLD: 1.0, BRENT: 0.24 },
            BRENT: { SPX: 0.32, NDX: 0.28, DAX: 0.35, N225: 0.22, NIFTY50: -0.15, GOLD: 0.24, BRENT: 1.0 },
          }}
          heatmap3m={{}}
          heatmap1y={{}}
        />
      ),
    },
    {
      id: "c23",
      code: "C23",
      name: "Divergence Panel (Normalized Spread & Z-Difference)",
      category: "monitoring",
      idea: "Statistical Disagreement Between Coupled Series",
      insight: "Dual normalized lines with standardized spread highlighting multi-sigma divergence.",
      deployedPage: "/radar",
      renderComponent: () => (
        <DivergencePanel
          divergence={{
            id: "spx_us10y",
            percentile: 98.2,
            active: true,
            asset_a: "spx",
            asset_b: "us10y",
            rationale: "Equity markets and 10-year yields have decoupled by 2.85 sigma.",
            spread_z: 2.85,
            consecutive_days: 12,
            history_dates: ["2026-10-01", "2026-10-15", "2026-11-01", "2026-11-24"],
            history_spread: [0.0, 1.2, 2.1, 2.85],
          }}
        />
      ),
    },
    {
      id: "c24",
      code: "C24",
      name: "CUSUM Change-Point Detector (Page's Two-Sided)",
      category: "monitoring",
      idea: "Persistent Regime Shift vs White Noise Filtering",
      insight: "Page's Cumulative Sum on standardized innovations with slack k=0.5 and limit h=4.0 to catch persistent drift.",
      deployedPage: "/radar",
      renderComponent: () => (
        <CusumChangeChart
          cusum={{
            dates: [
              "2026-08-31", "2026-09-08", "2026-09-15", "2026-09-22", "2026-09-29",
              "2026-10-06", "2026-10-13", "2026-10-20", "2026-10-27", "2026-11-03",
              "2026-11-10", "2026-11-17", "2026-11-24"
            ],
            cusum_pos: [0.0, 0.42, 0.85, 1.45, 1.92, 2.50, 3.12, 3.85, 4.40, 4.25, 3.90, 4.15, 4.40],
            cusum_neg: [0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20, 0.10, 0.0],
            threshold: 4.0,
            shifts: [
              { date: "2026-10-27", direction: "up", value: 4.40 },
              { date: "2026-11-24", direction: "up", value: 4.40 }
            ],
          }}
          pairName="SPX vs NDX (Tech Premium Innovation Drift)"
        />
      ),
    },
    {
      id: "c25",
      code: "C25",
      name: "Calendar Heatmap (Monthly Returns)",
      category: "monitoring",
      idea: "Empirical Monthly Seasonality with Chance Caveat",
      insight: "Month-by-year historical returns matrix accompanied by statutory caveat that historical patterns may reflect random chance.",
      deployedPage: "/context/spx",
      renderComponent: () => (
        <CalendarHeatmap
          data={[
            { year: 2024, month: 1, return: 0.016 },
            { year: 2024, month: 2, return: 0.052 },
            { year: 2024, month: 3, return: 0.031 },
            { year: 2024, month: 4, return: -0.041 },
            { year: 2024, month: 5, return: 0.048 },
            { year: 2024, month: 6, return: 0.035 },
            { year: 2025, month: 1, return: 0.027 },
            { year: 2025, month: 2, return: -0.018 },
            { year: 2025, month: 3, return: 0.042 },
            { year: 2026, month: 9, return: 0.021 },
            { year: 2026, month: 10, return: -0.009 },
            { year: 2026, month: 11, return: -0.037 },
          ]}
          assetName="Nifty 50"
        />
      ),
    },
    {
      id: "c26",
      code: "C26",
      name: "Quantile Fan Chart (Forward Trajectories)",
      category: "monitoring",
      idea: "Probabilistic Outcome Ranges Following Macro States",
      insight: "Quantile dispersion cone (10th, 25th, 50th, 75th, 90th) illustrating variance expansion over future horizons.",
      deployedPage: "/analogues",
      renderComponent: () => <FanChart />,
    },
    {
      id: "c27",
      code: "C27",
      name: "Stacked Area + Line (Net Liquidity Overlay)",
      category: "attribution",
      idea: "Balance Sheet Component Decomposition & Dual-Axis Overlay",
      insight: "Federal Reserve Net Liquidity = WALCL − WTREGEN − RRPONTSYD overlaid with S&P 500.",
      deployedPage: "/liquidity",
      renderComponent: () => (
        <NetLiquidityChart
          dates={["2026-08", "2026-09", "2026-10", "2026-11"]}
          netLiquidity={[5750, 5720, 5720, 5710]}
          fedAssets={[6850, 6820, 6790, 6740]}
          tga={[780, 810, 830, 850]}
          rrp={[320, 290, 240, 180]}
          spxOverlay={[5600, 5750, 5860, 5910]}
          rollingCorr60d={[0.52, 0.49, 0.46, 0.48]}
        />
      ),
    },
    {
      id: "c28",
      code: "C28",
      name: "Futures Positioning Ladder (CFTC Speculative Crowding)",
      category: "positioning",
      idea: "156-Week Expanding Percentile Positioning across 16 Futures Contracts",
      insight: "Ranks speculative positioning from net short to net long across equities, rates, commodities, and currencies.",
      deployedPage: "/positioning",
      renderComponent: () => (
        <PositioningLadder
          contracts={[
            {
              id: "sp500_emini",
              name: "E-mini S&P 500",
              cftc_code: "13874A",
              asset_class: "equities",
              n: 312,
              n_eff: 10,
              latest_date: "2026-11-17",
              open_interest: 3244953,
              net_speculative: -68846,
              net_pct_oi: -0.0212,
              chg_1w_pct_oi: 0.0145,
              chg_4w_pct_oi: 0.0196,
              chg_13w_pct_oi: -0.0211,
              percentile: 55.0,
              percentile_4w_ago: 37.5,
              percentile_13w_ago: 78.2,
              series: [],
            },
            {
              id: "us_10y_note",
              name: "10-Year U.S. Treasury Notes",
              cftc_code: "043602",
              asset_class: "rates",
              n: 312,
              n_eff: 10,
              latest_date: "2026-11-17",
              open_interest: 4892100,
              net_speculative: -745210,
              net_pct_oi: -0.1523,
              chg_1w_pct_oi: -0.0112,
              chg_4w_pct_oi: -0.0345,
              chg_13w_pct_oi: -0.0892,
              percentile: 8.5,
              percentile_4w_ago: 12.0,
              percentile_13w_ago: 22.4,
              is_extreme: true,
              extreme_type: "short",
              series: [],
            },
            {
              id: "gold",
              name: "Gold (COMEX)",
              cftc_code: "088691",
              asset_class: "commodities",
              n: 312,
              n_eff: 10,
              latest_date: "2026-11-17",
              open_interest: 512400,
              net_speculative: 245100,
              net_pct_oi: 0.4783,
              chg_1w_pct_oi: 0.0234,
              chg_4w_pct_oi: 0.0512,
              chg_13w_pct_oi: 0.1245,
              percentile: 94.2,
              percentile_4w_ago: 88.5,
              percentile_13w_ago: 82.0,
              is_extreme: true,
              extreme_type: "long",
              series: [],
            },
            {
              id: "eur_usd",
              name: "Euro FX (CME)",
              cftc_code: "099741",
              asset_class: "fx",
              n: 312,
              n_eff: 10,
              latest_date: "2026-11-17",
              open_interest: 720300,
              net_speculative: -42100,
              net_pct_oi: -0.0584,
              chg_1w_pct_oi: 0.0089,
              chg_4w_pct_oi: -0.0156,
              chg_13w_pct_oi: -0.0421,
              percentile: 42.0,
              percentile_4w_ago: 45.2,
              percentile_13w_ago: 51.0,
              series: [],
            },
          ]}
        />
      ),
    },
    {
      id: "c29",
      code: "C29",
      name: "Speculative Positioning Time Series & Open Interest",
      category: "positioning",
      idea: "Historical Speculative Net % of Open Interest with Expanding Percentile Bands",
      insight: "Tracks non-commercial position dynamics against total open interest over multi-year cycles.",
      deployedPage: "/positioning",
      renderComponent: () => (
        <PositioningChart
          data={{
            schema_version: "1.0",
            as_of: "2026-11-24",
            code_version: "v1.0.0",
            contracts: [
              {
                id: "sp500_emini",
                name: "E-mini S&P 500",
                cftc_code: "13874A",
                asset_class: "equities",
                n: 312,
                n_eff: 10,
                latest_date: "2026-11-17",
                open_interest: 3244953,
                net_speculative: -68846,
                net_pct_oi: -0.0212,
                percentile: 55.0,
                series: [
                  { date: "2026-09-01", open_interest: 3100000, noncomm_long: 420000, noncomm_short: 480000, net_speculative: -60000, net_pct_oi: -0.0194, percentile: 52.0 },
                  { date: "2026-10-01", open_interest: 3180000, noncomm_long: 410000, noncomm_short: 510000, net_speculative: -100000, net_pct_oi: -0.0314, percentile: 41.0 },
                  { date: "2026-11-01", open_interest: 3220000, noncomm_long: 430000, noncomm_short: 495000, net_speculative: -65000, net_pct_oi: -0.0202, percentile: 53.5 },
                  { date: "2026-11-17", open_interest: 3244953, noncomm_long: 442100, noncomm_short: 510946, net_speculative: -68846, net_pct_oi: -0.0212, percentile: 55.0 },
                ],
              },
            ],
          }}
        />
      ),
    },
    {
      id: "c30",
      code: "C30",
      name: "Implied Volatility Term Structure & SKEW",
      category: "rates_curves",
      idea: "Constant-Maturity VIX Term Structure (9D, 30D, 3M, 6M) & Regime Detection",
      insight: "Detects Contango vs Backwardation curvature and tail risk elevation via SKEW and trailing VRP.",
      deployedPage: "/volatility",
      renderComponent: () => (
        <TermStructureChart
          data={({
            schema_version: "1.0",
            as_of: "2026-11-24",
            code_version: "v1.0.0",
            params_hash: "mock",
            data: {
              as_of: "2026-11-24",
              current_state: "contango",
              slope_3m: 0.05,
              slope_9d: 0.0526,
              slope_percentile: 50.0,
              curve: [
                { tenor: "9D", days: 9, current: 14.8, one_month_ago: 18.08, one_year_median: 16.23, p10: 14.15, p90: 22.68 },
                { tenor: "30D", days: 30, current: 16.2, one_month_ago: 19.45, one_year_median: 17.50, p10: 15.20, p90: 24.10 },
                { tenor: "3M", days: 90, current: 18.1, one_month_ago: 20.80, one_year_median: 19.10, p10: 16.80, p90: 25.40 },
                { tenor: "6M", days: 180, current: 19.5, one_month_ago: 21.60, one_year_median: 20.20, p10: 17.90, p90: 26.20 },
              ],
              fear_tail_indices: [
                { id: "skew", name: "Cboe SKEW Index", current: 142.5, value: 142.5, z_score: 1.45, percentile: 88.0, chg_1w: 2.5, chg_1m: 4.2, interpretation: "Elevated crash protection demand" },
                { id: "vrp", name: "Trailing 30D VRP", current: 2.14, value: 2.14, z_score: 0.35, percentile: 45.0, chg_1w: -0.3, chg_1m: -0.8, interpretation: "Moderate volatility risk premium" },
              ],
              tail_fear: [
                { id: "skew", name: "Cboe SKEW Index", current: 142.5, value: 142.5, z_score: 1.45, percentile: 88.0, chg_1w: 2.5, chg_1m: 4.2, interpretation: "Elevated crash protection demand" },
                { id: "vrp", name: "Trailing 30D VRP", current: 2.14, value: 2.14, z_score: 0.35, percentile: 45.0, chg_1w: -0.3, chg_1m: -0.8, interpretation: "Moderate volatility risk premium" },
              ],
              implied_vs_realized: [],
              interpretation: "VIX term structure in contango",
              slope_history: [
                { date: "2026-10-01", vix: 16.5, vix3m: 17.8, slope_3m: 0.08, slope_9d: 0.05, state: "contango", percentile: 55 },
                { date: "2026-10-15", vix: 19.2, vix3m: 18.4, slope_3m: -0.04, slope_9d: -0.03, state: "backwardation", percentile: 25 },
                { date: "2026-11-01", vix: 17.0, vix3m: 17.3, slope_3m: 0.02, slope_9d: 0.01, state: "contango", percentile: 40 },
                { date: "2026-11-24", vix: 15.5, vix3m: 16.3, slope_3m: 0.05, slope_9d: 0.05, state: "contango", percentile: 50 },
              ],
            },
          } as any)}
        />
      ),
    },
    {
      id: "c31",
      code: "C31",
      name: "Sovereign Yield Curve Ribbon & NS Fit",
      category: "rates_curves",
      idea: "Multi-Date Yield Curve Snapshots & Nelson-Siegel Parametric Smoothing",
      insight: "Visualizes shifting curve shapes across Current, 1M, 6M, and 1Y horizons with Nelson-Siegel fitted curves.",
      deployedPage: "/rates",
      renderComponent: () => (
        <CurveRibbon
          data={({
            schema_version: "1.0",
            as_of: "2026-11-24",
            code_version: "v1.0.0",
            params_hash: "mock",
            data: {
              as_of: "2026-11-24",
              factors: { level: 5.32, slope: -1.53, curvature: 0.32, rmse: 0.12, level_percentile: 82.7, slope_percentile: 41.2, curvature_percentile: 90.5 },
              spreads: [],
              ten_year_anatomy: [],
              curves: [
                {
                  label: "Current",
                  date: "2026-11-24",
                  points: [{ maturity: "3M", months: 3, yield_pct: 5.42, fitted_pct: 5.40 }, { maturity: "2Y", months: 24, yield_pct: 4.88, fitted_pct: 4.85 }, { maturity: "5Y", months: 60, yield_pct: 4.35, fitted_pct: 4.38 }, { maturity: "10Y", months: 120, yield_pct: 4.45, fitted_pct: 4.44 }, { maturity: "30Y", months: 360, yield_pct: 4.62, fitted_pct: 4.61 }],
                  smooth_fitted: [{ months: 3, yield_pct: 5.40 }, { months: 12, yield_pct: 5.10 }, { months: 24, yield_pct: 4.85 }, { months: 60, yield_pct: 4.38 }, { months: 120, yield_pct: 4.44 }, { months: 360, yield_pct: 4.61 }],
                },
                {
                  label: "1 Month Ago",
                  date: "2026-10-20",
                  points: [{ maturity: "3M", months: 3, yield_pct: 5.45, fitted_pct: 5.44 }, { maturity: "2Y", months: 24, yield_pct: 4.95, fitted_pct: 4.92 }, { maturity: "5Y", months: 60, yield_pct: 4.45, fitted_pct: 4.48 }, { maturity: "10Y", months: 120, yield_pct: 4.52, fitted_pct: 4.51 }, { maturity: "30Y", months: 360, yield_pct: 4.70, fitted_pct: 4.69 }],
                  smooth_fitted: [{ months: 3, yield_pct: 5.44 }, { months: 12, yield_pct: 5.15 }, { months: 24, yield_pct: 4.92 }, { months: 60, yield_pct: 4.48 }, { months: 120, yield_pct: 4.51 }, { months: 360, yield_pct: 4.69 }],
                },
                {
                  label: "6 Months Ago",
                  date: "2026-05-20",
                  points: [{ maturity: "3M", months: 3, yield_pct: 5.38, fitted_pct: 5.36 }, { maturity: "2Y", months: 24, yield_pct: 4.75, fitted_pct: 4.72 }, { maturity: "5Y", months: 60, yield_pct: 4.25, fitted_pct: 4.28 }, { maturity: "10Y", months: 120, yield_pct: 4.32, fitted_pct: 4.31 }, { maturity: "30Y", months: 360, yield_pct: 4.48, fitted_pct: 4.47 }],
                  smooth_fitted: [{ months: 3, yield_pct: 5.36 }, { months: 12, yield_pct: 4.98 }, { months: 24, yield_pct: 4.72 }, { months: 60, yield_pct: 4.28 }, { months: 120, yield_pct: 4.31 }, { months: 360, yield_pct: 4.47 }],
                },
              ],
            },
          } as any)}
        />
      ),
    },
    {
      id: "c32",
      code: "C32",
      name: "Nelson-Siegel 3-Factor Decomposition",
      category: "rates_curves",
      idea: "Diebold-Li 3-Factor Decomposition: Level (L), Slope (S), Curvature (C)",
      insight: "Dissects yield curve shifts into parallel shifts, steepening/flattening, and curvature with historical percentiles.",
      deployedPage: "/rates",
      renderComponent: () => (
        <FactorPanel
          data={({
            schema_version: "1.0",
            as_of: "2026-11-24",
            code_version: "v1.0.0",
            params_hash: "mock",
            data: {
              as_of: "2026-11-24",
              factors: { level: 5.32, slope: -1.53, curvature: 0.32, rmse: 0.12, level_percentile: 82.7, slope_percentile: 41.2, curvature_percentile: 90.5 },
              spreads: [],
              ten_year_anatomy: [],
              curves: [],
            },
          } as any)}
        />
      ),
    },
    {
      id: "c33",
      code: "C33",
      name: "Par Yield Term Structure Heatmap & Anatomy",
      category: "rates_curves",
      idea: "Tenor x Snapshot Heatmap & 10Y Real/Breakeven Inflation Anatomy",
      insight: "Decomposes nominal yield into real 10Y TIPS and breakeven inflation rate with 2-year term structure heatmap.",
      deployedPage: "/rates",
      renderComponent: () => (
        <YieldHeatmap
          data={({
            schema_version: "1.0",
            as_of: "2026-11-24",
            code_version: "v1.0.0",
            params_hash: "mock",
            data: {
              as_of: "2026-11-24",
              factors: { level: 5.32, slope: -1.53, curvature: 0.32, rmse: 0.12, level_percentile: 82.7, slope_percentile: 41.2, curvature_percentile: 90.5 },
              spreads: [
                { name: "10Y - 2Y Curve Spread", current: 0.28, chg_1m: -0.03, percentile: 27.4, inverted: false },
                { name: "10Y - 3M Curve Spread", current: -0.97, chg_1m: 0.05, percentile: 14.2, inverted: true },
              ],
              ten_year_anatomy: [
                { component: "10Y Real Yield (TIPS)", yield_pct: 2.05, share_pct: 46.1, interpretation: "Real cost of capital" },
                { component: "10Y Breakeven Inflation", yield_pct: 2.40, share_pct: 53.9, interpretation: "Market implied inflation anchor" },
              ],
              curves: [
                {
                  label: "Current",
                  date: "2026-11-24",
                  points: [{ maturity: "3M", months: 3, yield_pct: 5.42 }, { maturity: "2Y", months: 24, yield_pct: 4.88 }, { maturity: "5Y", months: 60, yield_pct: 4.35 }, { maturity: "10Y", months: 120, yield_pct: 4.45 }, { maturity: "30Y", months: 360, yield_pct: 4.62 }],
                  smooth_fitted: [{ months: 3, yield_pct: 5.40 }, { months: 12, yield_pct: 5.10 }, { months: 24, yield_pct: 4.85 }, { months: 60, yield_pct: 4.38 }, { months: 120, yield_pct: 4.44 }, { months: 360, yield_pct: 4.61 }],
                },
              ],
              heatmap: {
                tenors: ["3M", "2Y", "5Y", "10Y", "30Y"],
                dates: ["2026-08-01", "2026-09-01", "2026-10-01", "2026-11-01"],
                matrix: [
                  [5.45, 4.90, 4.40, 4.50, 4.68],
                  [5.42, 4.85, 4.38, 4.48, 4.65],
                  [5.40, 4.82, 4.35, 4.46, 4.63],
                  [5.42, 4.88, 4.35, 4.45, 4.62],
                ],
              },
            },
          } as any)}
        />
      ),
    },
    {
      id: "c34",
      code: "C34",
      name: "Central Bank Decision Event Study & CAR Ribbon",
      category: "drivers",
      idea: "Cumulative Abnormal Return (CAR) Ribbon with Stationary Block-Bootstrap Bands",
      insight: "Plots [-5, +10] day CAR trajectories around FOMC & RBI policy announcements with 50% and 90% confidence ribbons.",
      deployedPage: "/events",
      renderComponent: () => (
        <ReactionRibbon
          data={{
            schema_version: "1.0",
            as_of: "2026-11-24",
            code_version: "v1.0.0",
            data: {
              event_type: "fomc",
              target_name: "S&P 500",
              n_events: 8,
              magnitude_ratio: 1.42,
              by_decision: {
                cut: { count: 4, mean_day_0_ar: 0.42, pct_positive: 75.0 },
                hold: { count: 3, mean_day_0_ar: -0.15, pct_positive: 33.3 },
                hike: { count: 1, mean_day_0_ar: -0.65, pct_positive: 0.0 },
              },
              ribbon: [
                { day: -5, mean_car: -0.53, median_car: -0.49, ci_lower_90: -1.16, ci_lower_50: -1.05, ci_upper_50: -0.01, ci_upper_90: 0.55 },
                { day: -3, mean_car: -0.32, median_car: -0.28, ci_lower_90: -0.92, ci_lower_50: -0.65, ci_upper_50: 0.12, ci_upper_90: 0.72 },
                { day: 0, mean_car: 0.00, median_car: 0.00, ci_lower_90: -0.45, ci_lower_50: -0.20, ci_upper_50: 0.20, ci_upper_90: 0.45 },
                { day: 2, mean_car: 0.42, median_car: 0.38, ci_lower_90: -0.35, ci_lower_50: 0.10, ci_upper_50: 0.75, ci_upper_90: 1.25 },
                { day: 5, mean_car: 0.85, median_car: 0.78, ci_lower_90: -0.15, ci_lower_50: 0.45, ci_upper_50: 1.30, ci_upper_90: 1.95 },
                { day: 10, mean_car: 1.15, median_car: 1.05, ci_lower_90: 0.05, ci_lower_50: 0.65, ci_upper_50: 1.70, ci_upper_90: 2.45 },
              ],
              events: [
                { id: "ev1", date: "2026-11-05", description: "FOMC Rate Cut (-25bps)", decision: "cut", rate_before: 5.0, rate_after: 4.75, day_0_ar: 0.35, car_5d: 1.45, car_10d: 1.82, pre_trend_t: 1.2 },
                { id: "ev2", date: "2026-09-18", description: "FOMC Rate Cut (-50bps)", decision: "cut", rate_before: 5.5, rate_after: 5.0, day_0_ar: 0.52, car_5d: 0.88, car_10d: 1.24, pre_trend_t: -0.4 },
              ],
            },
          }}
        />
      ),
    },
    {
      id: "c35",
      code: "C35",
      name: "Diebold-Yilmaz Directed Spillover Matrix",
      category: "spillover",
      idea: "Generalized Forecast Error Variance Decomposition (GFEVD) Spillover Matrix",
      insight: "Directional cross-country shock transmissions, Total Connectedness Index (TCI), and net transmitter/receiver status.",
      deployedPage: "/connectedness",
      renderComponent: () => (
        <ConnectednessMatrix
          data={{
            schema_version: "1.0",
            as_of: "2026-11-24",
            code_version: "v1.0.0",
            data: {
              as_of: "2026-11-24",
              horizon_weeks: 10,
              tci: 60.81,
              labels: ["Germany", "US", "UK", "Japan", "India"],
              countries: [
                { id: "germany", name: "Germany", own: 28.7, from_others: 71.3, to_others: 98.6, net: 27.3, role: "net_transmitter" },
                { id: "us", name: "US", own: 29.3, from_others: 70.7, to_others: 86.2, net: 15.5, role: "net_transmitter" },
                { id: "uk", name: "UK", own: 31.4, from_others: 68.6, to_others: 72.1, net: 3.5, role: "net_transmitter" },
                { id: "japan", name: "Japan", own: 45.2, from_others: 54.8, to_others: 42.1, net: -12.7, role: "net_receiver" },
                { id: "india", name: "India", own: 58.6, from_others: 41.4, to_others: 18.2, net: -23.2, role: "net_receiver" },
              ],
              matrix: [
                [28.7, 18.4, 15.2, 8.1, 4.2],
                [17.5, 29.3, 16.8, 9.4, 5.1],
                [16.2, 15.8, 31.4, 8.7, 4.8],
                [11.2, 12.4, 10.5, 45.2, 6.2],
                [8.4, 9.2, 7.8, 5.8, 58.6],
              ],
              tci_history: [
                { date: "2026-08-01", tci: 58.2 },
                { date: "2026-09-01", tci: 59.4 },
                { date: "2026-10-01", tci: 62.1 },
                { date: "2026-11-24", tci: 60.8 },
              ],
            },
          }}
        />
      ),
    },
    {
      id: "c36",
      code: "C36",
      name: "Cross-Country Net Spillover Balance",
      category: "spillover",
      idea: "Cross-Country Net Spillover Balance (Transmitter vs Receiver)",
      insight: "Ranks countries by net directional shock transmission (To Others minus From Others) in GFEVD.",
      deployedPage: "/connectedness",
      renderComponent: () => (
        <NetSpilloverBars
          data={{
            schema_version: "1.0",
            as_of: "2026-11-24",
            code_version: "v1.0.0",
            data: {
              as_of: "2026-11-24",
              horizon_weeks: 10,
              tci: 60.81,
              labels: ["Germany", "US", "UK", "Japan", "India"],
              countries: [
                { id: "germany", name: "Germany", own: 28.7, from_others: 71.3, to_others: 98.6, net: 27.3, role: "net_transmitter" },
                { id: "us", name: "US", own: 29.3, from_others: 70.7, to_others: 86.2, net: 15.5, role: "net_transmitter" },
                { id: "uk", name: "UK", own: 31.4, from_others: 68.6, to_others: 72.1, net: 3.5, role: "net_transmitter" },
                { id: "japan", name: "Japan", own: 45.2, from_others: 54.8, to_others: 42.1, net: -12.7, role: "net_receiver" },
                { id: "india", name: "India", own: 58.6, from_others: 41.4, to_others: 18.2, net: -23.2, role: "net_receiver" },
              ],
              matrix: [],
              tci_history: [],
            },
          }}
        />
      ),
    },
    {
      id: "c37",
      code: "C37",
      name: "Circular Connectedness & Spillover Graph",
      category: "spillover",
      idea: "Circular Graph Visualization of Directional Shock Propagation",
      insight: "Network graph with bilateral flow arrows sized by variance contribution.",
      deployedPage: "/connectedness",
      renderComponent: () => (
        <ChordView
          data={{
            schema_version: "1.0",
            as_of: "2026-11-24",
            code_version: "v1.0.0",
            data: {
              as_of: "2026-11-24",
              horizon_weeks: 10,
              tci: 60.81,
              labels: ["Germany", "US", "UK", "Japan", "India"],
              countries: [
                { id: "germany", name: "Germany", own: 28.7, from_others: 71.3, to_others: 98.6, net: 27.3, role: "net_transmitter" },
                { id: "us", name: "US", own: 29.3, from_others: 70.7, to_others: 86.2, net: 15.5, role: "net_transmitter" },
                { id: "uk", name: "UK", own: 31.4, from_others: 68.6, to_others: 72.1, net: 3.5, role: "net_transmitter" },
                { id: "japan", name: "Japan", own: 45.2, from_others: 54.8, to_others: 42.1, net: -12.7, role: "net_receiver" },
                { id: "india", name: "India", own: 58.6, from_others: 41.4, to_others: 18.2, net: -23.2, role: "net_receiver" },
              ],
              matrix: [
                [28.7, 18.4, 15.2, 8.1, 4.2],
                [17.5, 29.3, 16.8, 9.4, 5.1],
                [16.2, 15.8, 31.4, 8.7, 4.8],
                [11.2, 12.4, 10.5, 45.2, 6.2],
                [8.4, 9.2, 7.8, 5.8, 58.6],
              ],
              tci_history: [],
            },
          }}
        />
      ),
    },
    {
      id: "c38",
      code: "C38",
      name: "Energy 5-Year Seasonal Inventory Bands",
      category: "spillover",
      idea: "EIA 5-Year Seasonal Storage Envelopes (Excluding Current Year)",
      insight: "Compares weekly crude, gasoline, distillate, and natural gas inventories against historical 5-year min/max/mean envelopes.",
      deployedPage: "/fundamentals",
      renderComponent: () => (
        <SeasonalBandChart
          data={{
            schema_version: "1.0",
            as_of: "2026-11-24",
            code_version: "v1.0.0",
            data: {
              as_of: "2026-11-24",
              inventories: [
                {
                  id: "crude_oil",
                  name: "US Commercial Crude Oil Stocks (ex-SPR)",
                  unit: "Million Barrels",
                  as_of: "2026-11-24",
                  current_level: 433.5,
                  mean_5y: 425.0,
                  min_5y: 391.0,
                  max_5y: 459.0,
                  deviation: 8.5,
                  deviation_pct: 2.0,
                  deviation_percentile: 58.0,
                  build_draw_1w: -1.4,
                  band_years_label: "2021-2025 Range",
                  weeks: [
                    { week: 40, date: "2026-10-02", current_year: 422.0, mean_5y: 420.0, min_5y: 388.0, max_5y: 452.0 },
                    { week: 42, date: "2026-10-16", current_year: 425.5, mean_5y: 422.0, min_5y: 390.0, max_5y: 455.0 },
                    { week: 44, date: "2026-10-30", current_year: 428.1, mean_5y: 423.5, min_5y: 391.0, max_5y: 456.0 },
                    { week: 46, date: "2026-11-13", current_year: 434.9, mean_5y: 424.8, min_5y: 391.0, max_5y: 458.0 },
                    { week: 47, date: "2026-11-24", current_year: 433.5, mean_5y: 425.0, min_5y: 391.0, max_5y: 459.0 },
                  ],
                },
              ],
            },
          }}
        />
      ),
    },
    {
      id: "c39",
      code: "C39",
      name: "Extreme Value Theory (EVT) Tails & Mean Excess",
      category: "risk",
      idea: "Peaks-Over-Threshold (POT) Generalized Pareto Distribution (GPD)",
      insight: "Evaluates heavy-tailed fatness, shape parameter xi, and EVT 99%/99.5% VaR & Expected Shortfall vs Gaussian.",
      deployedPage: "/risk",
      renderComponent: () => (
        <EvtRegimePanel
          data={{
            as_of: "2026-11-24",
            target_name: "S&P 500",
            interpretation: "Heavy-tailed risk profile with xi = 0.347. EVT 99.5% VaR is 1.39x higher than Gaussian expectation.",
            evt: {
              shape_xi: 0.347,
              scale_beta: 0.634,
              threshold_u: 1.855,
              tail_fatness: "Heavy-Tailed (Fréchet)",
              var_99_pct: 4.28,
              es_99_pct: 6.53,
              var_995_pct: 5.43,
              es_995_pct: 8.30,
              gaussian_var_99: 3.08,
              gaussian_var_995: 3.41,
              empirical_var_99: 2.74,
              empirical_var_995: 3.58,
              excess_ratio_over_gaussian: 1.39,
              mean_excess_curve: [
                { threshold: 1.0, mean_excess: 0.85 },
                { threshold: 1.5, mean_excess: 1.15 },
                { threshold: 2.0, mean_excess: 1.48 },
                { threshold: 2.5, mean_excess: 1.92 },
              ],
            },
            regimes: {
              current_regime: "calm",
              prob_stress: 0.12,
              calm_expected_return: 0.08,
              stress_expected_return: -0.22,
              calm_volatility: 0.11,
              stress_volatility: 0.28,
              expected_duration_calm_days: 180,
              expected_duration_stress_days: 35,
              transition_matrix: { p_calm_to_calm: 0.985, p_calm_to_stress: 0.015, p_stress_to_calm: 0.065, p_stress_to_stress: 0.935 },
              history: [
                { date: "2026-11-10", prob_stress: 0.10, regime: "calm" },
                { date: "2026-11-15", prob_stress: 0.14, regime: "calm" },
                { date: "2026-11-24", prob_stress: 0.12, regime: "calm" },
              ],
            },
          }}
        />
      ),
    },
    {
      id: "c40",
      code: "C40",
      name: "Latent Regime Probability Ribbon",
      category: "risk",
      idea: "Hamilton 2-State Markov-Switching Latent Stress Probabilities",
      insight: "Point-in-time stress probability ribbon tracking market phase transitions and latent state switching.",
      deployedPage: "/risk",
      renderComponent: () => (
        <RegimeRibbon
          currentRegime="calm"
          currentProbStress={0.12}
          ribbon={[
            { date: "2026-10-01", prob_stress: 0.08, regime: "calm" },
            { date: "2026-10-15", prob_stress: 0.42, regime: "calm" },
            { date: "2026-11-01", prob_stress: 0.18, regime: "calm" },
            { date: "2026-11-10", prob_stress: 0.10, regime: "calm" },
            { date: "2026-11-24", prob_stress: 0.12, regime: "calm" },
          ]}
        />
      ),
    },
    {
      id: "c41",
      code: "C41",
      name: "Track Record Forward Evaluation vs Base Rates",
      category: "monitoring",
      idea: "Pre-Registered 20-Day Forward Signal Validation vs Base Rates",
      insight: "Tests macro rule hit rates against unconditioned base rates with stationary block-bootstrap confidence intervals.",
      deployedPage: "/track-record",
      renderComponent: () => (
        <TrackRecordBars
          rules={[
            {
              rule_id: "VOL_PERSISTENCE",
              rule_name: "Volatility Regime Persistence",
              hypothesis: "After 2.5σ volatility breakout, realized vol stays above 60-day median for 20 days",
              horizon_days: 20,
              n_alerts: 84,
              n_episodes: 42,
              hit_rate: 73.8,
              base_rate: 50.0,
              edge_pct: 23.8,
              ci_lower: 5.6,
              ci_upper: 50.5,
              median_lead_days: 3.0,
              status: "Validated (p < 0.05)",
            },
            {
              rule_id: "CURVE_INVERSION",
              rule_name: "10Y-2Y Inversion Unsteepening",
              hypothesis: "Rapid re-steepening from deep inversion signals cyclical stress within 20 days",
              horizon_days: 20,
              n_alerts: 52,
              n_episodes: 28,
              hit_rate: 68.2,
              base_rate: 45.0,
              edge_pct: 23.2,
              ci_lower: 4.1,
              ci_upper: 42.0,
              median_lead_days: 5.0,
              status: "Validated (p < 0.05)",
            },
          ]}
          calibration={[
            { severity: "Low", expected_rate: 15.0, observed_rate: 16.2, n_episodes: 24 },
            { severity: "Moderate", expected_rate: 30.0, observed_rate: 32.5, n_episodes: 42 },
            { severity: "Elevated", expected_rate: 50.0, observed_rate: 51.1, n_episodes: 58 },
            { severity: "High", expected_rate: 70.0, observed_rate: 68.4, n_episodes: 35 },
            { severity: "Extreme", expected_rate: 88.0, observed_rate: 85.0, n_episodes: 18 },
          ]}
        />
      ),
    },
    {
      id: "c42",
      code: "C42",
      name: "External Stress Index Cross-Check & Alignment",
      category: "monitoring",
      idea: "Cross-Correlation & Lag Analysis against Chicago Fed NFCI & St. Louis FSI",
      insight: "Measures internal stress indicator agreement and lead/lag dynamics against reference external benchmarks.",
      deployedPage: "/risk",
      renderComponent: () => (
        <StressCrossCheckPanel
          data={{
            policy_uncertainty_level: 125.4,
            policy_uncertainty_percentile: 68.0,
            episodes: [],
            stress_cross_checks: [
              {
                benchmark_name: "Chicago Fed NFCI",
                n_obs: 753,
                effective_n: 17.0,
                pearson_corr: 0.45,
                spearman_rank_corr: 0.388,
                best_lead_lag: -1,
                cross_correlation: [
                  { lag_weeks: -2, correlation: 0.43, interpretation: "Internal leads" },
                  { lag_weeks: -1, correlation: 0.45, interpretation: "Max lead" },
                  { lag_weeks: 0, correlation: 0.42, interpretation: "Contemporaneous" },
                  { lag_weeks: 1, correlation: 0.38, interpretation: "Benchmark leads" },
                ],
                agreement_rate: 0.76,
                interpretation: "Robust alignment with 1-week leading property.",
              },
              {
                benchmark_name: "St. Louis Fed FSI",
                n_obs: 753,
                effective_n: 19.5,
                pearson_corr: 0.52,
                spearman_rank_corr: 0.46,
                best_lead_lag: 0,
                cross_correlation: [
                  { lag_weeks: -1, correlation: 0.49, interpretation: "Internal leads" },
                  { lag_weeks: 0, correlation: 0.52, interpretation: "Max alignment" },
                  { lag_weeks: 1, correlation: 0.48, interpretation: "Benchmark leads" },
                ],
                agreement_rate: 0.81,
                interpretation: "Strong contemporaneous alignment.",
              },
            ],
            episode_stress_alignments: [],
          }}
        />
      ),
    },
    {
      id: "c43",
      code: "C43",
      name: "Sector Performance Heatmap",
      category: "attribution",
      idea: "Sector Performance & Weight Dispersion across 1D, 5D, 1M, 3M, 1Y",
      insight: "Sortable sector performance matrix with relative outperformance vs market benchmark.",
      deployedPage: "/sectors",
      renderComponent: () => (
        <SectorHeatmap
          benchmarkName="S&P 500"
          benchmarkReturns={{ "1D": -0.037, "5D": -0.086, "1M": -0.098, "3M": -0.100, "1Y": 0.009 }}
          sectors={[
            { id: "tech", name: "Information Technology", weight: 0.315, returns: { "1D": -0.042, "5D": -0.095, "1M": -0.112, "3M": -0.125, "1Y": 0.082 }, percentiles: { "1D": 78, "5D": 68, "1M": 55, "3M": 62, "1Y": 88 }, dispersion_z: 1.2, phase: "Leading", benchmark_rel: 0.073 },
            { id: "fin", name: "Financials", weight: 0.132, returns: { "1D": -0.031, "5D": -0.072, "1M": -0.084, "3M": -0.088, "1Y": 0.035 }, percentiles: { "1D": 62, "5D": 58, "1M": 52, "3M": 54, "1Y": 72 }, dispersion_z: -0.4, phase: "Fading", benchmark_rel: 0.026 },
            { id: "health", name: "Health Care", weight: 0.118, returns: { "1D": -0.021, "5D": -0.054, "1M": -0.062, "3M": -0.065, "1Y": -0.012 }, percentiles: { "1D": 45, "5D": 42, "1M": 48, "3M": 46, "1Y": 51 }, dispersion_z: -0.8, phase: "Recovering", benchmark_rel: -0.021 },
            { id: "energy", name: "Energy", weight: 0.038, returns: { "1D": -0.015, "5D": -0.038, "1M": -0.045, "3M": -0.052, "1Y": -0.064 }, percentiles: { "1D": 32, "5D": 35, "1M": 30, "3M": 28, "1Y": 38 }, dispersion_z: -1.4, phase: "Lagging", benchmark_rel: -0.073 },
          ]}
        />
      ),
    },
    {
      id: "c44",
      code: "C44",
      name: "Relative Rotation Graph (RRG) Sector Map",
      category: "attribution",
      idea: "Relative Strength Ratio vs RS-Momentum with 10D Trajectory Tails",
      insight: "4-Quadrant RRG tracking sectors through Leading, Weakening, Lagging, and Improving market phases.",
      deployedPage: "/sectors",
      renderComponent: () => (
        <SectorRotationMap
          benchmarkName="S&P 500"
          sectors={[
            { id: "tech", name: "Technology", weight: 0.315, returns: { "1D": -0.042, "5D": -0.095, "1M": -0.112, "3M": -0.125, "1Y": 0.082 }, dispersion_z: 1.2, phase: "Leading", benchmark_rel: 0.073, rs_ratio: 104.5, rs_momentum: 101.2, tail: [{ x: 102.1, y: 100.2 }, { x: 103.4, y: 100.8 }, { x: 104.5, y: 101.2 }] },
            { id: "fin", name: "Financials", weight: 0.132, returns: { "1D": -0.031, "5D": -0.072, "1M": -0.084, "3M": -0.088, "1Y": 0.035 }, dispersion_z: -0.4, phase: "Fading", benchmark_rel: 0.026, rs_ratio: 101.2, rs_momentum: 98.4, tail: [{ x: 102.5, y: 99.5 }, { x: 101.8, y: 98.9 }, { x: 101.2, y: 98.4 }] },
            { id: "energy", name: "Energy", weight: 0.038, returns: { "1D": -0.015, "5D": -0.038, "1M": -0.045, "3M": -0.052, "1Y": -0.064 }, dispersion_z: -1.4, phase: "Lagging", benchmark_rel: -0.073, rs_ratio: 96.5, rs_momentum: 97.2, tail: [{ x: 97.8, y: 98.0 }, { x: 97.0, y: 97.5 }, { x: 96.5, y: 97.2 }] },
            { id: "util", name: "Utilities", weight: 0.025, returns: { "1D": -0.008, "5D": -0.022, "1M": -0.031, "3M": -0.035, "1Y": 0.042 }, dispersion_z: 0.6, phase: "Recovering", benchmark_rel: 0.033, rs_ratio: 98.2, rs_momentum: 102.5, tail: [{ x: 96.5, y: 101.0 }, { x: 97.3, y: 101.8 }, { x: 98.2, y: 102.5 }] },
          ]}
        />
      ),
    },
    {
      id: "c45",
      code: "C45",
      name: "Sector Return Contribution Waterfall",
      category: "attribution",
      idea: "Benchmark Index Return Decomposed into Weighted Sector Contributions",
      insight: "Breaks index performance into sectoral attribution and residual interaction terms.",
      deployedPage: "/sectors",
      renderComponent: () => (
        <SectorContributionWaterfall
          benchmarkName="S&P 500"
          indexReturn={-0.037}
          residual={0.004}
          horizon="1D"
          weightsSource="S&P Dow Jones Indices"
          asOf="2026-11-24"
          sectors={[
            { id: "tech", name: "Information Technology", weight: 0.315, returns: { "1D": -0.042, "5D": -0.095, "1M": -0.112, "3M": -0.125, "1Y": 0.082 }, contributions: { "1D": -0.0132, "5D": -0.0299, "1M": -0.0353, "3M": -0.0394, "1Y": 0.0258 }, dispersion_z: 1.2, phase: "Leading", benchmark_rel: 0.073 },
            { id: "fin", name: "Financials", weight: 0.132, returns: { "1D": -0.031, "5D": -0.072, "1M": -0.084, "3M": -0.088, "1Y": 0.035 }, contributions: { "1D": -0.0041, "5D": -0.0095, "1M": -0.0111, "3M": -0.0116, "1Y": 0.0046 }, dispersion_z: -0.4, phase: "Fading", benchmark_rel: 0.026 },
            { id: "health", name: "Health Care", weight: 0.118, returns: { "1D": -0.021, "5D": -0.054, "1M": -0.062, "3M": -0.065, "1Y": -0.012 }, contributions: { "1D": -0.0025, "5D": -0.0064, "1M": -0.0073, "3M": -0.0077, "1Y": -0.0014 }, dispersion_z: -0.8, phase: "Recovering", benchmark_rel: -0.021 },
            { id: "comm", name: "Communication Services", weight: 0.089, returns: { "1D": -0.038, "5D": -0.082, "1M": -0.095, "3M": -0.102, "1Y": 0.054 }, contributions: { "1D": -0.0034, "5D": -0.0073, "1M": -0.0085, "3M": -0.0091, "1Y": 0.0048 }, dispersion_z: 0.5, phase: "Leading", benchmark_rel: 0.045 },
            { id: "energy", name: "Energy", weight: 0.038, returns: { "1D": -0.015, "5D": -0.038, "1M": -0.045, "3M": -0.052, "1Y": -0.064 }, contributions: { "1D": -0.0006, "5D": -0.0014, "1M": -0.0017, "3M": -0.0020, "1Y": -0.0024 }, dispersion_z: -1.4, phase: "Lagging", benchmark_rel: -0.073 },
          ]}
        />
      ),
    },
    {
      id: "c46",
      code: "C46",
      name: "FII & DII Net Flows Monitor",
      category: "positioning",
      idea: "Daily & 20-Day Cumulative Institutional Flow Dynamics (INR Crores)",
      insight: "Tracks Foreign (FII) vs Domestic (DII) institutional buying and selling momentum.",
      deployedPage: "/india",
      renderComponent: () => (
        <FiiDiiFlowChart
          dates={["2026-11-13", "2026-11-14", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-24"]}
          fiiDaily={[1450, -820, 2100, -1250, 1840, 1250]}
          diiDaily={[850, 1240, -450, 2100, 1520, 1840]}
          fiiCum20d={[11200, 10380, 12480, 11230, 13070, 16684]}
          diiCum20d={[8400, 9640, 9190, 11290, 12810, 13040]}
        />
      ),
    },
    {
      id: "c47",
      code: "C47",
      name: "F&O Open Interest Buildup & Put-Call Ratio",
      category: "positioning",
      idea: "Derivatives Buildup Classification (Long/Short Buildup, Covering, Unwinding)",
      insight: "Classifies key index and sectoral contracts into price-OI quadrants alongside Put-Call Ratio (PCR).",
      deployedPage: "/india",
      renderComponent: () => (
        <OiBuildupMap
          pcr={1.08}
          items={[
            { symbol: "NIFTY", price_chg_pct: 0.85, oi_chg_pct: 4.2, buildup: "Long Buildup", pcr: 1.12 },
            { symbol: "BANKNIFTY", price_chg_pct: -0.62, oi_chg_pct: 6.8, buildup: "Short Buildup", pcr: 0.85 },
            { symbol: "FINNIFTY", price_chg_pct: 0.42, oi_chg_pct: -2.1, buildup: "Short Covering", pcr: 1.05 },
            { symbol: "MIDCPNIFTY", price_chg_pct: -0.35, oi_chg_pct: -3.4, buildup: "Long Unwinding", pcr: 0.92 },
          ]}
        />
      ),
    },
    {
      id: "c48",
      code: "C48",
      name: "Cross-Asset Volatility Landscape & Percentile Heatmap",
      category: "risk",
      idea: "Multi-Horizon Volatility Term Structure (10D, 20D, 60D) & Historical Percentile",
      insight: "Ranks multi-asset instruments by 20D annualized volatility percentile and vol trend slope.",
      deployedPage: "/",
      renderComponent: () => (
        <VolatilityLandscape
          landscape={[
            { asset: "spx", name: "S&P 500 Index", group: "equity_index", region: "US", vol10: 0.145, vol20: 0.162, vol60: 0.138, vol_trend: 0.174, vol20_percentile: 65.4, n: 734, n_eff: 12.4 },
            { asset: "nifty50", name: "Nifty 50 Index", group: "india_equity", region: "IN", vol10: 0.118, vol20: 0.125, vol60: 0.132, vol_trend: -0.053, vol20_percentile: 28.5, n: 734, n_eff: 14.1 },
            { asset: "brent", name: "Brent Crude Oil", group: "commodity", region: "Global", vol10: 0.285, vol20: 0.312, vol60: 0.245, vol_trend: 0.273, vol20_percentile: 88.2, n: 734, n_eff: 10.8 },
            { asset: "gold", name: "Gold COMEX", group: "commodity", region: "Global", vol10: 0.142, vol20: 0.138, vol60: 0.151, vol_trend: -0.086, vol20_percentile: 42.1, n: 734, n_eff: 15.2 },
            { asset: "us10y", name: "US 10Y Yield", group: "rate", region: "US", vol10: 0.195, vol20: 0.212, vol60: 0.178, vol_trend: 0.191, vol20_percentile: 79.5, n: 734, n_eff: 11.5 },
          ]}
        />
      ),
    },
    {
      id: "c49",
      code: "C49",
      name: "Composite Financial Stress Stack by Component",
      category: "risk",
      idea: "Multi-Asset Composite Stress Decomposition (6 Sub-Indices)",
      insight: "Decomposes composite financial stress into Equity, Rates, FX, Commodity, Correlation, and Credit sub-indices with crisis overlays.",
      deployedPage: "/risk",
      renderComponent: () => (
        <StressStackChart
          data={{
            schema_version: "1.0",
            as_of: "2026-11-24",
            code_version: "v1.0.0",
            data: {
              latest: {
                date: "2026-11-24",
                composite: 42.0,
                tercile: "Moderate",
                components: { equity: 48.0, rates: 52.0, fx: 31.0, commodity: 44.0, credit: 38.0, correlation: 35.0 },
              },
              components: [
                { id: "equity", name: "Equity Stress", weight: 0.25, current: 0.48, z_score: 0.65, percentile: 68.0 },
                { id: "rates", name: "Rates Stress", weight: 0.20, current: 0.52, z_score: 0.82, percentile: 74.0 },
                { id: "fx", name: "FX Stress", weight: 0.15, current: 0.31, z_score: 0.15, percentile: 48.0 },
                { id: "commodity", name: "Commodity Stress", weight: 0.15, current: 0.44, z_score: 0.55, percentile: 64.0 },
                { id: "credit", name: "Credit Stress", weight: 0.15, current: 0.38, z_score: 0.35, percentile: 55.0 },
                { id: "correlation", name: "Cross-Asset Correlation", weight: 0.10, current: 0.35, z_score: 0.22, percentile: 52.0 },
              ],
              series: [
                { date: "2026-08-01", composite: 0.28, components: { equity: 0.30, rates: 0.25, fx: 0.22, commodity: 0.32, credit: 0.26, correlation: 0.28 } },
                { date: "2026-09-01", composite: 0.35, components: { equity: 0.38, rates: 0.42, fx: 0.26, commodity: 0.36, credit: 0.30, correlation: 0.31 } },
                { date: "2026-10-01", composite: 0.58, components: { equity: 0.68, rates: 0.62, fx: 0.45, commodity: 0.54, credit: 0.51, correlation: 0.48 } },
                { date: "2026-11-24", composite: 0.42, components: { equity: 0.48, rates: 0.52, fx: 0.31, commodity: 0.44, credit: 0.38, correlation: 0.35 } },
              ],
              episodes: [
                { name: "October 2026 Spike", start: "2026-09-28", end: "2026-10-18", peak_date: "2026-10-08", peak_composite: 0.68 },
              ],
            },
          }}
        />
      ),
    },
    {
      id: "c50",
      code: "C50",
      name: "Historical Financial Stress Index & Crisis Episodes",
      category: "risk",
      idea: "Long-Horizon Financial Stress Time Series with Historical Episode Markers",
      insight: "Tracks composite stress level over time with highlighted historical crisis periods.",
      deployedPage: "/risk",
      renderComponent: () => (
        <StressHistoryChart
          dates={["2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01", "2026-10-01", "2026-11-01", "2026-11-24"]}
          history={[0.22, 0.25, 0.28, 0.35, 0.58, 0.46, 0.42]}
          episodes={[
            { id: "oct26", name: "October 2026 Liquidity Squeeze", start: "2026-09-28", end: "2026-10-18", peak_date: "2026-10-08", peak_composite: 0.68 },
          ]}
        />
      ),
    },
    {
      id: "c51",
      code: "C51",
      name: "Source Data Health, Ingestion & Circuit Breakers",
      category: "monitoring",
      idea: "Source Integrity Circuit Breaker & Schema Drift Audit Monitor",
      insight: "Tracks ingestion health, schema drift, stale thresholds, and fallback activation across all upstream feeds.",
      deployedPage: "/data-health",
      renderComponent: () => (
        <SourceStatusTable
          overallStatus="healthy"
          totalTracked={7}
          totalHealthy={7}
          sources={[
            {
              id: "cboe",
              name: "Cboe Global Markets",
              provider: "Cboe Options Exchange",
              documentation_url: "https://www.cboe.com/tradable_products/vix/",
              terms_url: "https://www.cboe.com/terms/",
              rate_limit: "20 req/min",
              schedule: "Daily 17:00 US Eastern",
              status: "healthy",
              last_success: "2026-11-24 22:30 UTC",
              last_verified: "2026-11-24",
              tracked_series_count: 5,
              stale_series_count: 0,
              circuit_state: "Closed",
              owner_notes: "VIX term structure",
            },
            {
              id: "treasury",
              name: "U.S. Department of the Treasury",
              provider: "U.S. Department of the Treasury",
              documentation_url: "https://home.treasury.gov/",
              terms_url: "https://home.treasury.gov/utility/privacy-policy",
              rate_limit: "60 req/min",
              schedule: "Daily 17:00 US Eastern",
              status: "healthy",
              last_success: "2026-11-24 21:45 UTC",
              last_verified: "2026-11-24",
              tracked_series_count: 12,
              stale_series_count: 0,
              circuit_state: "Closed",
              owner_notes: "Daily Treasury par yield curve",
            },
            {
              id: "cftc",
              name: "Commodity Futures Trading Commission",
              provider: "CFTC (Socrata Public API)",
              documentation_url: "https://www.cftc.gov/",
              terms_url: "https://www.cftc.gov/About/PrivacyWebsiteNotices",
              rate_limit: "60 req/min",
              schedule: "Weekly (Friday 15:30 US Eastern)",
              status: "healthy",
              last_success: "2026-11-24 20:30 UTC",
              last_verified: "2026-11-24",
              tracked_series_count: 16,
              stale_series_count: 0,
              circuit_state: "Closed",
              owner_notes: "Commitments of Traders legacy reports",
            },
            {
              id: "eia",
              name: "U.S. Energy Information Administration",
              provider: "U.S. Department of Energy",
              documentation_url: "https://www.eia.gov/opendata/",
              terms_url: "https://www.eia.gov/about/copyrights_reuse.php",
              rate_limit: "60 req/min",
              schedule: "Weekly (Wed/Thu)",
              status: "healthy",
              last_success: "2026-11-18 15:30 UTC",
              last_verified: "2026-11-24",
              tracked_series_count: 4,
              stale_series_count: 0,
              circuit_state: "Closed",
              owner_notes: "Petroleum & natural gas storage",
            },
            {
              id: "fred",
              name: "Federal Reserve Economic Data (FRED)",
              provider: "Federal Reserve Bank of St. Louis",
              documentation_url: "https://fred.stlouisfed.org/docs/api/fred/",
              terms_url: "https://fred.stlouisfed.org/legal/",
              rate_limit: "120 req/min",
              schedule: "Daily & Weekly",
              status: "healthy",
              last_success: "2026-11-24 23:00 UTC",
              last_verified: "2026-11-24",
              tracked_series_count: 8,
              stale_series_count: 0,
              circuit_state: "Closed",
              owner_notes: "Macro aggregates & reference benchmarks",
            },
          ]}
        />
      ),
    },
  ];

  const filtered = catalogue.filter((item) => {
    const matchesCategory = activeCategory === "all" || item.category === activeCategory;
    const query = searchQuery.trim().toLowerCase();
    const matchesSearch =
      !query ||
      item.code.toLowerCase().includes(query) ||
      item.name.toLowerCase().includes(query) ||
      item.idea.toLowerCase().includes(query) ||
      item.insight.toLowerCase().includes(query) ||
      item.deployedPage.toLowerCase().includes(query);
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-16">
      {/* Clean Header */}
      <div className="pb-4 border-b border-[var(--line)]">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">MacroMonitor</span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[var(--bg-2)] border border-[var(--line)] text-[var(--accent)] font-semibold">
            51 Charts &amp; Models
          </span>
        </div>
        <h1 className="text-2xl font-serif font-bold tracking-tight text-[var(--tx-0)] mt-1">
          Chart Gallery
        </h1>
        <p className="text-xs text-[var(--tx-1)] mt-1 max-w-3xl leading-relaxed">
          Interactive gallery of macroeconomic charts, yield curve models, positioning indicators, shock spillovers, and risk analytics across asset classes.
        </p>
      </div>

      {/* Filter Tabs & Search Bar */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 border-b border-[var(--line)] pb-3 text-xs font-mono overflow-x-auto">
          {[
            { id: "all", label: "All (51)" },
            { id: "attribution", label: "Returns & Attribution (8)" },
            { id: "drivers", label: "Macro Drivers & Events (7)" },
            { id: "rates_curves", label: "Yield Curves & Volatility (4)" },
            { id: "positioning", label: "Positioning & Flows (4)" },
            { id: "spillover", label: "Connectedness & Energy (4)" },
            { id: "risk", label: "Risk & Regimes (14)" },
            { id: "monitoring", label: "Monitoring & Health (10)" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveCategory(tab.id as any)}
              className={`px-3 py-1.5 rounded transition-colors whitespace-nowrap ${activeCategory === tab.id
                ? "bg-[var(--accent)] text-slate-900 font-semibold"
                : "text-[var(--tx-2)] hover:text-[var(--tx-0)] bg-[var(--bg-1)] border border-[var(--line)]"
                }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-1">
          <div className="relative flex-1 max-w-md">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search 51 charts (e.g. C30, VIX, Nelson-Siegel, CFTC, EVT, Spillover, FII)..."
              className="w-full px-3.5 py-2 pl-9 bg-[var(--bg-1)] border border-[var(--line)] rounded-md text-xs font-mono text-[var(--tx-0)] placeholder-[var(--tx-2)] focus:outline-none focus:border-[var(--accent)] transition"
            />
            <Search className="w-4 h-4 text-[var(--tx-2)] absolute left-2.5 top-2.5" />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-2.5 top-2 text-xs text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              >
                ✕
              </button>
            )}
          </div>
          <div className="text-xs font-mono text-[var(--tx-2)]">
            Showing <span className="font-bold text-[var(--accent)]">{filtered.length}</span> of 51 charts
          </div>
        </div>
      </div>

      {/* Visual Grid: All 51 Instruments */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {filtered.map((item) => (
          <div
            key={item.id}
            className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] hover:border-[var(--accent)] transition space-y-4 flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="flex items-center justify-between border-b border-[var(--line)] pb-2">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-[var(--bg-0)] border border-[var(--accent)] font-mono font-bold text-xs text-[var(--accent)]">
                    {item.code}
                  </span>
                  <h3 className="text-sm font-semibold text-[var(--tx-0)]">{item.name}</h3>
                </div>
                <Link
                  href={item.deployedPage}
                  className="flex items-center gap-1 text-[11px] font-mono text-[var(--accent)] hover:underline"
                >
                  <span>Open {item.deployedPage}</span>
                  <ExternalLink className="w-3 h-3" />
                </Link>
              </div>

              <div className="text-xs font-mono text-[var(--tx-2)]">
                <strong className="text-[var(--tx-1)]">Method:</strong> {item.idea}
              </div>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">{item.insight}</p>
            </div>

            {/* Interactive Live Chart Preview */}
            <div className="pt-2 border-t border-[var(--line)]">
              <LazyChartItem render={item.renderComponent} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
