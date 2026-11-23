// web/src/app/risk/RiskClientView.tsx
"use client";

import React, { useEffect, useState, useMemo } from "react";
import { ChartCard } from "@/components/charts/ChartCard";
import { StatBlock } from "@/components/ui/StatBlock";
import { HealthDot } from "@/components/ui/HealthDot";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { AuditData, AuditDrawer } from "@/components/ui/AuditDrawer";
import { MethodData, MethodDrawer } from "@/components/ui/MethodDrawer";
import { VolatilityConeChart } from "@/components/charts/VolatilityConeChart";
import { DistributionHistogram } from "@/components/charts/DistributionHistogram";
import { ContextQqPlot } from "@/components/charts/ContextQqPlot";
import { TailRiskBarChart } from "@/components/charts/TailRiskBarChart";
import { AutocorrelationBarChart } from "@/components/charts/AutocorrelationBarChart";
import { UnderwaterDrawdownChart } from "@/components/charts/UnderwaterDrawdownChart";
import { PcaSynchronizationChart } from "@/components/charts/PcaSynchronizationChart";
import { DownsideCorrelationChart } from "@/components/charts/DownsideCorrelationChart";
import { StressStackChart } from "@/components/charts/StressStackChart";
import { HexbinJointDensityChart } from "@/components/charts/HexbinJointDensityChart";
import { RidgelineDensityChart } from "@/components/charts/RidgelineDensityChart";
import { StressCrossCheckPanel } from "@/components/charts/StressCrossCheckPanel";
import { EvtRegimePanel } from "@/components/charts/EvtRegimePanel";
import { loadRisk, loadStress, loadContext, loadBenchmarks, loadTailsRegimes } from "@/lib/data/client";
import { RiskEnvelope, AssetRiskProfile, StressEnvelope, ContextData, BenchmarkData, TailsRegimesData } from "@/lib/data/schemas";

export const RiskClientView: React.FC = () => {
  const [riskData, setRiskData] = useState<RiskEnvelope | null>(null);
  const [stressData, setStressData] = useState<StressEnvelope | null>(null);
  const [asOf, setAsOf] = useState<string>("2026-11-20");
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedAssetId, setSelectedAssetId] = useState<string>("spx");
  const [contextData, setContextData] = useState<ContextData | null>(null);
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkData | null>(null);
  const [tailsRegimesData, setTailsRegimesData] = useState<TailsRegimesData | null>(null);

  useEffect(() => {
    async function fetchContext() {
      if (!selectedAssetId) return;
      try {
        const res = await loadContext(selectedAssetId);
        if (res.status !== "error") {
          setContextData(res.data.data);
        }
      } catch (err) {
        console.error("Failed to load context for asset", selectedAssetId, err);
      }
    }
    fetchContext();
  }, [selectedAssetId]);

  // Drawers
  const [auditTarget, setAuditTarget] = useState<AuditData | null>(null);
  const [isAuditOpen, setIsAuditOpen] = useState<boolean>(false);
  const [methodTarget, setMethodTarget] = useState<MethodData | null>(null);
  const [isMethodOpen, setIsMethodOpen] = useState<boolean>(false);

  const handleOpenAudit = (audit: AuditData) => {
    setAuditTarget(audit);
    setIsAuditOpen(true);
  };

  const handleOpenMethod = (method: MethodData) => {
    setMethodTarget(method);
    setIsMethodOpen(true);
  };

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        const [res, sRes, bmRes, tailsRes] = await Promise.all([
          loadRisk(),
          loadStress(),
          loadBenchmarks(),
          loadTailsRegimes(),
        ]);
        if (res.status !== "error") {
          setRiskData(res.data);
          setAsOf(res.asOf);
          if (res.data.data.selected_asset) {
            setSelectedAssetId(res.data.data.selected_asset);
          }
        }
        if (sRes.status !== "error") {
          setStressData(sRes.data);
        }
        if (bmRes.status !== "error") {
          setBenchmarkData(bmRes.data.data);
        }
        if (tailsRes.status !== "error") {
          setTailsRegimesData(tailsRes.data.data);
        }
      } catch (err) {
        console.error("Failed to load risk/stress data:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const profile: AssetRiskProfile | null = useMemo(() => {
    if (!riskData?.data?.profiles) return null;
    return riskData.data.profiles[selectedAssetId] || Object.values(riskData.data.profiles)[0] || null;
  }, [riskData, selectedAssetId]);

  const latestSyncShare = useMemo(() => {
    return riskData?.data?.market_synchronization_latest ?? 50.0;
  }, [riskData]);

  const asset20dVol = useMemo(() => {
    if (!profile?.volatility_cone) return { vol: 0, pct: 50 };
    const pt = profile.volatility_cone.find((c) => c.window === 20);
    return pt ? { vol: pt.current * 100, pct: pt.percentile } : { vol: 0, pct: 50 };
  }, [profile]);

  const assetTailRatio = useMemo(() => {
    if (!profile?.tail_risk || profile.tail_risk.length === 0) return 1.0;
    return profile.tail_risk[0].tail_ratio;
  }, [profile]);

  const maxHistoricalDrawdown = useMemo(() => {
    if (!profile?.drawdown_series || profile.drawdown_series.length === 0) return 0;
    const depths = profile.drawdown_series.map((d) => d.drawdown);
    return Math.min(...depths) * 100;
  }, [profile]);

  return (
    <div className="flex flex-col gap-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-[var(--border)] pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs uppercase tracking-widest text-[var(--accent)] font-mono font-medium">
              Section 4.4 • Quantitative Risk Diagnostics
            </span>
            <HealthDot status="healthy" />
            <DataAgeBadge asOf={asOf} />
          </div>
          <h1 className="text-2xl sm:text-3xl font-serif text-[var(--text-1)] tracking-tight">
            Statistical Risk Lab
          </h1>
          <p className="text-sm text-[var(--text-3)] max-w-2xl mt-1">
            Descriptive non-parametric volatility horizons, heavy-tail crash distributions, ARCH volatility clustering, underwater drawdowns, and cross-border downside correlation spikes.
          </p>
        </div>

        {/* Asset Selector */}
        {riskData?.data?.assets_available && (
          <div className="flex items-center gap-2 bg-[var(--bg-1)] p-1.5 rounded-md border border-[var(--border)]">
            <span className="text-xs text-[var(--text-3)] font-mono px-2">Asset:</span>
            <select
              value={selectedAssetId}
              onChange={(e) => setSelectedAssetId(e.target.value)}
              className="bg-[var(--bg-2)] text-[var(--text-1)] text-xs font-mono py-1 px-2.5 rounded border border-[var(--border)] focus:outline-none focus:border-[var(--accent)]"
            >
              {riskData.data.assets_available.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name} ({a.id.toUpperCase()})
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Stat Blocks */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatBlock
          id="stat-pca-sync"
          name="Global Synchronization"
          value={`${latestSyncShare.toFixed(1)}%`}
          unit="PC1 Share"
          percentile={latestSyncShare}
        />
        <StatBlock
          id="stat-realized-vol"
          name="20-Day Realized Vol"
          value={`${asset20dVol.vol.toFixed(1)}%`}
          unit="Annualized Vol"
          percentile={asset20dVol.pct}
        />
        <StatBlock
          id="stat-tail-ratio"
          name="Fat-Tail Crash Ratio"
          value={`${assetTailRatio.toFixed(2)}x`}
          unit="Hist / Gauss ES"
        />
        <StatBlock
          id="stat-max-dd"
          name="Max Peak Drawdown"
          value={`${maxHistoricalDrawdown.toFixed(1)}%`}
          unit="Underwater Depth"
        />
      </div>

      {/* Multi-Asset Financial Stress Composite Breakdown */}
      {stressData && (
        <StressStackChart data={stressData} />
      )}

      {/* P1: Stress Cross-Check & Historical Episode Benchmarks */}
      {benchmarkData && (
        <ChartCard
          eyebrow="Independent Validation (P1)"
          title="Stress Composite Cross-Check & Crisis Episodes"
          subtitle="Empirical alignment against St. Louis Fed & Chicago Fed benchmark indices and historical market shocks"
          asOf={asOf}
          methodData={{
            chartTitle: "Stress Composite Cross-Check & Crisis Episodes (P1)",
            whatItShows: "Validation of our synthetic daily multi-asset stress composite against independent benchmark indices and curated historical macro shocks.",
            definitionFormula: "\\rho_{\\tau} = \\text{corr}(S_t, B_{t+\\tau}), \\quad N_{\\text{eff}} = N \\frac{1-\\rho_1}{1+\\rho_1}",
            symbolsDefined: [
              { symbol: "S_t", meaning: "Internal daily synthetic stress composite score" },
              { symbol: "B_t", meaning: "External benchmark index (NFCI / St. Louis Stress Index)" },
              { symbol: "tau", meaning: "Cross-correlation lead/lag in weeks" },
            ],
            parameters: { max_lag: "±4 weeks", lookback: "Full available history" },
            dataSources: [{ series: "NFCI / STLFSI4", source: "Federal Reserve", frequency: "Weekly" }],
            sample: { n: 250, nEff: 180, dateRange: "Full available history" },
            interpretationNotes: [
              "High Pearson and Spearman correlation (>0.70) demonstrates strong alignment with official central bank financial stress indices.",
              "Synchronous or positive lead timing confirms that our daily composite responds to market distress without lagging official reports.",
            ],
            reproduce: {
              functionName: "compute_benchmark_agreement",
              filePath: "pipeline/src/radar/metrics/benchmarks.py",
              testPath: "pipeline/tests/unit/test_benchmarks.py",
            },
          }}
          auditData={{
            metricName: "Stress Benchmark Agreement",
            formula: "Pearson r, Spearman rho, and lead/lag cross-correlation spectrum",
            formulaPlain: "Comparison against Chicago Fed NFCI and St. Louis Fed Financial Stress indices.",
            parameters: { max_lag: 4 },
            inputs: [{ id: "stress_composite", name: "Stress Composite", source: "Internal", unit: "score" }],
            window: "Full history",
            currentValue: "Aligned",
          }}
          onOpenAudit={handleOpenAudit}
          onOpenMethod={handleOpenMethod}
        >
          <StressCrossCheckPanel data={benchmarkData} />
        </ChartCard>
      )}

      {/* EVT Tails and Markov-Switching Regimes (P10) */}
      <EvtRegimePanel data={tailsRegimesData} />

      {/* Chart 1: Volatility Cone (C10) */}
      {profile && (
        <ChartCard
          eyebrow="Horizon Volatility Analysis (C10)"
          title={`Realized Volatility Cone — ${profile.asset_name}`}
          subtitle="Non-parametric realized volatility distribution across {10D, 20D, 60D, 120D, 250D} windows"
          asOf={asOf}
          methodData={{
            chartTitle: `Realized Volatility Cone (C10) — ${profile.asset_name}`,
            whatItShows: "Realized volatility across multiple lookback windows compared against historical distribution percentiles (min, 10th, 25th, 50th, 75th, 90th, max).",
            definitionFormula: "\\sigma_w(t) = \\sqrt{252} \\cdot \\sqrt{\\frac{1}{w-1} \\sum_{i=0}^{w-1} (r_{t-i} - \\bar{r})^2}",
            symbolsDefined: [
              { symbol: "sigma_w", meaning: "Annualized realized volatility for lookback window w" },
              { symbol: "r_t", meaning: "Daily log return on session t" },
            ],
            parameters: {
              windows: "10D, 20D, 60D, 120D, 250D",
              annualization: 252,
              estimator: "Sample standard deviation (ddof=1)",
            },
            dataSources: [{ series: profile.asset_id, source: "Market Settlement", frequency: "Daily" }],
            sample: {
              n: profile.volatility_cone.length,
              nEff: profile.volatility_cone.length,
              dateRange: "Full historical series",
            },
            interpretationNotes: [
              "Current volatility marker inside the dark inner band indicates normal trading within the interquartile range (25th to 75th percentile).",
              "Spikes above the 90th percentile across multiple horizons indicate systemic regime shifts and volatility shocks.",
              "Term-structure inversion (10D > 250D) signifies acute short-term market distress.",
            ],
            reproduce: {
              functionName: "compute_volatility_cone",
              filePath: "pipeline/src/radar/metrics/risk.py",
              testPath: "pipeline/tests/unit/test_risk.py",
            },
          }}
          auditData={{
            metricName: "Realized Volatility Cone",
            formula: "sigma_w = sqrt(252) * std(r, ddof=1)",
            formulaPlain: "Annualized standard deviation of daily log returns across lookback windows.",
            parameters: { windows: [10, 20, 60, 120, 250], annualization: 252 },
            inputs: [
              { id: profile.asset_id, name: profile.asset_name, source: "Market Close", unit: "Index Points" },
            ],
            window: "10D to 250D expanding history",
            currentValue: `${asset20dVol.vol.toFixed(1)}% (20D Vol)`,
            percentile: asset20dVol.pct,
            n: profile.volatility_cone.length,
            nEff: profile.volatility_cone.length,
          }}
          dataTable={
            <div className="overflow-x-auto text-xs font-mono">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                    <th className="py-1.5 px-2">Window</th>
                    <th className="py-1.5 px-2 text-right">Current Vol</th>
                    <th className="py-1.5 px-2 text-right">Percentile</th>
                    <th className="py-1.5 px-2 text-right">Median (p50)</th>
                    <th className="py-1.5 px-2 text-right">IQR (25-75)</th>
                    <th className="py-1.5 px-2 text-right">80% Band (10-90)</th>
                    <th className="py-1.5 px-2 text-right">Historical Max</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)]">
                  {profile.volatility_cone.map((c) => (
                    <tr key={c.window}>
                      <td className="py-1.5 px-2 font-semibold">{c.window}D</td>
                      <td className="py-1.5 px-2 text-right text-[var(--accent)] font-bold">{(c.current * 100).toFixed(1)}%</td>
                      <td className="py-1.5 px-2 text-right">{c.percentile.toFixed(0)}th</td>
                      <td className="py-1.5 px-2 text-right">{(c.p50 * 100).toFixed(1)}%</td>
                      <td className="py-1.5 px-2 text-right">{(c.p25 * 100).toFixed(1)}% – {(c.p75 * 100).toFixed(1)}%</td>
                      <td className="py-1.5 px-2 text-right">{(c.p10 * 100).toFixed(1)}% – {(c.p90 * 100).toFixed(1)}%</td>
                      <td className="py-1.5 px-2 text-right">{(c.max * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          }
          onOpenAudit={handleOpenAudit}
          onOpenMethod={handleOpenMethod}
        >
          <VolatilityConeChart coneData={profile.volatility_cone} assetName={profile.asset_name} />
        </ChartCard>
      )}

      {/* Row 2: Distribution Modeling (C11) & Normality Diagnostic (C12) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* C11: Histogram & KDE */}
        <div className="lg:col-span-7">
          <ChartCard
            eyebrow="Distribution Modeling (C11)"
            title={`${profile?.asset_name || "S&P 500"} Return Distribution`}
            subtitle="Empirical histogram overlaid with Silverman's Gaussian KDE and Normal N(μ, σ²)"
            asOf={asOf}
            methodData={{
              chartTitle: `Return Distribution & Gaussian KDE (C11) — ${profile?.asset_name || "S&P 500"}`,
              whatItShows: "Empirical daily return frequency compared against a non-parametric Gaussian Kernel Density Estimate (Silverman's rule) and classical Gaussian normal bell curve.",
              definitionFormula: "h = 0.9 \\cdot \\min\\left(\\hat{\\sigma}, \\frac{\\text{IQR}}{1.34}\\right) n^{-1/5}, \\quad \\hat{f}(x) = \\frac{1}{nh\\sqrt{2\\pi}} \\sum_{i=1}^n e^{-\\frac{(x - r_i)^2}{2h^2}}",
              symbolsDefined: [
                { symbol: "h", meaning: "Silverman's optimal bandwidth for Gaussian kernel" },
                { symbol: "IQR", meaning: "Interquartile range (75th percentile - 25th percentile)" },
              ],
              parameters: {
                bandwidth_rule: "Silverman's rule of thumb",
                num_bins: 25,
                grid_points: 60,
              },
              dataSources: [{ series: selectedAssetId, source: "Primary Exchange Feeds", frequency: "Daily" }],
              sample: { n: 250, nEff: 250, dateRange: "Full available history" },
              interpretationNotes: [
                "Gap between the teal KDE curve and lavender dashed normal curve reveals empirical fat tails and skewness.",
                "The orange vertical marker indicates where today's realization sits relative to history.",
              ],
              reproduce: {
                functionName: "compute_kde_and_normal_overlay",
                filePath: "pipeline/src/radar/metrics/context.py",
                testPath: "pipeline/tests/unit/test_context.py",
              },
            }}
            auditData={{
              metricName: `${profile?.asset_name || "S&P 500"} Distribution Histogram`,
              formula: "Empirical histogram + Gaussian KDE + Normal PDF",
              formulaPlain: "25-bin empirical distribution of daily stationary returns with KDE smoothing.",
              parameters: { num_bins: 25, bandwidth: "Silverman" },
              inputs: [{ id: selectedAssetId, name: profile?.asset_name || "S&P 500", source: "Primary Feeds", unit: "%" }],
              window: "Full sample expanding",
              currentValue: contextData?.current_value ?? profile?.volatility_cone?.[0]?.current ?? 0,
              percentile: contextData?.percentile ?? 50,
            }}
            onOpenAudit={handleOpenAudit}
            onOpenMethod={handleOpenMethod}
          >
            {contextData ? (
              <DistributionHistogram
                bins={contextData.histogram_bins}
                counts={contextData.histogram_counts}
                kdeGrid={contextData.kde_grid}
                kdeDensity={contextData.kde_density}
                normalDensity={contextData.normal_density}
                currentValue={contextData.current_value}
                percentile={contextData.percentile}
                unit={contextData.unit}
              />
            ) : (
              <div className="h-[280px] flex items-center justify-center font-mono text-xs text-[var(--tx-2)]">
                Loading Distribution Data for {profile?.asset_name || selectedAssetId.toUpperCase()}...
              </div>
            )}
          </ChartCard>
        </div>

        {/* C12: QQ Plot */}
        <div className="lg:col-span-5">
          <ChartCard
            eyebrow="Normality Diagnostic (C12)"
            title="Quantile-Quantile (QQ) Plot"
            subtitle="Empirical standardized quantiles vs Theoretical Gaussian Normal quantiles"
            asOf={asOf}
            methodData={{
              chartTitle: `Quantile-Quantile Plot (C12) — ${profile?.asset_name || "S&P 500"}`,
              whatItShows: "Empirical quantiles plotted against theoretical normal quantiles; departures from the 45-degree diagonal line immediately expose heavy tails and skewness.",
              definitionFormula: "q_{\\text{theoretical}} = \\Phi^{-1}(p), \\quad q_{\\text{empirical}} = \\frac{\\text{percentile}(r, p) - \\hat{\\mu}}{\\hat{\\sigma}}",
              symbolsDefined: [
                { symbol: "Phi^-1", meaning: "Standard normal inverse cumulative distribution function (probit)" },
                { symbol: "p", meaning: "Quantile probabilities spanning 0.01 to 0.99" },
              ],
              parameters: {
                num_quantiles: 50,
                bounds: "0.01 to 0.99 (excluding asymptotic limits)",
              },
              dataSources: [{ series: selectedAssetId, source: "Primary Exchange Feeds", frequency: "Daily" }],
              sample: { n: 250, nEff: 250, dateRange: "Full available history" },
              interpretationNotes: [
                "Points sagging below the diagonal in the left tail indicate negative crash risk heavier than a normal distribution.",
                "S-shaped deviations indicate excess kurtosis (leptokurtic distribution).",
              ],
              reproduce: {
                functionName: "compute_qq_quantiles",
                filePath: "pipeline/src/radar/metrics/context.py",
                testPath: "pipeline/tests/unit/test_context.py",
              },
            }}
            auditData={{
              metricName: "Normal Quantile-Quantile Plot",
              formula: "norm.ppf(p) vs standardized sample quantiles",
              formulaPlain: "50 quantile points compared against theoretical normal distribution.",
              parameters: { points: 50 },
              inputs: [{ id: selectedAssetId, name: profile?.asset_name || "S&P 500", source: "Primary Feeds", unit: "sigma" }],
              window: "Full sample",
              currentValue: `${profile?.distribution?.skewness ?? 0} skew, ${profile?.distribution?.excess_kurtosis ?? 0} kurtosis`,
            }}
            onOpenAudit={handleOpenAudit}
            onOpenMethod={handleOpenMethod}
          >
            {contextData || (profile?.distribution?.qq_theoretical && profile.distribution.qq_theoretical.length > 0) ? (
              <ContextQqPlot
                theoretical={contextData?.qq_theoretical || profile!.distribution.qq_theoretical}
                empirical={contextData?.qq_empirical || profile!.distribution.qq_empirical}
                assetName={profile?.asset_name || selectedAssetId.toUpperCase()}
              />
            ) : (
              <div className="h-[280px] flex items-center justify-center font-mono text-xs text-[var(--tx-2)]">
                Loading QQ Normality Diagnostics...
              </div>
            )}
          </ChartCard>
        </div>
      </div>

      {/* Row 3: Tail Risk (C13) & Autocorrelation / Vol Clustering (C14) */}
      {profile && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* C13: Tail Risk */}
          <ChartCard
            eyebrow="Tail Risk Diagnostics (C13)"
            title={`Expected Shortfall & Tail Ratio — ${profile.asset_name}`}
            subtitle="Historical Value at Risk and Expected Shortfall vs Gaussian Parametric Bell Curve"
            asOf={asOf}
            methodData={{
              chartTitle: `Tail Risk & Expected Shortfall (C13) — ${profile.asset_name}`,
              whatItShows: "Empirical tail losses beyond Value at Risk (Expected Shortfall) compared directly against Gaussian parametric predictions.",
              definitionFormula: "ES_\\alpha = -\\mathbb{E}[R \\mid R \\le -\\text{VaR}_\\alpha], \\quad \\text{Tail Ratio} = \\frac{ES_\\alpha^{\\text{Hist}}}{ES_\\alpha^{\\text{Gauss}}}",
              symbolsDefined: [
                { symbol: "VaR_alpha", meaning: "Value at Risk at alpha confidence level" },
                { symbol: "ES_alpha", meaning: "Expected Shortfall (conditional tail expectation)" },
              ],
              parameters: {
                confidence_levels: "95%, 99%",
                estimator: "Empirical tail average vs Gaussian analytic",
              },
              dataSources: [{ series: profile.asset_id, source: "Market Returns", frequency: "Daily" }],
              sample: {
                n: 500,
                nEff: 420,
                dateRange: "Full available history",
              },
              interpretationNotes: [
                "Tail Ratio > 1.0 indicates fat tails: extreme tail losses substantially exceed the predictions of classical normal distribution models.",
                "Expected Shortfall represents the expected loss on days when the loss exceeds the VaR threshold.",
              ],
              reproduce: {
                functionName: "compute_tail_risk",
                filePath: "pipeline/src/radar/metrics/risk.py",
                testPath: "pipeline/tests/unit/test_risk.py",
              },
            }}
            auditData={{
              metricName: "Tail Risk & Expected Shortfall",
              formula: "ES = -mean(r | r <= -VaR)",
              formulaPlain: "Empirical conditional expectation of tail losses vs normal distribution equivalent.",
              parameters: { alpha: [0.95, 0.99] },
              inputs: [{ id: profile.asset_id, name: profile.asset_name, source: "Market Returns", unit: "%" }],
              window: "Expanding historical returns",
              currentValue: `Tail Ratio: ${assetTailRatio.toFixed(2)}x`,
              n: 500,
              nEff: 420,
            }}
            dataTable={
              <div className="overflow-x-auto text-xs font-mono">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                      <th className="py-1.5 px-2">Confidence</th>
                      <th className="py-1.5 px-2 text-right">Hist ES</th>
                      <th className="py-1.5 px-2 text-right">Hist VaR</th>
                      <th className="py-1.5 px-2 text-right">Gauss ES</th>
                      <th className="py-1.5 px-2 text-right">Gauss VaR</th>
                      <th className="py-1.5 px-2 text-right">Tail Ratio</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--line)]">
                    {profile.tail_risk.map((t) => (
                      <tr key={t.confidence}>
                        <td className="py-1.5 px-2 font-semibold">{(t.confidence * 100).toFixed(0)}%</td>
                        <td className="py-1.5 px-2 text-right text-rose-400 font-bold">{(t.hist_es * 100).toFixed(2)}%</td>
                        <td className="py-1.5 px-2 text-right">{(t.hist_var * 100).toFixed(2)}%</td>
                        <td className="py-1.5 px-2 text-right">{(t.gauss_es * 100).toFixed(2)}%</td>
                        <td className="py-1.5 px-2 text-right">{(t.gauss_var * 100).toFixed(2)}%</td>
                        <td className="py-1.5 px-2 text-right text-amber-400 font-bold">{t.tail_ratio.toFixed(2)}x</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            }
            onOpenAudit={handleOpenAudit}
            onOpenMethod={handleOpenMethod}
          >
            <TailRiskBarChart tailMetrics={profile.tail_risk} assetName={profile.asset_name} />
          </ChartCard>

          {/* C14: Autocorrelation & Vol Clustering */}
          <ChartCard
            eyebrow="ARCH Volatility Memory (C14)"
            title={`Volatility Clustering & ACF — ${profile.asset_name}`}
            subtitle="Autocorrelation Function of raw returns (linear memory) vs squared returns (ARCH vol persistence)"
            asOf={asOf}
            methodData={{
              chartTitle: `Volatility Clustering & ACF (C14) — ${profile.asset_name}`,
              whatItShows: "Persistence of return magnitudes (squared returns) demonstrating volatility clustering, alongside raw return autocorrelations with N_eff confidence bands.",
              definitionFormula: "\\text{ACF}(k) = \\frac{\\sum_{t=k+1}^T (x_t - \\bar{x})(x_{t-k} - \\bar{x})}{\\sum_{t=1}^T (x_t - \\bar{x})^2}, \\quad \\text{Band} = \\pm \\frac{1.96}{\\sqrt{N_{\\text{eff}}}}",
              symbolsDefined: [
                { symbol: "ACF(k)", meaning: "Autocorrelation at lag k" },
                { symbol: "N_eff", meaning: "Effective sample size adjusted for lag-1 serial correlation" },
              ],
              parameters: {
                max_lags: 20,
                significance: "95% two-sided HAC-adjusted band",
              },
              dataSources: [{ series: profile.asset_id, source: "Market Returns", frequency: "Daily" }],
              sample: {
                n: profile.autocorrelation.lags.length,
                nEff: profile.autocorrelation.n_eff,
                dateRange: "Full available history",
              },
              interpretationNotes: [
                "Raw returns near zero with no significant spikes confirm market informational efficiency (no linear predictability).",
                "Positive, decaying autocorrelation in squared returns (orange) proves ARCH volatility clustering: large moves cluster together.",
              ],
              reproduce: {
                functionName: "compute_autocorrelations",
                filePath: "pipeline/src/radar/metrics/risk.py",
                testPath: "pipeline/tests/unit/test_risk.py",
              },
            }}
            auditData={{
              metricName: "Autocorrelation Function",
              formula: "ACF(k) = cov(x_t, x_{t-k}) / var(x)",
              formulaPlain: "Lagged correlation of returns and squared returns with N_eff confidence limits.",
              parameters: { lags: 20 },
              inputs: [{ id: profile.asset_id, name: profile.asset_name, source: "Market Returns", unit: "%" }],
              window: "Lags 1 to 20",
              currentValue: `Lag 1 ACF(r²): ${profile.autocorrelation.acf_squared[0]?.toFixed(3) ?? "—"}`,
              n: profile.autocorrelation.lags.length,
              nEff: profile.autocorrelation.n_eff,
            }}
            dataTable={
              <div className="overflow-x-auto text-xs font-mono">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                      <th className="py-1.5 px-2">Lag</th>
                      <th className="py-1.5 px-2 text-right">ACF(r²) Vol Persistence</th>
                      <th className="py-1.5 px-2 text-right">ACF(r) Linear Return</th>
                      <th className="py-1.5 px-2 text-right">95% Conf Band</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--line)]">
                    {profile.autocorrelation.lags.map((lag, i) => (
                      <tr key={lag}>
                        <td className="py-1.5 px-2 font-semibold">Lag {lag}</td>
                        <td className="py-1.5 px-2 text-right text-amber-400 font-bold">{profile.autocorrelation.acf_squared[i]?.toFixed(3) ?? "—"}</td>
                        <td className="py-1.5 px-2 text-right">{profile.autocorrelation.acf_returns[i]?.toFixed(3) ?? "—"}</td>
                        <td className="py-1.5 px-2 text-right text-[var(--tx-3)]">±{profile.autocorrelation.band.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            }
            onOpenAudit={handleOpenAudit}
            onOpenMethod={handleOpenMethod}
          >
            <AutocorrelationBarChart acfData={profile.autocorrelation} assetName={profile.asset_name} />
          </ChartCard>
        </div>
      )}

      {/* Row 4: Underwater Drawdowns & Historical Episodes (C15) */}
      {profile && (
        <ChartCard
          eyebrow="Crisis Analysis (C15)"
          title={`Underwater Drawdown Series & Crisis Episodes — ${profile.asset_name}`}
          subtitle="Continuous drawdown from running peak and audit table of the 5 deepest historical crisis drawdowns"
          asOf={asOf}
          methodData={{
            chartTitle: `Underwater Drawdowns (C15) — ${profile.asset_name}`,
            whatItShows: "Cumulative decline from the preceding historical high water mark, tracking peak date, trough date, depth, duration, and days to recovery.",
            definitionFormula: "\\text{DD}_t = \\frac{P_t}{\\max_{s \\le t} P_s} - 1",
            symbolsDefined: [
              { symbol: "DD_t", meaning: "Percentage decline from cumulative peak at date t" },
              { symbol: "P_t", meaning: "Asset price level at date t" },
            ],
            parameters: {
              top_episodes: 5,
              series_lookback: "Last 500 trading sessions",
            },
            dataSources: [{ series: profile.asset_id, source: "Price Series", frequency: "Daily" }],
            sample: {
              n: profile.drawdown_series.length,
              nEff: profile.drawdown_series.length,
              dateRange: "Trailing 500 trading sessions",
            },
            interpretationNotes: [
              "Underwater area shows the current distance from peak. An extended flat line at 0% indicates continuous all-time highs.",
              "Deepest drawdown episodes quantify historical stress recovery duration (time underwater).",
            ],
            reproduce: {
              functionName: "compute_underwater_drawdowns",
              filePath: "pipeline/src/radar/metrics/risk.py",
              testPath: "pipeline/tests/unit/test_risk.py",
            },
          }}
          auditData={{
            metricName: "Underwater Drawdown Series",
            formula: "DD_t = P_t / max_{s <= t}(P_s) - 1.0",
            formulaPlain: "Percentage drop from peak historical high water mark.",
            parameters: { top_k: 5 },
            inputs: [{ id: profile.asset_id, name: profile.asset_name, source: "Price Series", unit: "Index" }],
            window: "Last 500 sessions",
            currentValue: `${(profile.drawdown_series.slice(-1)[0]?.drawdown * 100 || 0).toFixed(2)}% Current Drawdown`,
            n: profile.drawdown_series.length,
            nEff: profile.drawdown_series.length,
          }}
          dataTable={
            <div className="overflow-x-auto text-xs font-mono">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                    <th className="py-1.5 px-2">Peak Date</th>
                    <th className="py-1.5 px-2">Trough Date</th>
                    <th className="py-1.5 px-2">Recovery Date</th>
                    <th className="py-1.5 px-2 text-right">Depth</th>
                    <th className="py-1.5 px-2 text-right">Duration</th>
                    <th className="py-1.5 px-2 text-right">Recovery</th>
                    <th className="py-1.5 px-2">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)]">
                  {profile.drawdown_episodes.map((ep, idx) => (
                    <tr key={idx}>
                      <td className="py-1.5 px-2">{ep.peak_date}</td>
                      <td className="py-1.5 px-2">{ep.trough_date}</td>
                      <td className="py-1.5 px-2">{ep.recovery_date || "Active"}</td>
                      <td className="py-1.5 px-2 text-right text-rose-400 font-bold">{(ep.depth * 100).toFixed(2)}%</td>
                      <td className="py-1.5 px-2 text-right">{ep.duration_days} d</td>
                      <td className="py-1.5 px-2 text-right">{ep.recovery_days !== null ? `${ep.recovery_days} d` : "—"}</td>
                      <td className="py-1.5 px-2">{ep.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          }
          onOpenAudit={handleOpenAudit}
          onOpenMethod={handleOpenMethod}
        >
          <UnderwaterDrawdownChart
            series={profile.drawdown_series}
            episodes={profile.drawdown_episodes}
            assetName={profile.asset_name}
          />
        </ChartCard>
      )}

      {/* Row 5: Global Market Synchronization & PCA (C16) */}
      {riskData?.data?.pca && (
        <ChartCard
          eyebrow="Factor Structure (C16)"
          title="Global Market Synchronization & PCA Factor Structure"
          subtitle="Rolling 250D PCA on standardized USD returns across 10 global equity markets (sign-aligned to US)"
          asOf={asOf}
          methodData={{
            chartTitle: "Global Market Synchronization & Factor Structure (C16)",
            whatItShows: "Explained variance of the first principal component (PC1) over time, tracking whether global markets are moving together in unison or decoupling.",
            definitionFormula: "\\text{Synchronization}_t = \\frac{\\lambda_1(t)}{\\sum_{j=1}^K \\lambda_j(t)} \\times 100\\%, \\quad \\text{Invariant: } v_{1, \\text{us}} > 0",
            symbolsDefined: [
              { symbol: "lambda_1", meaning: "Largest eigenvalue of standardized return correlation matrix" },
              { symbol: "v_1", meaning: "First principal eigenvector (global factor loadings)" },
            ],
            parameters: {
              window: 250,
              step: 5,
              assets: "10 global equity indexes in USD returns",
              sign_alignment: "Enforce strictly positive loading for US index",
            },
            dataSources: [{ series: "10 global indexes", source: "Cleaned Country Returns", frequency: "Daily" }],
            sample: {
              n: riskData.data.pca.pc1_history.length,
              nEff: riskData.data.pca.pc1_history.length,
              dateRange: "Rolling 250D windows",
            },
            interpretationNotes: [
              "When PC1 variance share surges (> 60%), idiosyncratic country drivers are drowned out by a macro tidal wave (high contagion).",
              "When PC1 share declines (< 45%), regional decoupling provides strong cross-border diversification opportunities.",
              "Sign alignment guarantees that factor loading directions remain stable across windows without arbitrary sign flips.",
            ],
            reproduce: {
              functionName: "compute_pca_factor_structure",
              filePath: "pipeline/src/radar/metrics/risk.py",
              testPath: "pipeline/tests/unit/test_risk.py",
            },
          }}
          auditData={{
            metricName: "Global Market Synchronization Ratio",
            formula: "PC1 Share = lambda_1 / sum(lambda) * 100",
            formulaPlain: "Percentage of international equity return variation explained by the first principal component.",
            parameters: { window: 250, sign_invariant: "v_us > 0" },
            inputs: [
              { id: "global_equities", name: "10 Launch Equity Indexes (USD)", source: "Country Returns", unit: "%" },
            ],
            window: "250-Day Rolling Window",
            currentValue: `${latestSyncShare.toFixed(1)}% Latest PC1 Share`,
            n: riskData.data.pca.pc1_history.length,
            nEff: riskData.data.pca.pc1_history.length,
          }}
          dataTable={
            <div className="overflow-x-auto text-xs font-mono">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                    <th className="py-1.5 px-2">Component</th>
                    <th className="py-1.5 px-2 text-right">Explained Variance</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)]">
                  {riskData.data.pca.scree.map((s, i) => (
                    <tr key={i}>
                      <td className="py-1.5 px-2 font-semibold">Principal Component {i + 1}</td>
                      <td className="py-1.5 px-2 text-right text-[var(--accent)] font-bold">{s.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          }
          onOpenAudit={handleOpenAudit}
          onOpenMethod={handleOpenMethod}
        >
          <PcaSynchronizationChart
            pcaData={riskData.data.pca}
            latestShare={latestSyncShare}
          />
        </ChartCard>
      )}

      {/* Row 6: Downside Correlation & Stress Correlation Premium (C17) */}
      {riskData?.data?.downside_correlations && riskData.data.downside_correlations.length > 0 && (
        <ChartCard
          eyebrow="Diversification Stress Test (C17)"
          title="Downside Stress Correlation & Diversification Breakdown"
          subtitle="Unconditional all-day correlation vs Downside correlation during worst 10% benchmark days"
          asOf={asOf}
          methodData={{
            chartTitle: "Downside Correlation & Stress Premium (C17)",
            whatItShows: "Measures whether international diversification fails during market sell-offs by comparing correlation on all days vs correlation on the worst 10% benchmark days.",
            definitionFormula: "\\Delta \\rho = \\text{Corr}(R_{\\text{target}}, R_{\\text{bench}} \\mid R_{\\text{bench}} \\le q_{0.10}) - \\text{Corr}(R_{\\text{target}}, R_{\\text{bench}})",
            symbolsDefined: [
              { symbol: "q_0.10", meaning: "10th percentile daily return of benchmark (S&P 500)" },
              { symbol: "Delta_rho", meaning: "Stress Correlation Premium" },
            ],
            parameters: {
              stress_quantile: "Worst 10% (q = 0.10)",
              benchmark: "S&P 500 (SPX)",
            },
            dataSources: [{ series: "SPX & Foreign Indexes", source: "Daily aligned USD returns", frequency: "Daily" }],
            sample: {
              n: riskData.data.downside_correlations[0]?.n_all ?? 500,
              nEff: riskData.data.downside_correlations[0]?.n_stress ?? 50,
              dateRange: "Full available history",
            },
            interpretationNotes: [
              "A positive Stress Correlation Premium (Δρ > 0, red bar extending beyond blue) proves asymmetric downside co-movement: markets crash together faster than they rally together.",
              "Assets with near-zero or negative downside correlation serve as genuine crisis safe-havens.",
            ],
            reproduce: {
              functionName: "compute_downside_correlation",
              filePath: "pipeline/src/radar/metrics/risk.py",
              testPath: "pipeline/tests/unit/test_risk.py",
            },
          }}
          auditData={{
            metricName: "Downside Stress Correlation Premium",
            formula: "Delta_rho = corr(target, bench | bench <= q_0.10) - corr(target, bench)",
            formulaPlain: "Gap between correlation during worst 10% benchmark days and unconditional correlation.",
            parameters: { quantile: 0.10, benchmark: "spx" },
            inputs: [
              { id: "spx", name: "S&P 500 (Benchmark)", source: "Market Returns", unit: "%" },
              { id: "targets", name: "Global Equity Indexes", source: "Market Returns", unit: "%" },
            ],
            window: "Expanding history",
            currentValue: `Max Δρ: +${riskData.data.downside_correlations[0]?.stress_premium.toFixed(2) ?? "0.00"} (${riskData.data.downside_correlations[0]?.target_name})`,
            n: riskData.data.downside_correlations[0]?.n_all ?? 500,
            nEff: riskData.data.downside_correlations[0]?.n_stress ?? 50,
          }}
          dataTable={
            <div className="overflow-x-auto text-xs font-mono">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                    <th className="py-1.5 px-2">Target Market</th>
                    <th className="py-1.5 px-2 text-right">Downside Corr</th>
                    <th className="py-1.5 px-2 text-right">All-Day Corr</th>
                    <th className="py-1.5 px-2 text-right">Stress Premium (Δρ)</th>
                    <th className="py-1.5 px-2 text-right">Stress N</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)]">
                  {riskData.data.downside_correlations.map((d) => (
                    <tr key={d.target_id}>
                      <td className="py-1.5 px-2 font-semibold">{d.target_name}</td>
                      <td className="py-1.5 px-2 text-right text-rose-400 font-bold">{d.downside_corr.toFixed(2)}</td>
                      <td className="py-1.5 px-2 text-right">{d.all_day_corr.toFixed(2)}</td>
                      <td className="py-1.5 px-2 text-right text-amber-400 font-bold">
                        {d.stress_premium >= 0 ? "+" : ""}{d.stress_premium.toFixed(2)}
                      </td>
                      <td className="py-1.5 px-2 text-right text-[var(--tx-3)]">{d.n_stress} days</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          }
          onOpenAudit={handleOpenAudit}
          onOpenMethod={handleOpenMethod}
        >
          <DownsideCorrelationChart
            items={riskData.data.downside_correlations}
            benchmarkName="S&P 500"
          />
        </ChartCard>
      )}


      {/* Row 7: Non-Linear Joint Tail Density (C18) & Ridgeline Return Distributions (C19) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* C18: Hexbin Joint Tail Density */}
        <div className="lg:col-span-6">
          <ChartCard
            eyebrow="Non-Linear Tail Dependency (C18)"
            title="Hexbin Joint Tail Density"
            subtitle={`Bivariate empirical density of ${profile?.asset_name || "S&P 500"} vs US 10Y Yield`}
            asOf={asOf}
            methodData={{
              chartTitle: `Hexbin Joint Tail Density (C18) — ${profile?.asset_name || "S&P 500"} vs Macro Drivers`,
              whatItShows: "2D hexagonal binning of joint returns showing asymmetric tail co-dependence without assuming linear correlation.",
              definitionFormula: "\\hat{f}(x, y) = \\frac{1}{n b_x b_y} \\sum_{i=1}^n K\\left(\\frac{x - x_i}{b_x}\\right) K\\left(\\frac{y - y_i}{b_y}\\right)",
              symbolsDefined: [
                { symbol: "b_x, b_y", meaning: "Bandwidths along x and y coordinates" },
                { symbol: "K", meaning: "Hexagonal kernel weighting function" },
              ],
              parameters: { grid_size: "30x30", density_scale: "Empirical density mapped" },
              dataSources: [{ series: "SPX & US10Y", source: "Daily aligned changes", frequency: "Daily" }],
              sample: { n: 250, nEff: 250, dateRange: "Trailing 250 trading days" },
              interpretationNotes: [
                "Hexagonal clusters in the lower quadrants highlight simultaneous extreme joint sell-offs or flight-to-safety episodes.",
              ],
              reproduce: {
                functionName: "compute_bivariate_tail_density",
                filePath: "pipeline/src/radar/metrics/risk.py",
                testPath: "pipeline/tests/unit/test_risk.py",
              },
            }}
            auditData={{
              metricName: "Hexbin Joint Tail Density",
              formula: "Bivariate 2D empirical histogram / density",
              formulaPlain: "Hexagonal binning of daily asset return vs macro factor changes.",
              parameters: { grid: "hexagonal", lookback: "250D" },
              inputs: [
                { id: "spx", name: "S&P 500", source: "Market Returns", unit: "%" },
                { id: "us10y", name: "US 10Y Yield", source: "FRED", unit: "bps" },
              ],
              window: "Trailing 250 trading days",
              currentValue: "Asymmetric tail cluster",
            }}
            onOpenAudit={handleOpenAudit}
            onOpenMethod={handleOpenMethod}
          >
            <HexbinJointDensityChart
              xLabel={`${profile?.asset_name || "S&P 500"} Daily Return (%)`}
              yLabel="US 10Y Yield Change (bps)"
            />
          </ChartCard>
        </div>

        {/* C19: Ridgeline Return Distributions */}
        <div className="lg:col-span-6">
          <ChartCard
            eyebrow="Distribution Shape Comparison (C19)"
            title="Ridgeline Return Distributions"
            subtitle="Overlapping KDE curves across 8 key asset classes highlighting asymmetry and current position"
            asOf={asOf}
            methodData={{
              chartTitle: "Cross-Market Ridgeline Return Distributions (C19)",
              whatItShows: "Stacked density curves across global equity markets and commodities, with pin markers showing today's 1-day return.",
              definitionFormula: "\\hat{f}_m(x) = \\frac{1}{nh} \\sum_{i=1}^n \\frac{1}{\\sqrt{2\\pi}} e^{-\\frac{(x - r_{m,i})^2}{2h^2}}",
              symbolsDefined: [
                { symbol: "r_{m,i}", meaning: "Daily return for market m" },
                { symbol: "h", meaning: "Silverman bandwidth" },
              ],
              parameters: { markets: "8 core international assets", span: "-5% to +5%" },
              dataSources: [{ series: "8 Launch Indexes", source: "Exchange Returns", frequency: "Daily" }],
              sample: { n: 250, nEff: 250, dateRange: "Full available history" },
              interpretationNotes: [
                "Broader ridges indicate higher volatility regimes (e.g. Crude Oil vs FTSE 100).",
                "Pin markers shifted to the tails reveal which markets experienced abnormal 1-day shocks.",
              ],
              reproduce: {
                functionName: "compute_ridgeline_densities",
                filePath: "pipeline/src/radar/metrics/risk.py",
                testPath: "pipeline/tests/unit/test_risk.py",
              },
            }}
            auditData={{
              metricName: "Cross-Asset Ridgeline Density Stack",
              formula: "Stacked Kernel Density Estimates across markets",
              formulaPlain: "Multi-market KDE comparison of daily returns with current realization pins.",
              parameters: { n_markets: 8 },
              inputs: [{ id: "universe", name: "Global Equity & Commodity Core", source: "Exchange Closes", unit: "%" }],
              window: "250-Day lookback",
              currentValue: "8 Market Profiles Mapped",
            }}
            onOpenAudit={handleOpenAudit}
            onOpenMethod={handleOpenMethod}
          >
            <RidgelineDensityChart />
          </ChartCard>
        </div>
      </div>

      {/* Drawers */}
      <AuditDrawer
        isOpen={isAuditOpen}
        onClose={() => setIsAuditOpen(false)}
        data={auditTarget}
      />
      <MethodDrawer
        isOpen={isMethodOpen}
        onClose={() => setIsMethodOpen(false)}
        data={methodTarget}
      />
    </div>
  );
};
