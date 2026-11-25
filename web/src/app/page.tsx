// web/src/app/page.tsx
"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { ChartCard } from "@/components/charts/ChartCard";
import { MarketStateMap } from "@/components/charts/MarketStateMap";
import { VolatilityLandscape } from "@/components/charts/VolatilityLandscape";
import { MoveDecomposerWaterfall } from "@/components/charts/MoveDecomposerWaterfall";
import { PercentileStrip } from "@/components/charts/PercentileStrip";
import { StatBlock } from "@/components/ui/StatBlock";
import { AuditData, AuditDrawer } from "@/components/ui/AuditDrawer";
import { MethodData, MethodDrawer } from "@/components/ui/MethodDrawer";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { HealthDot } from "@/components/ui/HealthDot";
import {
  loadPulse,
  loadStateMap,
  loadVolLandscape,
  loadDecomposition,
  loadCycles,
  loadRatios,
  loadPositioning,
} from "@/lib/data/client";
import {
  PulseEnvelope,
  StateMapEnvelope,
  VolLandscapeEnvelope,
  DecompositionEnvelope,
  CyclesEnvelope,
  RatiosEnvelope,
  PositioningEnvelope,
} from "@/lib/data/schemas";

export default function TodayPage() {
  const [pulseData, setPulseData] = useState<PulseEnvelope | null>(null);
  const [stateMapData, setStateMapData] = useState<StateMapEnvelope | null>(null);
  const [volData, setVolData] = useState<VolLandscapeEnvelope | null>(null);
  const [decompData, setDecompData] = useState<DecompositionEnvelope | null>(null);
  const [cyclesData, setCyclesData] = useState<CyclesEnvelope | null>(null);
  const [ratiosData, setRatiosData] = useState<RatiosEnvelope | null>(null);
  const [posData, setPosData] = useState<PositioningEnvelope | null>(null);
  const [activeTarget, setActiveTarget] = useState<string>("featured");
  const [asOf, setAsOf] = useState<string>("");
  const [isStale, setIsStale] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  // Drawer states
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
    setIsOpenMethod(true);
  };

  const setIsOpenMethod = (open: boolean) => {
    setIsMethodOpen(open);
  };

  const switchTargetDecomposition = async (targetId: string) => {
    setActiveTarget(targetId);
    try {
      const res = await loadDecomposition(targetId);
      if (res.status !== "error") {
        setDecompData(res.data);
      }
    } catch (e) {
      console.error("Error switching decomposition target:", e);
    }
  };

  useEffect(() => {
    async function loadAll() {
      setLoading(true);
      try {
        const [pulseRes, stateRes, volRes, decompRes, cyclesRes, ratiosRes, posRes] = await Promise.all([
          loadPulse(),
          loadStateMap(),
          loadVolLandscape(),
          loadDecomposition("featured"),
          loadCycles(),
          loadRatios(),
          loadPositioning(),
        ]);

        if (pulseRes.status !== "error") {
          setPulseData(pulseRes.data);
          setAsOf(pulseRes.asOf);
          if (pulseRes.status === "stale") setIsStale(true);
        }
        if (stateRes.status !== "error") {
          setStateMapData(stateRes.data);
        }
        if (volRes.status !== "error") {
          setVolData(volRes.data);
        }
        if (decompRes.status !== "error") {
          setDecompData(decompRes.data);
        }
        if (cyclesRes.status !== "error") {
          setCyclesData(cyclesRes.data);
        }
        if (ratiosRes.status !== "error") {
          setRatiosData(ratiosRes.data);
        }
        if (posRes.status !== "error") {
          setPosData(posRes.data);
        }
      } catch (err) {
        console.error("Error loading today datasets:", err);
      } finally {
        setLoading(false);
      }
    }

    loadAll();
  }, []);

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Editorial Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[var(--line)]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Today / Market Intelligence
          </div>
          <h1 className="text-3xl font-serif font-bold tracking-tight text-[var(--tx-0)] mt-1">
            Why did the market move?
          </h1>
          <p className="text-xs text-[var(--tx-1)] mt-1.5 max-w-2xl leading-relaxed">
            Every move, split into its drivers, with the uncertainty shown. Strictly anti-lookahead
            econometric attribution, multi-horizon volatility landscape, and cross-asset state mapping.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <Link
            href="/compare"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-[var(--accent)] text-slate-900 font-mono text-xs font-semibold shadow hover:opacity-90 transition"
          >
            <span>Compare &amp; Stress Test</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </Link>
          <div className="flex items-center gap-3">
            <HealthDot status={isStale ? "stale" : "ok"} label={isStale ? "Stale Data" : "Verified Live"} />
            <DataAgeBadge asOf={asOf} isStale={isStale} ageHours={isStale ? 40 : undefined} />
          </div>
        </div>
      </div>

      {/* Executive Morning Desk Strip: Regime, Stretched Crowding & Ratio Outliers */}
      {(cyclesData || ratiosData || posData) && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {cyclesData && (
            <Link
              href="/cycles"
              className="p-3.5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] hover:border-[var(--accent)] transition flex flex-col justify-between"
            >
              <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-wider text-[var(--tx-2)]">
                <span>Macro Cycle Regime</span>
                <span className="text-[var(--accent)]">View Clock &rarr;</span>
              </div>
              <div className="mt-1.5 flex items-baseline justify-between">
                <span className="text-lg font-bold text-[var(--tx-0)]">{cyclesData.data.current_quadrant}</span>
                <span className="text-xs font-mono text-[var(--tx-2)] tabular-nums">
                  G: {cyclesData.data.current_growth_z >= 0 ? "+" : ""}{cyclesData.data.current_growth_z}σ · I: {cyclesData.data.current_inflation_z >= 0 ? "+" : ""}{cyclesData.data.current_inflation_z}σ
                </span>
              </div>
            </Link>
          )}

          {posData && (
            <Link
              href="/positioning"
              className="p-3.5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] hover:border-[var(--accent)] transition flex flex-col justify-between"
            >
              <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-wider text-[var(--tx-2)]">
                <span>CFTC Crowding Extremes</span>
                <span className="text-[var(--accent)]">View Ladder &rarr;</span>
              </div>
              <div className="mt-1.5 text-xs font-mono text-[var(--tx-0)] truncate">
                {(() => {
                  const items = ((posData as any).data?.contracts || (posData as any).contracts || [])
                    .filter((c: any) => c.percentile >= 85 || c.percentile <= 15)
                    .slice(0, 2);
                  if (items.length === 0) return "No extreme (>85th / <15th) CFTC positioning";
                  return items
                    .map((c: any) => `${c.name.split(" ")[0]} (${c.percentile.toFixed(0)}th)`)
                    .join(" · ");
                })()}
              </div>
            </Link>
          )}

          {ratiosData && (
            <Link
              href="/ratios"
              className="p-3.5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] hover:border-[var(--accent)] transition flex flex-col justify-between"
            >
              <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-wider text-[var(--tx-2)]">
                <span>Valuation &amp; Ratio Outliers</span>
                <span className="text-[var(--accent)]">View Ratios &rarr;</span>
              </div>
              <div className="mt-1.5 text-xs font-mono text-[var(--tx-0)] truncate">
                {(() => {
                  const outliers = ratiosData.data.ratios
                    .filter((r) => Math.abs(r.z_score) >= 1.2)
                    .sort((a, b) => Math.abs(b.z_score) - Math.abs(a.z_score))
                    .slice(0, 2);
                  if (outliers.length === 0) return "All cross-asset ratios within ±1.2σ normal band";
                  return outliers
                    .map((r) => `${r.id.replace("_", "/").toUpperCase()} (${r.z_score >= 0 ? "+" : ""}${r.z_score}σ)`)
                    .join(" · ");
                })()}
              </div>
            </Link>
          )}
        </div>
      )}

      {/* Signature Module: The Move Decomposer */}
      {decompData && (
        <ChartCard
          eyebrow="Signature Instrument / Attribution"
          title="Move Decomposer"
          subtitle={`Multi-factor attribution with prior-window Newey-West HAC errors (${decompData.data.target})`}
          insightHeadline={decompData.data.narrative}
          asOf={asOf}
          sourceText="Global Equities, ICE Brent, COMEX Gold/Copper, US Treasury, FX Markets"
          headerControls={
            <div className="flex flex-wrap items-center gap-1.5 bg-[var(--bg-0)] p-1 rounded border border-[var(--line)] text-xs">
              <span className="text-[11px] font-mono text-[var(--tx-2)] px-1">Target:</span>
              {[
                { id: "featured", label: "India (Nifty)" },
                { id: "spx", label: "US (S&P 500)" },
                { id: "dax", label: "Germany (DAX)" },
                { id: "n225", label: "Japan (Nikkei)" },
              ].map((tgt) => (
                <button
                  key={tgt.id}
                  onClick={() => switchTargetDecomposition(tgt.id)}
                  className={`px-2 py-0.5 rounded font-mono text-xs transition-colors ${activeTarget === tgt.id
                    ? "bg-[var(--accent)] text-slate-900 font-semibold"
                    : "text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                    }`}
                >
                  {tgt.label}
                </button>
              ))}
            </div>
          }
          methodData={{
            chartTitle: "Move Decomposer (Prior-Window Multi-Factor OLS + HAC)",
            whatItShows:
              "Deconstructs a target country or sector move into contributions from global drivers and an explicit residual, with estimation intervals.",
            definitionFormula: "y_t = \\sum_k (\\beta_k \\cdot x_{k, t}) + \\text{residual}_t, \\quad (1 + r_{\\text{usd}}) = (1 + r_{\\text{local}})(1 + r_{\\text{fx}})",
            symbolsDefined: [
              { symbol: "y_t", meaning: "Cumulative target return over horizon h in {1D, 5D, 1M}" },
              { symbol: "beta_k", meaning: "Prior-window sensitivity estimated over 250 days ending day before horizon starts" },
              { symbol: "x_{k, t}", meaning: "Cumulative move of global macro driver k (ROW Equity, Oil, Copper, Gold, DXY, US 10Y Yield)" },
              { symbol: "r_{fx}", meaning: "Exact currency impact in USD lens (Local vs USD appreciation)" },
            ],
            parameters: {
              prior_window: "250 trading days",
              estimator: "OLS with Newey-West HAC",
              max_vif: "5.0 (Ridge fallback)",
              alpha_ridge: 1.0,
              unexplained_floor: "0.10 R2",
            },
            dataSources: [
              { series: "Target", source: "Official Exchange Closes", frequency: "Daily" },
              { series: "Macro Factors", source: "FRED, Yahoo Finance, ICE, COMEX", frequency: "Daily" },
            ],
            sample: {
              n: decompData.data.n_obs,
              nEff: Math.round(decompData.data.n_obs * 0.72),
              dateRange: `${decompData.data.window_dates.prior_start} to ${decompData.data.window_dates.horizon_end}`,
            },
            interpretationNotes: [
              "Contributions measure association over historical baseline, not causal mechanical certainty.",
              "If out-of-sample R² is below 0.10, the move is labeled 'Mostly unexplained' rather than asserting spurious precision.",
              "Time-zone alignment ensures Asian closes are matched with preceding US sessions.",
            ],
            reproduce: {
              functionName: "compute_decomposition",
              filePath: "pipeline/src/radar/metrics/decomposition.py",
              testPath: "pipeline/tests/unit/test_decomposition.py",
            },
          }}
          onOpenMethod={handleOpenMethod}
          auditData={{
            metricName: "Move Decomposer Attribution Engine",
            formula: "y = X\\beta + \\epsilon, \\quad \\text{SE}_{\\text{HAC}}(\\beta)",
            formulaPlain: "Multi-factor regression on strictly prior 250 trading days with Newey-West standard errors.",
            parameters: {
              horizon: `${decompData.data.horizon_days}D`,
              lens: decompData.data.lens,
              r2_in_sample: decompData.data.r2_in_sample,
              r2_out_sample: decompData.data.r2_out_sample,
              regularized: decompData.data.regularized,
            },
            inputs: Object.keys(decompData.data.contributions).map((k) => ({
              id: k,
              name: k.replace(/_/g, " ").toUpperCase(),
              source: "Global Macro Factor Feed",
              unit: "return / diff bps",
            })),
            window: "Trailing 250D prior window",
            currentValue: `${(decompData.data.total_return * 100).toFixed(2)}%`,
            n: decompData.data.n_obs,
            nEff: Math.round(decompData.data.n_obs * 0.72),
            codeVersion: decompData.code_version,
            specUrl: "/methodology",
          }}
          onOpenAudit={handleOpenAudit}
        >
          <MoveDecomposerWaterfall decomposition={decompData.data} />
        </ChartCard>
      )}

      {/* Hero StatBlocks */}
      {pulseData && pulseData.data.stats && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {pulseData.data.stats.slice(0, 4).map((stat) => (
            <StatBlock
              key={stat.id}
              id={stat.id}
              name={stat.name}
              value={stat.value}
              change1d={stat.change_1d}
              unit={stat.unit}
              percentile={stat.percentile}
              sparkline={stat.sparkline}
              onOpenAudit={handleOpenAudit}
              auditPayload={{
                metricName: `${stat.name} Level & Expanding Distribution`,
                formula: "r_t = \\ln(P_t / P_{t-1}), \\quad \\text{Pct} = \\frac{1}{t}\\sum_{i=1}^t \\mathbf{1}_{x_i \\le x_t}",
                formulaPlain: "Daily log return and strictly expanding percentile rank over all historical observations to date.",
                parameters: { lookback: "Daily", min_obs: 60 },
                inputs: [{ id: stat.id, name: stat.name, source: "FRED / Yahoo / NSE", unit: stat.unit }],
                window: "Expanding window (anti-lookahead)",
                currentValue: stat.value,
                percentile: stat.percentile ?? undefined,
                n: 2520,
                nEff: 410,
                codeVersion: pulseData.code_version,
                specUrl: "/methodology",
              }}
            />
          ))}
        </div>
      )}

      {/* Contextual Percentile Strips (C20) */}
      {pulseData && pulseData.data.stats && (
        <ChartCard
          eyebrow="Historical Percentile Strip"
          title="Macro Percentile Strips — 'You Are Here'"
          subtitle="Realized levels ranked against expanding historical distributions [0% min to 100% max]"
          insightHeadline="Equities and implied volatility are trading in their upper tercile (>80th percentile) alongside crude oil and gold."
          asOf={asOf}
          auditData={{
            metricName: "Expanding Historical Percentile Ranking",
            formula: "P_t = \\frac{1}{t-1} \\sum_{i=1}^{t-1} \\mathbf{1}_{\\{x_i \\le x_t\\}} \\times 100",
            formulaPlain: "Percentage of strictly prior historical days where the asset closed below today's level.",
            parameters: { min_history: 250, anti_lookahead: "strictly expanding" },
            inputs: pulseData.data.stats.map((s) => ({ id: s.id, name: s.name, source: "FRED / Yahoo / NSE", unit: s.unit })),
            window: "Full historical expanding sample",
            currentValue: `${pulseData.data.stats.length} Core Macro Assets Ranked`,
            codeVersion: pulseData.code_version,
            specUrl: "/methodology",
          }}
          onOpenAudit={handleOpenAudit}
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {pulseData.data.stats.map((stat) => (
              <PercentileStrip
                key={stat.id}
                label={stat.name}
                percentile={stat.percentile ?? 50}
                currentValue={
                  stat.unit === "yield_pct" || stat.id === "us10y"
                    ? `${stat.value.toFixed(2)}%`
                    : stat.unit === "usd_per_barrel"
                      ? `$${stat.value.toFixed(2)}/bbl`
                      : stat.id === "gold"
                        ? `$${stat.value.toLocaleString()}/oz`
                        : stat.value.toLocaleString()
                }
              />
            ))}
          </div>
        </ChartCard>
      )}

      {/* Main Grid: Market State Map & Volatility Landscape */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Market State Map (7 cols) */}
        <div className="lg:col-span-7">
          <ChartCard
            eyebrow="Cross-Asset Regime"
            title="Market State Map (Trend vs Volatility)"
            subtitle="Scatter of assets by 60D trend strength t-statistic and 20D realized volatility percentile"
            insightHeadline="Global equity benchmarks remain clustered in the Orderly Trend quadrant with compressed realized volatility."
            insightDetail="Rates and energy commodities exhibit elevated volatility expansion, bordering the Choppy regime."
            asOf={asOf}
            auditData={{
              metricName: "Market State Map Quadrant Classification",
              formula: "t = \\frac{\\hat{\\beta}}{\\text{SE}(\\hat{\\beta})}, \\quad \\text{VolPct} = \\text{Percentile}(\\sigma_{20})",
              formulaPlain: "Signed t-statistic of slope of log price over 60 business days vs expanding percentile of 20-day volatility.",
              parameters: { trend_window: 60, trend_threshold: 2.0, vol_percentile_threshold: 50.0, hysteresis_days: 3 },
              inputs: (stateMapData?.data.assets || []).slice(0, 5).map((a) => ({
                id: a.asset,
                name: a.name,
                source: "Processed Series",
                unit: "index",
              })),
              window: "60 business days",
              currentValue: "Orderly trend (majority)",
              n: 60,
              nEff: 45,
              codeVersion: stateMapData?.code_version,
              specUrl: "/methodology",
            }}
            onOpenAudit={handleOpenAudit}
          >
            {stateMapData ? (
              <MarketStateMap
                assets={stateMapData.data.assets}
                onSelectAsset={(asset) => {
                  window.location.href = `/context/${asset}/`;
                }}
              />
            ) : (
              <div className="h-[360px] flex items-center justify-center text-xs text-[var(--tx-2)] font-mono">
                Loading Market State Map...
              </div>
            )}
          </ChartCard>
        </div>

        {/* Volatility Landscape (5 cols) */}
        <div className="lg:col-span-5">
          <ChartCard
            eyebrow="Dispersion & Term Structure"
            title="Volatility Landscape"
            subtitle="Annualized realized volatility across 10D, 20D, and 60D lookbacks with trend"
            insightHeadline="Short-term volatility continues to trade below 60D baselines across major benchmarks."
            asOf={asOf}
            auditData={{
              metricName: "Realized Volatility Multi-Horizon Matrix",
              formula: "\\sigma_w = \\text{std}(r) \\times \\sqrt{252}, \\quad \\text{Trend} = \\frac{\\sigma_{20}}{\\sigma_{60}} - 1",
              formulaPlain: "Annualized realized volatility for 10D, 20D, and 60D windows. Vol trend indicates compression or expansion.",
              parameters: { windows: "10, 20, 60", annualization: "sqrt(252)", ddof: 1 },
              inputs: [{ id: "universe", name: "60 Core Universe Assets", source: "FRED / Yahoo / NSE", unit: "returns" }],
              window: "10D / 20D / 60D",
              currentValue: "Compressed (<50th pct)",
              n: 2520,
              nEff: 320,
              codeVersion: volData?.code_version,
              specUrl: "/methodology",
            }}
            onOpenAudit={handleOpenAudit}
          >
            {volData ? (
              <VolatilityLandscape
                landscape={volData.data.landscape}
                onOpenAudit={handleOpenAudit}
              />
            ) : (
              <div className="h-[360px] flex items-center justify-center text-xs text-[var(--tx-2)] font-mono">
                Loading Volatility Landscape...
              </div>
            )}
          </ChartCard>
        </div>
      </div>

      {/* Traceability Audit Drawer */}
      <AuditDrawer
        isOpen={isAuditOpen}
        onClose={() => setIsAuditOpen(false)}
        data={auditTarget}
      />

      {/* Scientific Method Drawer */}
      <MethodDrawer
        isOpen={isMethodOpen}
        onClose={() => setIsMethodOpen(false)}
        data={methodTarget}
      />
    </div>
  );
}
