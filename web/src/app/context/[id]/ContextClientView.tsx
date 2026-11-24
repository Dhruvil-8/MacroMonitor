// web/src/app/context/[id]/ContextClientView.tsx
"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Calendar, ShieldCheck } from "lucide-react";
import { ChartCard } from "@/components/charts/ChartCard";
import { DistributionHistogram } from "@/components/charts/DistributionHistogram";
import { ContextQqPlot } from "@/components/charts/ContextQqPlot";
import { CalendarHeatmap } from "@/components/charts/CalendarHeatmap";
import { StatBlock } from "@/components/ui/StatBlock";
import { AuditData, AuditDrawer } from "@/components/ui/AuditDrawer";
import { MethodData, MethodDrawer } from "@/components/ui/MethodDrawer";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { HealthDot } from "@/components/ui/HealthDot";
import { loadContext } from "@/lib/data/client";
import { ContextData } from "@/lib/data/schemas";

export function ContextClientView({ id }: { id: string }) {
  const [ctxData, setCtxData] = useState<ContextData | null>(null);
  const [asOf, setAsOf] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);

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
    async function load() {
      setLoading(true);
      try {
        const res = await loadContext(id);
        if (res.status !== "error") {
          setCtxData(res.data.data);
          setAsOf(res.asOf);
        }
      } catch (err) {
        console.error("Failed to load context for", id, err);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [id]);

  const jsonLd = ctxData
    ? {
      "@context": "https://schema.org",
      "@type": "Dataset",
      name: `${ctxData.name} Historical Context & Empirical Distribution`,
      description: ctxData.description,
      identifier: ctxData.id,
      isAccessibleForFree: true,
      temporalCoverage: "2000-01-01/..",
      spatialCoverage: ctxData.region,
      creator: {
        "@type": "Organization",
        name: "MacroMonitor",
      },
    }
    : null;

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {jsonLd && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      )}

      {/* Back Link & Header */}
      <div>
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-xs font-mono text-[var(--tx-2)] hover:text-[var(--accent)] transition-colors mb-3"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Today Overview</span>
        </Link>

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-[var(--line)]">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--accent)] font-medium">
              Section 4.5 • Historical Distribution & Regime Context
            </div>
            <h1 className="text-2xl sm:text-3xl font-serif text-[var(--tx-0)] tracking-tight mt-1">
              {ctxData?.name || id.toUpperCase()}{" "}
              <span className="text-xs font-mono font-normal text-[var(--accent)] px-2 py-0.5 rounded bg-[var(--bg-2)] border border-[var(--line)]">
                {id.toUpperCase()}
              </span>
            </h1>
            <p className="text-sm text-[var(--tx-2)] max-w-2xl mt-1">
              Stationary return distributions, Gaussian KDE overlays, tail quantile departures, calendar seasonality, and forward returns following declustered historical analogs.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <HealthDot status="healthy" />
            <DataAgeBadge asOf={asOf} />
          </div>
        </div>
      </div>

      {/* Stat Blocks */}
      {ctxData && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatBlock
            id="current-level"
            name="Current Level"
            value={ctxData.current_value.toLocaleString()}
            unit={ctxData.unit}
          />
          <StatBlock
            id="hist-percentile"
            name="Expanding Percentile"
            value={`${ctxData.percentile.toFixed(0)}th`}
            unit="Historical Rank"
            percentile={ctxData.percentile}
          />
          <StatBlock
            id="expanding-zscore"
            name="Expanding Z-Score"
            value={`${ctxData.z_score > 0 ? "+" : ""}${ctxData.z_score.toFixed(2)}σ`}
            unit="Std Deviations"
          />
          <StatBlock
            id="effective-n"
            name="Effective Sample (N_eff)"
            value={ctxData.n_eff.toLocaleString()}
            unit={`of ${ctxData.n} Raw Days`}
          />
        </div>
      )}

      {/* Charts Grid */}
      {ctxData ? (
        <div className="space-y-6">
          {/* Row 1: Histogram with KDE (C11) & QQ Plot (C12) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* C11: Histogram & KDE */}
            <div className="lg:col-span-7">
              <ChartCard
                eyebrow="Distribution Modeling (C11)"
                title={`${ctxData.name} Return Distribution`}
                subtitle="Empirical histogram overlaid with Silverman's Gaussian KDE and Normal N(μ, σ²)"
                asOf={asOf}
                methodData={{
                  chartTitle: `Return Distribution & Gaussian KDE (C11) — ${ctxData.name}`,
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
                  dataSources: [{ series: ctxData.id, source: ctxData.source_note, frequency: "Daily" }],
                  sample: {
                    n: ctxData.n,
                    nEff: ctxData.n_eff,
                    dateRange: "Full available history",
                  },
                  interpretationNotes: [
                    "Gap between the teal KDE curve and lavender dashed normal curve reveals empirical fat tails and skewness.",
                    "The orange vertical marker indicates where today's return sits relative to history.",
                  ],
                  reproduce: {
                    functionName: "compute_kde_and_normal_overlay",
                    filePath: "pipeline/src/radar/metrics/context.py",
                    testPath: "pipeline/tests/unit/test_context.py",
                  },
                }}
                auditData={{
                  metricName: `${ctxData.name} Distribution Histogram`,
                  formula: "Empirical histogram + Gaussian KDE + Normal PDF",
                  formulaPlain: "25-bin empirical distribution of daily stationary returns with KDE smoothing.",
                  parameters: { num_bins: 25, bandwidth: "Silverman" },
                  inputs: [{ id: ctxData.id, name: ctxData.name, source: ctxData.source_note, unit: ctxData.unit }],
                  window: "Full sample expanding",
                  currentValue: ctxData.current_value,
                  percentile: ctxData.percentile,
                  n: ctxData.n,
                  nEff: ctxData.n_eff,
                }}
                onOpenAudit={handleOpenAudit}
                onOpenMethod={handleOpenMethod}
              >
                <DistributionHistogram
                  bins={ctxData.histogram_bins}
                  counts={ctxData.histogram_counts}
                  kdeGrid={ctxData.kde_grid}
                  kdeDensity={ctxData.kde_density}
                  normalDensity={ctxData.normal_density}
                  currentValue={ctxData.current_value}
                  percentile={ctxData.percentile}
                  unit={ctxData.unit}
                />
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
                  chartTitle: `Quantile-Quantile Plot (C12) — ${ctxData.name}`,
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
                  dataSources: [{ series: ctxData.id, source: ctxData.source_note, frequency: "Daily" }],
                  sample: {
                    n: ctxData.n,
                    nEff: ctxData.n_eff,
                    dateRange: "Full available history",
                  },
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
                  inputs: [{ id: ctxData.id, name: ctxData.name, source: ctxData.source_note, unit: "sigma" }],
                  window: "Full sample",
                  currentValue: `${ctxData.z_score.toFixed(2)}σ`,
                  n: ctxData.n,
                  nEff: ctxData.n_eff,
                }}
                onOpenAudit={handleOpenAudit}
                onOpenMethod={handleOpenMethod}
              >
                <ContextQqPlot
                  theoretical={ctxData.qq_theoretical}
                  empirical={ctxData.qq_empirical}
                  assetName={ctxData.name}
                />
              </ChartCard>
            </div>
          </div>

          {/* Row 2: Seasonality Calendar Heatmap (C25) */}
          {ctxData.calendar_heatmap && ctxData.calendar_heatmap.length > 0 && (() => {
            const isDiff = ctxData.unit === "bps" || ctxData.unit === "pct" || ctxData.unit === "usd_bn" || ctxData.unit === "usd_mn" || ctxData.id === "vix" || ctxData.id === "indiavix";
            const lastVal = ctxData.calendar_heatmap.slice(-1)[0]?.return;
            const currentFormatted = lastVal !== undefined
              ? isDiff
                ? `${lastVal > 0 ? "+" : ""}${lastVal} ${ctxData.unit}`
                : `${lastVal > 0 ? "+" : ""}${(lastVal * 100).toFixed(1)}%`
              : "—";

            return (
              <ChartCard
                eyebrow="Seasonality Matrix (C25)"
                title={`${ctxData.name} ${isDiff ? "Calendar Change Heatmap" : "Calendar Return Heatmap"}`}
                subtitle={isDiff ? "Monthly net changes grouped by Year and Month" : "Monthly compounded returns grouped by Year and Month"}
                asOf={asOf}
                methodData={{
                  chartTitle: `Calendar Heatmap (C25) — ${ctxData.name}`,
                  whatItShows: isDiff
                    ? "Monthly net level changes across calendar years, accompanied by a mandatory caveat that small sample sizes preclude statistical significance."
                    : "Monthly returns across calendar years, accompanied by a mandatory caveat that small sample sizes preclude statistical significance.",
                  definitionFormula: isDiff
                    ? "\\Delta_{\\text{month}} = P_{\\text{month-end}} - P_{\\text{prev-month-end}}"
                    : "R_{\\text{month}} = \\frac{P_{\\text{month-end}}}{P_{\\text{prev-month-end}}} - 1",
                  parameters: {
                    aggregation: isDiff ? "Monthly net difference" : "Monthly geometric compounded return",
                  },
                  dataSources: [{ series: ctxData.id, source: ctxData.source_note, frequency: "Monthly" }],
                  sample: {
                    n: ctxData.calendar_heatmap.length,
                    dateRange: "All available calendar years",
                  },
                  interpretationNotes: [
                    "Calendar patterns often reflect data mining or random noise; never base strategic positioning solely on calendar seasonality.",
                    "Displayed strictly for descriptive historical context.",
                  ],
                  reproduce: {
                    functionName: "compute_calendar_heatmap",
                    filePath: "pipeline/src/radar/metrics/context.py",
                    testPath: "pipeline/tests/unit/test_context.py",
                  },
                }}
                auditData={{
                  metricName: isDiff ? "Monthly Calendar Net Changes" : "Monthly Calendar Returns",
                  formula: isDiff ? "P_end - P_prev_end" : "P_end / P_prev_end - 1.0",
                  formulaPlain: isDiff ? "Net change for each calendar month." : "Compounded monthly percentage return for each calendar month.",
                  parameters: { sample_years: Array.from(new Set(ctxData.calendar_heatmap.map((d) => d.year))).length },
                  inputs: [{ id: ctxData.id, name: ctxData.name, source: ctxData.source_note, unit: ctxData.unit }],
                  window: "Multi-year monthly aggregation",
                  currentValue: currentFormatted,
                  n: ctxData.calendar_heatmap.length,
                }}
                onOpenAudit={handleOpenAudit}
                onOpenMethod={handleOpenMethod}
              >
                <CalendarHeatmap data={ctxData.calendar_heatmap} assetName={ctxData.name} unit={ctxData.unit} />
              </ChartCard>
            );
          })()}

          {/* Row 3: Declustered Analogs & Forward Outcomes (C26) */}
          <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-[var(--line)]">
              <div>
                <h3 className="text-sm font-semibold tracking-tight text-[var(--tx-0)]">
                  Declustered Similar Historical Readings (C26)
                </h3>
                <p className="text-xs text-[var(--tx-2)]">
                  Top 5 closest historical episodes with a mandatory 20-day spacing to eliminate temporal clustering, with realized forward 5-day and 20-day return outcomes.
                </p>
              </div>
              <span className="text-[11px] px-2.5 py-1 rounded bg-[var(--bg-2)] border border-[var(--line)] font-mono text-[var(--accent)]">
                Min. 20-Day Separation
              </span>
            </div>

            <div className="overflow-x-auto text-xs font-mono">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                    <th className="py-2 px-3">Episode</th>
                    <th className="py-2 px-3">Historical Date</th>
                    <th className="py-2 px-3 text-right">Historical Level</th>
                    <th className="py-2 px-3 text-right">Distance to Today</th>
                    <th className="py-2 px-3 text-right">Subsequent 5D Return</th>
                    <th className="py-2 px-3 text-right">Subsequent 20D Return</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)]">
                  {ctxData.analogs.map((a, idx) => (
                    <tr key={a.date} className="hover:bg-white/5 transition-colors">
                      <td className="py-2.5 px-3 font-semibold text-[var(--accent)]">Episode #{idx + 1}</td>
                      <td className="py-2.5 px-3 flex items-center gap-1.5 text-[var(--tx-0)]">
                        <Calendar className="w-3.5 h-3.5 text-[var(--accent)]" />
                        <span>{a.date}</span>
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold text-[var(--tx-0)]">
                        {a.value.toLocaleString()} {ctxData.unit}
                      </td>
                      <td className="py-2.5 px-3 text-right text-[var(--tx-2)]">
                        {a.distance.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold" style={{ color: (a.fwd_5d ?? 0) >= 0 ? "#3fb950" : "#f85149" }}>
                        {a.fwd_5d !== null && a.fwd_5d !== undefined ? `${a.fwd_5d >= 0 ? "+" : ""}${(a.fwd_5d * 100).toFixed(2)}%` : "—"}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold" style={{ color: (a.fwd_20d ?? 0) >= 0 ? "#3fb950" : "#f85149" }}>
                        {a.fwd_20d !== null && a.fwd_20d !== undefined ? `${a.fwd_20d >= 0 ? "+" : ""}${(a.fwd_20d * 100).toFixed(2)}%` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-16 text-center rounded-lg bg-[var(--bg-1)] border border-[var(--line)] font-mono text-xs text-[var(--tx-2)]">
          {loading ? `Loading Historical Context for ${id.toUpperCase()}...` : "Context data unavailable."}
        </div>
      )}

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
}
