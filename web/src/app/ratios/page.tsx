// web/src/app/ratios/page.tsx
"use client";

import React, { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { ChartCard } from "@/components/charts/ChartCard";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { HealthDot } from "@/components/ui/HealthDot";
import { loadRatios } from "@/lib/data/client";
import { RatiosEnvelope, RatioItem } from "@/lib/data/schemas";

export default function RatiosPage() {
  const [env, setEnv] = useState<RatiosEnvelope | null>(null);
  const [selectedRatioId, setSelectedRatioId] = useState<string>("gold_silver");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadRatios()
      .then((res) => {
        if (res.status === "error") setError(res.error);
        else setEnv(res.data);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const data = env?.data;

  const activeRatio: RatioItem | undefined = useMemo(() => {
    if (!data?.ratios?.length) return undefined;
    return data.ratios.find((r) => r.id === selectedRatioId) || data.ratios[0];
  }, [data, selectedRatioId]);

  const ratioHistoryOption = useMemo(() => {
    if (!activeRatio || activeRatio.history_dates.length === 0) return {};
    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(10,14,20,0.95)",
        borderColor: "rgba(255,255,255,0.12)",
        textStyle: { color: "#e6edf3", fontSize: 11, fontFamily: "monospace" },
      },
      grid: { top: 25, right: 20, bottom: 25, left: 45, containLabel: true },
      xAxis: {
        type: "category",
        data: activeRatio.history_dates,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.15)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "monospace" },
      },
      yAxis: {
        type: "value",
        scale: true,
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "monospace" },
      },
      series: [
        {
          name: activeRatio.name,
          type: "line",
          data: activeRatio.history_values,
          showSymbol: false,
          lineStyle: { width: 2, color: "#58a6ff" },
          areaStyle: { color: "rgba(88,166,255,0.08)" },
        },
      ],
    };
  }, [activeRatio]);

  const goldFairValueOption = useMemo(() => {
    const gv = data?.gold_valuation;
    if (!gv || !gv.dates?.length) return {};
    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(10,14,20,0.95)",
        borderColor: "rgba(255,255,255,0.12)",
        textStyle: { color: "#e6edf3", fontSize: 11, fontFamily: "monospace" },
      },
      legend: {
        data: ["Actual Gold ($/oz)", "Real-Rate + DXY Fair Value"],
        textStyle: { color: "#8b949e", fontSize: 11, fontFamily: "monospace" },
        top: 0,
      },
      grid: { top: 40, right: 20, bottom: 25, left: 50, containLabel: true },
      xAxis: {
        type: "category",
        data: gv.dates,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.15)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "monospace" },
      },
      yAxis: {
        type: "value",
        scale: true,
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "monospace" },
      },
      series: [
        {
          name: "Actual Gold ($/oz)",
          type: "line",
          data: gv.actual,
          showSymbol: false,
          lineStyle: { width: 2, color: "#e6b94d" },
          itemStyle: { color: "#e6b94d" },
        },
        {
          name: "Real-Rate + DXY Fair Value",
          type: "line",
          data: gv.fair_value,
          showSymbol: false,
          lineStyle: { width: 1.8, type: "dashed", color: "#3fb6a8" },
          itemStyle: { color: "#3fb6a8" },
        },
      ],
    };
  }, [data]);

  const cuAuOption = useMemo(() => {
    const cu = data?.cu_au_vs_10y;
    if (!cu || !cu.dates?.length) return {};
    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(10,14,20,0.95)",
        borderColor: "rgba(255,255,255,0.12)",
        textStyle: { color: "#e6edf3", fontSize: 11, fontFamily: "monospace" },
      },
      legend: {
        data: ["Copper/Gold (Z)", "US 10Y Yield (Z)"],
        textStyle: { color: "#8b949e", fontSize: 11, fontFamily: "monospace" },
        top: 0,
      },
      grid: { top: 40, right: 20, bottom: 25, left: 40, containLabel: true },
      xAxis: {
        type: "category",
        data: cu.dates,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.15)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "monospace" },
      },
      yAxis: {
        type: "value",
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "monospace" },
      },
      series: [
        {
          name: "Copper/Gold (Z)",
          type: "line",
          data: cu.cu_au_z,
          showSymbol: false,
          lineStyle: { width: 2, color: "#f0883e" },
          itemStyle: { color: "#f0883e" },
        },
        {
          name: "US 10Y Yield (Z)",
          type: "line",
          data: cu.us10y_z,
          showSymbol: false,
          lineStyle: { width: 2, color: "#58a6ff" },
          itemStyle: { color: "#58a6ff" },
        },
      ],
    };
  }, [data]);

  const creditVolOption = useMemo(() => {
    const cv = data?.credit_vs_vol;
    if (!cv || !cv.dates?.length) return {};
    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(10,14,20,0.95)",
        borderColor: "rgba(255,255,255,0.12)",
        textStyle: { color: "#e6edf3", fontSize: 11, fontFamily: "monospace" },
      },
      legend: {
        data: ["HY Credit Spread (Z)", "Equity VIX (Z)"],
        textStyle: { color: "#8b949e", fontSize: 11, fontFamily: "monospace" },
        top: 0,
      },
      grid: { top: 40, right: 20, bottom: 25, left: 40, containLabel: true },
      xAxis: {
        type: "category",
        data: cv.dates,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.15)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "monospace" },
      },
      yAxis: {
        type: "value",
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "monospace" },
      },
      series: [
        {
          name: "HY Credit Spread (Z)",
          type: "line",
          data: cv.hy_z,
          showSymbol: false,
          lineStyle: { width: 2, color: "#e8765a" },
          itemStyle: { color: "#e8765a" },
        },
        {
          name: "Equity VIX (Z)",
          type: "line",
          data: cv.vix_z,
          showSymbol: false,
          lineStyle: { width: 2, color: "#a371f7" },
          itemStyle: { color: "#a371f7" },
        },
      ],
    };
  }, [data]);

  if (error) {
    return (
      <div className="p-5 rounded-lg border border-rose-500/40 bg-[var(--bg-1)] text-rose-400 text-xs font-mono">
        Failed to load Cross-Asset Ratios: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-96 rounded-lg border border-[var(--line)] bg-[var(--bg-1)] animate-pulse flex items-center justify-center font-mono text-xs text-[var(--tx-2)]">
        Loading Relative Value &amp; Valuation Models...
      </div>
    );
  }

  const gv = data.gold_valuation;

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-[var(--line)]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Relative Value &amp; Dislocations
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--tx-0)] mt-0.5">
            Macro Ratios &amp; Fair-Value Models
          </h1>
          <p className="text-xs text-[var(--tx-1)] mt-1">
            Cross-asset relative ratios, Gold real-rate valuation residuals, Copper/Gold vs. 10Y yield divergence, and Credit vs. VIX stress spreads.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot status="ok" label="Live Cross-Asset" />
          <DataAgeBadge asOf={env?.as_of || ""} />
        </div>
      </div>

      {/* Ratios Matrix Table */}
      <ChartCard
        eyebrow="Cross-Asset Ratios Monitor"
        title="Key Institutional Ratios & Breadth Gauges"
        subtitle="Click any row to inspect its trailing 252-day trajectory"
        asOf={env?.as_of}
        sourceText="COMEX, ICE, NYSE, NSE"
        dataTable={
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                <th className="py-1.5 px-2">Ratio</th>
                <th className="py-1.5 px-2">Category</th>
                <th className="py-1.5 px-2 text-right">Current</th>
                <th className="py-1.5 px-2 text-right">1D</th>
                <th className="py-1.5 px-2 text-right">20D</th>
                <th className="py-1.5 px-2 text-right">Z-Score</th>
                <th className="py-1.5 px-2 text-right">Pctile</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--line)]">
              {data.ratios.map((r) => (
                <tr key={r.id}>
                  <td className="py-1.5 px-2">{r.name}</td>
                  <td className="py-1.5 px-2">{r.category}</td>
                  <td className="py-1.5 px-2 text-right">{r.current.toFixed(2)}</td>
                  <td className="py-1.5 px-2 text-right">{r.chg_1d.toFixed(2)}</td>
                  <td className="py-1.5 px-2 text-right">{r.chg_20d.toFixed(2)}</td>
                  <td className="py-1.5 px-2 text-right">{r.z_score.toFixed(2)}</td>
                  <td className="py-1.5 px-2 text-right">{r.percentile.toFixed(0)}th</td>
                </tr>
              ))}
            </tbody>
          </table>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
            <thead>
              <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                <th className="py-2 px-2 font-sans">Ratio</th>
                <th className="py-2 px-2 font-sans">Category</th>
                <th className="py-2 px-2 text-right">Level</th>
                <th className="py-2 px-2 text-right">1D</th>
                <th className="py-2 px-2 text-right">20D</th>
                <th className="py-2 px-2 text-right">Z-Score</th>
                <th className="py-2 px-2 text-right">Rank</th>
                <th className="py-2 px-2 font-sans">Current Signal</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--line)]">
              {data.ratios.map((r) => {
                const isSelected = activeRatio?.id === r.id;
                const unitSuffix = r.op === "sub" ? "" : "%";
                return (
                  <tr
                    key={r.id}
                    onClick={() => setSelectedRatioId(r.id)}
                    className={`cursor-pointer transition ${
                      isSelected ? "bg-[var(--bg-2)]/80" : "hover:bg-[var(--bg-0)]/60"
                    }`}
                  >
                    <td className="py-2 px-2 font-sans font-medium text-[var(--tx-0)]">{r.name}</td>
                    <td className="py-2 px-2 font-sans text-[var(--tx-2)]">{r.category}</td>
                    <td className="py-2 px-2 text-right font-semibold text-[var(--tx-0)]">{r.current.toFixed(2)}</td>
                    <td className={`py-2 px-2 text-right ${r.chg_1d >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]"}`}>
                      {r.chg_1d >= 0 ? "+" : ""}
                      {r.chg_1d.toFixed(2)}
                      {unitSuffix}
                    </td>
                    <td className={`py-2 px-2 text-right ${r.chg_20d >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]"}`}>
                      {r.chg_20d >= 0 ? "+" : ""}
                      {r.chg_20d.toFixed(2)}
                      {unitSuffix}
                    </td>
                    <td
                      className={`py-2 px-2 text-right font-semibold ${
                        Math.abs(r.z_score) >= 1.5 ? "text-amber-400" : "text-[var(--tx-1)]"
                      }`}
                    >
                      {r.z_score >= 0 ? "+" : ""}
                      {r.z_score.toFixed(2)}σ
                    </td>
                    <td className="py-2 px-2 text-right text-[var(--tx-2)]">{r.percentile.toFixed(0)}th</td>
                    <td className="py-2 px-2 font-sans text-[var(--tx-2)]">{r.interpretation}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </ChartCard>

      {/* Selected Ratio 252D History + Gold Valuation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6">
          <ChartCard
            eyebrow="Selected Ratio Trajectory"
            title={activeRatio ? `${activeRatio.name} (252D)` : "Ratio History"}
            subtitle={activeRatio ? `Current Z-Score: ${activeRatio.z_score >= 0 ? "+" : ""}${activeRatio.z_score}σ (${activeRatio.percentile.toFixed(0)}th pct)` : ""}
            asOf={env?.as_of}
          >
            <ReactECharts option={ratioHistoryOption} style={{ height: "320px", width: "100%" }} notMerge={true} />
          </ChartCard>
        </div>

        <div className="lg:col-span-6">
          <ChartCard
            eyebrow="Real-Rate Fair Value Model"
            title="Gold Actual vs. Real-Rate & Dollar Fair Value"
            subtitle={
              gv?.current_fair_value
                ? `Actual $${gv.current_actual} vs Model $${gv.current_fair_value} (Gap: ${gv.valuation_gap_pct >= 0 ? "+" : ""}${gv.valuation_gap_pct}%, R²=${gv.r2})`
                : "OLS fit of log(Gold) ~ US 10Y Real Yield + log(DXY)"
            }
            asOf={env?.as_of}
            sourceText="COMEX Gold, FRED DFII10, DXY"
          >
            <ReactECharts option={goldFairValueOption} style={{ height: "320px", width: "100%" }} />
          </ChartCard>
        </div>
      </div>

      {/* Growth vs Rates & Credit vs Vol */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6">
          <ChartCard
            eyebrow="Metals vs. Sovereign Rates"
            title="Copper/Gold vs. US 10Y Yield Divergence"
            subtitle={
              data.cu_au_vs_10y?.divergence_z !== undefined
                ? `Current 10Y - Cu/Au Gap: ${data.cu_au_vs_10y.divergence_z >= 0 ? "+" : ""}${data.cu_au_vs_10y.divergence_z}σ`
                : "Standardized 504-day comparison"
            }
            asOf={env?.as_of}
          >
            <ReactECharts option={cuAuOption} style={{ height: "320px", width: "100%" }} />
          </ChartCard>
        </div>

        <div className="lg:col-span-6">
          <ChartCard
            eyebrow="Cross-Market Stress Lead"
            title="High-Yield Credit OAS vs. Equity VIX"
            subtitle={
              data.credit_vs_vol?.dislocation_z !== undefined
                ? `Credit - VIX Spread: ${data.credit_vs_vol.dislocation_z >= 0 ? "+" : ""}${data.credit_vs_vol.dislocation_z}σ`
                : "Identifies when credit widens ahead of equity volatility"
            }
            asOf={env?.as_of}
          >
            <ReactECharts option={creditVolOption} style={{ height: "320px", width: "100%" }} />
          </ChartCard>
        </div>
      </div>
    </div>
  );
}
