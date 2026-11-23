// web/src/components/charts/TermStructureChart.tsx
"use client";

import React, { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { ImpliedVolEnvelope, FearTailItem } from "@/lib/data/schemas";
import { EChartBox } from "@/components/charts/EChartBox";
import { ChartPanel } from "@/components/insight/ChartPanel";
import { useRadarStore } from "@/store/useRadarStore";
import { Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, ShieldAlert, Sparkles, TrendingUp } from "lucide-react";

interface Props {
  data: ImpliedVolEnvelope;
}

export function TermStructureChart({ data }: Props) {
  const theme = useRadarStore((s) => s.theme);
  const isDark = theme === "dark";
  const vol = (data as any).data || data;

  const [activeTab, setActiveTab] = useState<"curve" | "slope_history">("curve");

  const textCol = "#8c9bb0";
  const gridLine = "#1d2633";

  // ECharts Option for Term Structure Curve (9D, 30D, 3M, 6M)
  const curveOption = useMemo<EChartsOption>(() => {
    if (!vol.curve || vol.curve.length === 0) return {};

    const tenors = vol.curve.map((p: any) => p.tenor);
    const curVals = vol.curve.map((p: any) => p.current);
    const m1Vals = vol.curve.map((p: any) => p.one_month_ago);
    const medVals = vol.curve.map((p: any) => p.one_year_median);
    const p10Vals = vol.curve.map((p: any) => p.p10);
    const p90Vals = vol.curve.map((p: any) => p.p90);

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0c1017",
        borderColor: gridLine,
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
      },
      legend: {
        top: 0,
        textStyle: { color: textCol, fontSize: 11 },
        data: ["Current Curve", "1 Month Ago", "1-Year Median", "10-90th Pct Band"],
      },
      grid: { left: 45, right: 30, top: 40, bottom: 25 },
      xAxis: {
        type: "category",
        data: tenors,
        axisLabel: { color: textCol, fontSize: 11, fontFamily: "monospace" },
        axisLine: { lineStyle: { color: gridLine } },
      },
      yAxis: {
        type: "value",
        name: "Volatility Index Pts",
        nameTextStyle: { color: textCol, fontSize: 10 },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace" },
      },
      series: [
        // Shaded Band P10-P90
        {
          name: "10-90th Pct Band",
          type: "line",
          data: p10Vals,
          lineStyle: { opacity: 0 },
          stack: "confidence-band",
          symbol: "none",
        },
        {
          name: "10-90th Pct Band",
          type: "line",
          data: p90Vals.map((v: number, i: number) => v - p10Vals[i]),
          lineStyle: { opacity: 0 },
          areaStyle: { color: "rgba(56, 189, 248, 0.12)" },
          stack: "confidence-band",
          symbol: "none",
        },
        {
          name: "1-Year Median",
          type: "line",
          data: medVals,
          lineStyle: { color: "#64748b", width: 1.5, type: "dotted" },
          symbol: "circle",
          symbolSize: 4,
          itemStyle: { color: "#64748b" },
        },
        {
          name: "1 Month Ago",
          type: "line",
          data: m1Vals,
          lineStyle: { color: "#94a3b8", width: 1.8, type: "dashed" },
          symbol: "circle",
          symbolSize: 4,
          itemStyle: { color: "#94a3b8" },
        },
        {
          name: "Current Curve",
          type: "line",
          data: curVals,
          lineStyle: {
            color: vol.current_state === "backwardation" ? "#f43f5e" : "#2dd4bf",
            width: 3,
          },
          symbol: "circle",
          symbolSize: 8,
          itemStyle: {
            color: vol.current_state === "backwardation" ? "#f43f5e" : "#2dd4bf",
            borderColor: "#ffffff",
            borderWidth: 2,
          },
        },
      ],
    };
  }, [vol]);

  // ECharts Option for Slope History
  const slopeOption = useMemo<EChartsOption>(() => {
    if (!vol.slope_history || vol.slope_history.length === 0) return {};

    const dates = vol.slope_history.map((s: any) => s.date);
    const slope3m = vol.slope_history.map((s: any) => s.slope_3m * 100);

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0c1017",
        borderColor: gridLine,
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          const item = params[0];
          return `<b>${item.name}</b><br/>3M/30D Slope: <b>${item.value > 0 ? "+" : ""}${Number(item.value).toFixed(2)}%</b>`;
        },
      },
      grid: { left: 45, right: 30, top: 25, bottom: 25 },
      xAxis: {
        type: "category",
        data: dates,
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace" },
        axisLine: { lineStyle: { color: gridLine } },
      },
      yAxis: {
        type: "value",
        name: "Slope % (VIX3M / VIX - 1)",
        nameTextStyle: { color: textCol, fontSize: 10 },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace", formatter: "{value}%" },
      },
      series: [
        {
          name: "3M Slope",
          type: "line",
          showSymbol: false,
          data: slope3m,
          lineStyle: { width: 1.8, color: "#38bdf8" },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "#f43f5e", type: "dashed" },
            data: [{ yAxis: -2, label: { formatter: "Backwardation Threshold (-2%)", color: "#f43f5e" } }],
          },
          markArea: {
            itemStyle: { color: "rgba(244, 63, 94, 0.12)" },
            data: [
              [
                { yAxis: -50 },
                { yAxis: -2 },
              ],
            ],
          },
        },
      ],
    };
  }, [vol]);

  const stateBadgeColor =
    vol.current_state === "backwardation"
      ? "bg-rose-950/70 text-rose-400 border-rose-800"
      : vol.current_state === "contango"
      ? "bg-emerald-950/70 text-emerald-400 border-emerald-800"
      : "bg-amber-950/70 text-amber-400 border-amber-800";

  return (
    <div className="space-y-6">
      {/* Top Status & Narrative */}
      <div className="p-4 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono uppercase text-[var(--tx-2)]">Term Structure State:</span>
            <span className={`text-xs px-2.5 py-0.5 rounded border font-mono font-bold uppercase tracking-wider ${stateBadgeColor}`}>
              {vol.current_state}
            </span>
            <span className="text-xs text-[var(--tx-2)] font-mono">
              (3M/30D Slope: <b>{vol.slope_3m > 0 ? "+" : ""}{(vol.slope_3m * 100).toFixed(1)}%</b> • {vol.slope_percentile.toFixed(0)}th percentile)
            </span>
          </div>
          <p className="text-xs text-[var(--tx-1)] font-mono">
            {vol.interpretation}
          </p>
        </div>

        <div className="flex rounded border border-[var(--line)] overflow-hidden font-mono text-xs self-start md:self-auto">
          <button
            onClick={() => setActiveTab("curve")}
            className={`px-3 py-1 transition-colors ${
              activeTab === "curve"
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
            }`}
          >
            Term Structure Curve
          </button>
          <button
            onClick={() => setActiveTab("slope_history")}
            className={`px-3 py-1 transition-colors ${
              activeTab === "slope_history"
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
            }`}
          >
            Historical Slope
          </button>
        </div>
      </div>

      {/* Main Chart Panel */}
      <div className="bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)]">
        <div className="flex items-center justify-between mb-3 border-b border-[var(--line)] pb-2">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-[var(--accent)]" />
            <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-[var(--tx-0)]">
              {activeTab === "curve" ? "Cboe Volatility Term Structure (9D - 6M)" : "Term Structure Slope History & Inversions"}
            </h3>
          </div>
          <span className="text-[10px] font-mono text-[var(--tx-2)]">
            As of: {vol.as_of}
          </span>
        </div>

        <div className="h-80 w-full">
          <EChartBox option={activeTab === "curve" ? curveOption : slopeOption} />
        </div>
      </div>

      {/* Fear, Tail & Vol-of-Vol Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {vol.fear_tail_indices?.map((idx: FearTailItem) => (
          <div key={idx.id} className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] flex flex-col justify-between">
            <div>
              <div className="text-[10px] font-mono uppercase text-[var(--tx-2)] flex items-center justify-between">
                <span>{idx.name}</span>
                <span className="text-[9px] px-1 py-0.2 rounded bg-[var(--bg-1)] border border-[var(--line)]">
                  Z: {idx.z_score > 0 ? "+" : ""}{idx.z_score.toFixed(2)}σ
                </span>
              </div>
              <div className="text-2xl font-bold font-mono text-[var(--tx-0)] mt-1">
                {idx.current.toFixed(1)}
              </div>
            </div>

            <div className="mt-3 pt-2 border-t border-[var(--line)] flex items-center justify-between text-xs font-mono">
              <span className="text-[var(--tx-2)] text-[11px]">Expanding Pct:</span>
              <span className="font-bold text-[var(--accent)]">{idx.percentile.toFixed(0)}th</span>
            </div>
            <div className="flex items-center justify-between text-[10px] font-mono text-[var(--tx-2)] mt-0.5">
              <span>1W Δ: <b className={idx.chg_1w >= 0 ? "text-rose-400" : "text-emerald-400"}>{idx.chg_1w > 0 ? "+" : ""}{idx.chg_1w.toFixed(1)}</b></span>
              <span>1M Δ: <b className={idx.chg_1m >= 0 ? "text-rose-400" : "text-emerald-400"}>{idx.chg_1m > 0 ? "+" : ""}{idx.chg_1m.toFixed(1)}</b></span>
            </div>
          </div>
        ))}
      </div>

      {/* Implied vs Realized & Variance Risk Premium (VRP) */}
      <div className="bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)] space-y-4">
        <div className="flex items-center justify-between border-b border-[var(--line)] pb-2">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-sky-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-[var(--tx-0)]">
              Implied vs. Realized Volatility & Variance Risk Premium (VRP)
            </h3>
          </div>
          <span className="text-[10px] font-mono text-[var(--tx-2)]">
            Trailing 21-Day Realized vs 30-Day Implied Index
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {vol.implied_vs_realized?.map((item: any, i: number) => (
            <div key={i} className="p-3 bg-[var(--bg-1)] rounded border border-[var(--line)] space-y-2">
              <div className="flex items-center justify-between font-mono">
                <span className="text-xs font-bold text-[var(--tx-0)]">{item.market}</span>
                <span className="text-[10px] text-[var(--tx-2)]">Gap Rank: <b>{item.gap_percentile}th pct</b></span>
              </div>

              <div className="grid grid-cols-3 gap-2 font-mono text-xs pt-1">
                <div>
                  <div className="text-[10px] text-[var(--tx-2)]">Implied (VIX)</div>
                  <div className="font-bold text-[var(--tx-0)]">{item.implied_vol.toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-[10px] text-[var(--tx-2)]">Realized (21D)</div>
                  <div className="font-bold text-[var(--tx-0)]">{item.realized_vol_21d.toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-[10px] text-[var(--tx-2)]">Spread / Gap</div>
                  <div className={`font-bold ${item.gap >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    {item.gap > 0 ? "+" : ""}{item.gap.toFixed(1)}%
                  </div>
                </div>
              </div>

              {item.vrp_proxy !== undefined && (
                <div className="mt-2 pt-2 border-t border-[var(--line)] flex items-center justify-between text-[11px] font-mono">
                  <span className="text-[var(--tx-2)]">Variance Risk Premium (Trailing Proxy):</span>
                  <span className="font-bold text-sky-400">{item.vrp_proxy > 0 ? "+" : ""}{item.vrp_proxy} pts²</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
