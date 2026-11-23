// web/src/components/charts/YieldHeatmap.tsx
"use client";

import React, { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { YieldCurveEnvelope, YieldSpreadItem, TenYearAnatomyItem } from "@/lib/data/schemas";
import { EChartBox } from "@/components/charts/EChartBox";
import { useRadarStore } from "@/store/useRadarStore";
import { Activity, AlertTriangle, Layers, Percent, TrendingDown, TrendingUp } from "lucide-react";

interface Props {
  data: YieldCurveEnvelope;
}

export function YieldHeatmap({ data }: Props) {
  const theme = useRadarStore((s) => s.theme);
  const yc = (data as any).data || data;

  const [activeTab, setActiveTab] = useState<"curves" | "anatomy" | "heatmap">("curves");

  const textCol = "#8c9bb0";
  const gridLine = "#1d2633";

  // Curve Ribbon Option (Current, 1M, 6M, 1Y)
  const curveOption = useMemo<EChartsOption>(() => {
    if (!yc.curves || yc.curves.length === 0) return {};

    const colors = ["#38bdf8", "#34d399", "#fbbf24", "#94a3b8"];

    const seriesList: any[] = [];
    const legendData: string[] = [];

    yc.curves.forEach((c: any, idx: number) => {
      const color = colors[idx % colors.length];
      legendData.push(c.label);

      // Actual points (scatter/dots)
      seriesList.push({
        name: c.label,
        type: "scatter",
        symbolSize: 8,
        itemStyle: { color, borderColor: "#ffffff", borderWidth: 1.5 },
        data: c.points.map((p: any) => [p.months, p.yield_pct]),
        tooltip: {
          formatter: (params: any) =>
            `<b>${c.label} (${c.date})</b><br/>${params.value[0]} Months: <b>${params.value[1]}%</b>`,
        },
      });

      // Smooth fitted curve line
      if (c.smooth_fitted && c.smooth_fitted.length > 0) {
        seriesList.push({
          name: `${c.label} (NS Fit)`,
          type: "line",
          showSymbol: false,
          smooth: true,
          lineStyle: { color, width: idx === 0 ? 2.5 : 1.5, type: idx === 0 ? "solid" : "dashed" },
          data: c.smooth_fitted.map((p: any) => [p.months, p.yield_pct]),
        });
      }
    });

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "item",
        backgroundColor: "#0c1017",
        borderColor: gridLine,
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
      },
      legend: {
        top: 0,
        textStyle: { color: textCol, fontSize: 11 },
        data: legendData,
      },
      grid: { left: 45, right: 30, top: 40, bottom: 35 },
      xAxis: {
        type: "value",
        name: "Maturity (Months)",
        nameTextStyle: { color: textCol, fontSize: 10 },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace" },
      },
      yAxis: {
        type: "value",
        name: "Yield (%)",
        nameTextStyle: { color: textCol, fontSize: 10 },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace", formatter: "{value}%" },
      },
      series: seriesList,
    };
  }, [yc]);

  // 10Y Anatomy Option (Nominal = Real + Breakeven)
  const anatomyOption = useMemo<EChartsOption>(() => {
    if (!yc.anatomy || yc.anatomy.length === 0) return {};

    const dates = yc.anatomy.map((a: TenYearAnatomyItem) => a.date);
    const nominal = yc.anatomy.map((a: TenYearAnatomyItem) => a.nominal_10y);
    const real = yc.anatomy.map((a: TenYearAnatomyItem) => a.real_10y);
    const breakeven = yc.anatomy.map((a: TenYearAnatomyItem) => a.breakeven_10y);

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
        data: ["10Y Nominal Yield", "10Y TIPS Real Yield", "10Y Breakeven Inflation"],
      },
      grid: { left: 45, right: 30, top: 40, bottom: 25 },
      xAxis: {
        type: "category",
        data: dates,
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace" },
        axisLine: { lineStyle: { color: gridLine } },
      },
      yAxis: {
        type: "value",
        name: "Yield / Inflation (%)",
        nameTextStyle: { color: textCol, fontSize: 10 },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace", formatter: "{value}%" },
      },
      series: [
        {
          name: "10Y Nominal Yield",
          type: "line",
          data: nominal,
          lineStyle: { width: 2.2, color: "#38bdf8" },
          showSymbol: false,
        },
        {
          name: "10Y TIPS Real Yield",
          type: "line",
          data: real,
          lineStyle: { width: 1.8, color: "#34d399", type: "dashed" },
          showSymbol: false,
        },
        {
          name: "10Y Breakeven Inflation",
          type: "line",
          data: breakeven,
          lineStyle: { width: 1.8, color: "#f59e0b", type: "dotted" },
          showSymbol: false,
        },
      ],
    };
  }, [yc]);

  // Yield Term Structure Heatmap (Maturities x Curves)
  const heatmapOption = useMemo<EChartsOption>(() => {
    if (!yc.curves || yc.curves.length === 0) return {};

    const matLabels = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"];
    const matMonths = [1, 3, 6, 12, 24, 36, 60, 84, 120, 240, 360];
    const curveLabels = yc.curves.map((c: any) => `${c.label}\n(${c.date})`);

    const heatmapData: [number, number, number][] = [];
    yc.curves.forEach((c: any, cIdx: number) => {
      matMonths.forEach((m, mIdx) => {
        const pt = c.points.find((p: any) => p.months === m);
        const yVal = pt ? pt.yield_pct : (c.smooth_fitted?.find((p: any) => p.months === m)?.yield_pct ?? 4.0);
        heatmapData.push([cIdx, mIdx, Number(yVal.toFixed(2))]);
      });
    });

    return {
      backgroundColor: "transparent",
      tooltip: {
        position: "top",
        backgroundColor: "#0c1017",
        borderColor: gridLine,
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          const [cIdx, mIdx, val] = params.value;
          return `Snapshot: <b>${yc.curves[cIdx]?.label}</b><br/>Maturity: <b>${matLabels[mIdx]}</b><br/>Par Yield: <b>${val}%</b>`;
        },
      },
      grid: { left: 55, right: 30, top: 20, bottom: 50 },
      xAxis: {
        type: "category",
        data: curveLabels,
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace" },
        axisLine: { lineStyle: { color: gridLine } },
      },
      yAxis: {
        type: "category",
        data: matLabels,
        name: "Maturity",
        nameTextStyle: { color: textCol, fontSize: 10 },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace" },
        axisLine: { lineStyle: { color: gridLine } },
      },
      visualMap: {
        min: 2.5,
        max: 5.5,
        calculable: true,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        textStyle: { color: textCol, fontSize: 10 },
        inRange: {
          color: ["#0284c7", "#0f766e", "#eab308", "#ef4444"],
        },
      },
      series: [
        {
          name: "Yield (%)",
          type: "heatmap",
          data: heatmapData,
          label: {
            show: true,
            color: "#ffffff",
            fontSize: 10,
            fontFamily: "monospace",
            formatter: (p: any) => `${p.value[2]}%`,
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          },
        },
      ],
    };
  }, [yc]);

  return (
    <div className="space-y-6">
      {/* Narrative & View Tabs */}
      <div className="p-4 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono uppercase text-[var(--tx-2)]">Yield Curve Regime:</span>
            {yc.spreads?.some((s: any) => s.inverted) ? (
              <span className="text-xs px-2.5 py-0.5 rounded border border-rose-800 bg-rose-950/70 text-rose-400 font-mono font-bold uppercase tracking-wider flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" /> Inverted Curve
              </span>
            ) : (
              <span className="text-xs px-2.5 py-0.5 rounded border border-emerald-800 bg-emerald-950/70 text-emerald-400 font-mono font-bold uppercase tracking-wider">
                Normal Term Structure
              </span>
            )}
          </div>
          <p className="text-xs text-[var(--tx-1)] font-mono">{yc.interpretation}</p>
        </div>

        <div className="flex rounded border border-[var(--line)] overflow-hidden font-mono text-xs self-start md:self-auto">
          <button
            onClick={() => setActiveTab("curves")}
            className={`px-3 py-1 transition-colors ${
              activeTab === "curves"
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
            }`}
          >
            Nelson-Siegel Ribbon
          </button>
          <button
            onClick={() => setActiveTab("anatomy")}
            className={`px-3 py-1 transition-colors ${
              activeTab === "anatomy"
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
            }`}
          >
            10Y Anatomy (Real vs Breakeven)
          </button>
          <button
            onClick={() => setActiveTab("heatmap")}
            className={`px-3 py-1 transition-colors ${
              activeTab === "heatmap"
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
            }`}
          >
            Yield Heatmap
          </button>
        </div>
      </div>

      {/* Main Chart */}
      <div className="bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)]">
        <div className="flex items-center justify-between mb-3 border-b border-[var(--line)] pb-2">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[var(--accent)]" />
            <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-[var(--tx-0)]">
              {activeTab === "curves"
                ? "US Treasury Yield Curve Snapshots & Nelson-Siegel Fits"
                : activeTab === "anatomy"
                ? "10-Year Yield Anatomy: Real Rates & Inflation Expectations"
                : "US Treasury Par Yield Term Structure Heatmap"}
            </h3>
          </div>
          <span className="text-[10px] font-mono text-[var(--tx-2)]">As of: {yc.as_of}</span>
        </div>

        <div className="h-80 w-full">
          <EChartBox
            option={
              activeTab === "curves"
                ? curveOption
                : activeTab === "anatomy"
                ? anatomyOption
                : heatmapOption
            }
          />
        </div>
      </div>


      {/* Nelson-Siegel 3-Factor Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] space-y-1 font-mono">
          <div className="text-[10px] uppercase text-[var(--tx-2)] flex items-center justify-between">
            <span>Factor L (Level)</span>
            <span className="text-[9px] px-1 py-0.2 rounded bg-[var(--bg-1)]">Long-term anchor</span>
          </div>
          <div className="text-2xl font-bold text-[var(--tx-0)]">{yc.factors.level.toFixed(2)}%</div>
          <div className="text-xs text-[var(--tx-2)] flex justify-between pt-1 border-t border-[var(--line)]">
            <span>Historical Percentile:</span>
            <strong className="text-[var(--accent)]">{yc.factors.level_percentile.toFixed(0)}th</strong>
          </div>
        </div>

        <div className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] space-y-1 font-mono">
          <div className="text-[10px] uppercase text-[var(--tx-2)] flex items-center justify-between">
            <span>Factor S (Slope)</span>
            <span className="text-[9px] px-1 py-0.2 rounded bg-[var(--bg-1)]">Short vs Long</span>
          </div>
          <div className={`text-2xl font-bold ${yc.factors.slope < 0 ? "text-rose-400" : "text-emerald-400"}`}>
            {yc.factors.slope > 0 ? "+" : ""}{yc.factors.slope.toFixed(2)}%
          </div>
          <div className="text-xs text-[var(--tx-2)] flex justify-between pt-1 border-t border-[var(--line)]">
            <span>Historical Percentile:</span>
            <strong className="text-[var(--accent)]">{yc.factors.slope_percentile.toFixed(0)}th</strong>
          </div>
        </div>

        <div className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] space-y-1 font-mono">
          <div className="text-[10px] uppercase text-[var(--tx-2)] flex items-center justify-between">
            <span>Factor C (Curvature)</span>
            <span className="text-[9px] px-1 py-0.2 rounded bg-[var(--bg-1)]">Belly vs Wings</span>
          </div>
          <div className="text-2xl font-bold text-[var(--tx-0)]">{yc.factors.curvature.toFixed(2)}%</div>
          <div className="text-xs text-[var(--tx-2)] flex justify-between pt-1 border-t border-[var(--line)]">
            <span>Fit RMSE:</span>
            <strong className="text-sky-400">{yc.factors.rmse.toFixed(3)}%</strong>
          </div>
        </div>
      </div>

      {/* Benchmark Spreads */}
      <div className="bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)] space-y-3 font-mono">
        <h4 className="text-xs font-bold uppercase tracking-wider text-[var(--tx-0)] border-b border-[var(--line)] pb-2">
          Benchmark Curve Spreads & Inversion Tracker
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {yc.spreads?.map((sp: YieldSpreadItem, idx: number) => (
            <div key={idx} className="p-3 bg-[var(--bg-1)] rounded border border-[var(--line)] space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-[var(--tx-0)]">{sp.name}</span>
                {sp.inverted && (
                  <span className="px-1.5 py-0.2 rounded bg-rose-950/70 text-rose-400 text-[9px] border border-rose-800 font-bold uppercase">
                    Inverted
                  </span>
                )}
              </div>
              <div className="text-xl font-bold text-[var(--tx-0)]">
                {sp.current > 0 ? "+" : ""}{(sp.current * 100).toFixed(0)} bps
              </div>
              <div className="text-[11px] text-[var(--tx-2)] flex justify-between pt-1 border-t border-[var(--line)]">
                <span>1M Δ: {sp.chg_1m > 0 ? "+" : ""}{(sp.chg_1m * 100).toFixed(0)} bps</span>
                <span>{sp.percentile.toFixed(0)}th pct</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
