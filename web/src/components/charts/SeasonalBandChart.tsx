// web/src/components/charts/SeasonalBandChart.tsx
"use client";

import React, { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { FundamentalsEnvelope, InventoryItem, SeasonalWeekPoint } from "@/lib/data/schemas";
import { EChartBox } from "@/components/charts/EChartBox";
import { useRadarStore } from "@/store/useRadarStore";
import { Activity, Clock, Droplets, Flame, Fuel, Info, Sparkles, TrendingDown, TrendingUp } from "lucide-react";

interface Props {
  data: FundamentalsEnvelope;
}

export function SeasonalBandChart({ data }: Props) {
  const theme = useRadarStore((s) => s.theme);
  const fund = (data as any).data || data;

  const [selectedId, setSelectedId] = useState<string>("crude_oil");

  const activeItem: InventoryItem = useMemo(() => {
    return (
      fund.inventories?.find((item: InventoryItem) => item.id === selectedId) ||
      fund.inventories?.[0] ||
      null
    );
  }, [fund, selectedId]);

  const textCol = "#8c9bb0";
  const gridLine = "#1d2633";

  // ECharts Option for 5-Year Seasonal Band
  const chartOption = useMemo<EChartsOption>(() => {
    if (!activeItem || !activeItem.weeks || activeItem.weeks.length === 0) return {};

    const weeks = activeItem.weeks.map((w: SeasonalWeekPoint) => `Wk ${w.week}`);
    const minVals = activeItem.weeks.map((w: SeasonalWeekPoint) => w.min_5y);
    const maxVals = activeItem.weeks.map((w: SeasonalWeekPoint) => w.max_5y);
    const meanVals = activeItem.weeks.map((w: SeasonalWeekPoint) => w.mean_5y);
    const currVals = activeItem.weeks.map((w: SeasonalWeekPoint) => w.current_year);
    const prevVals = activeItem.weeks.map((w: SeasonalWeekPoint) => w.prev_year);

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0c1017",
        borderColor: gridLine,
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          const wLabel = params[0].name;
          const idx = params[0].dataIndex;
          const pt = activeItem.weeks[idx];
          return `<b>${activeItem.name} — ${wLabel}</b><br/>
Current Year: <b>${pt.current_year ? `${pt.current_year} ${activeItem.unit}` : "N/A"}</b><br/>
Previous Year: ${pt.prev_year ? `${pt.prev_year}` : "N/A"}<br/>
5-Year Average: ${pt.mean_5y}<br/>
5-Year Range: [${pt.min_5y} - ${pt.max_5y}]`;
        },
      },
      legend: {
        top: 0,
        textStyle: { color: textCol, fontSize: 11 },
        data: ["Current Year", "Previous Year", "5-Year Average", activeItem.band_years_label],
      },
      grid: { left: 55, right: 30, top: 40, bottom: 25 },
      xAxis: {
        type: "category",
        data: weeks,
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace", interval: 3 },
        axisLine: { lineStyle: { color: gridLine } },
      },
      yAxis: {
        type: "value",
        name: activeItem.unit,
        nameTextStyle: { color: textCol, fontSize: 10 },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace" },
      },
      series: [
        // 5-Year Range Min
        {
          name: activeItem.band_years_label,
          type: "line",
          data: minVals,
          lineStyle: { opacity: 0 },
          stack: "band",
          symbol: "none",
        },
        // 5-Year Range Fill
        {
          name: activeItem.band_years_label,
          type: "line",
          data: maxVals.map((v: number, i: number) => v - minVals[i]),
          lineStyle: { opacity: 0 },
          areaStyle: { color: "rgba(56, 189, 248, 0.15)" },
          stack: "band",
          symbol: "none",
        },
        // 5-Year Mean
        {
          name: "5-Year Average",
          type: "line",
          data: meanVals,
          lineStyle: { color: "#64748b", width: 1.8, type: "dashed" },
          showSymbol: false,
        },
        // Previous Year
        {
          name: "Previous Year",
          type: "line",
          data: prevVals,
          lineStyle: { color: "#94a3b8", width: 1.5, type: "dotted" },
          showSymbol: false,
        },
        // Current Year
        {
          name: "Current Year",
          type: "line",
          data: currVals,
          lineStyle: { width: 2.8, color: "#2dd4bf" },
          symbol: "circle",
          symbolSize: 6,
          itemStyle: { color: "#2dd4bf", borderColor: "#ffffff", borderWidth: 1.5 },
        },
      ],
    };
  }, [activeItem]);

  if (!activeItem) return null;

  return (
    <div className="space-y-6">
      {/* Narrative & Commodity Tabs */}
      <div className="p-4 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono uppercase text-[var(--tx-2)]">Physical Fundamentals:</span>
            <span className="text-xs px-2.5 py-0.5 rounded border border-teal-800 bg-teal-950/70 text-teal-400 font-mono font-bold uppercase tracking-wider">
              {activeItem.name}
            </span>
            <span className="text-xs text-[var(--tx-2)] font-mono">
              (Deviation: <b>{activeItem.deviation_pct > 0 ? "+" : ""}{activeItem.deviation_pct}%</b> • {activeItem.deviation_percentile.toFixed(0)}th pct)
            </span>
          </div>
          <p className="text-xs text-[var(--tx-1)] font-mono">{fund.interpretation}</p>
        </div>

        <div className="flex items-center rounded border border-[var(--line)] overflow-x-auto max-w-full font-mono text-xs self-start md:self-auto shrink-0">
          {fund.inventories?.map((it: InventoryItem) => (
            <button
              key={it.id}
              onClick={() => setSelectedId(it.id)}
              className={`px-3 py-1 transition-colors whitespace-nowrap cursor-pointer ${
                selectedId === it.id
                  ? "bg-[var(--accent)] text-white font-medium"
                  : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              }`}
            >
              {it.id === "crude_oil" ? "Crude" : it.id === "gasoline" ? "Gasoline" : it.id === "distillate" ? "Distillates" : "Natural Gas"}
            </button>
          ))}
        </div>
      </div>

      {/* Main Seasonal Band Chart */}
      <div className="bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)]">
        <div className="flex items-center justify-between mb-3 border-b border-[var(--line)] pb-2">
          <div className="flex items-center gap-2">
            {selectedId === "natgas" ? (
              <Flame className="w-4 h-4 text-amber-400" />
            ) : (
              <Droplets className="w-4 h-4 text-[var(--accent)]" />
            )}
            <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-[var(--tx-0)]">
              {activeItem.name} vs. 5-Year Seasonal Band ({activeItem.band_years_label})
            </h3>
          </div>
          <span className="text-[10px] font-mono text-[var(--tx-2)]">As of: {activeItem.as_of}</span>
        </div>

        <div className="h-80 w-full">
          <EChartBox option={chartOption} />
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 font-mono">
        <div className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] space-y-1">
          <div className="text-[10px] uppercase text-[var(--tx-2)]">Current Storage Level</div>
          <div className="text-2xl font-bold text-[var(--tx-0)]">
            {activeItem.current_level.toLocaleString()}
          </div>
          <div className="text-[11px] text-[var(--tx-2)]">{activeItem.unit}</div>
        </div>

        <div className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] space-y-1">
          <div className="text-[10px] uppercase text-[var(--tx-2)]">5-Year Normal Average</div>
          <div className="text-2xl font-bold text-[var(--tx-0)]">
            {activeItem.mean_5y.toLocaleString()}
          </div>
          <div className="text-[11px] text-[var(--tx-2)]">
            Range: [{activeItem.min_5y} - {activeItem.max_5y}]
          </div>
        </div>

        <div className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] space-y-1">
          <div className="text-[10px] uppercase text-[var(--tx-2)]">Deviation vs. 5Y Mean</div>
          <div className={`text-2xl font-bold ${activeItem.deviation >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
            {activeItem.deviation > 0 ? "+" : ""}{activeItem.deviation}
          </div>
          <div className="text-[11px] text-[var(--tx-2)] flex justify-between">
            <span>{activeItem.deviation_pct > 0 ? "+" : ""}{activeItem.deviation_pct}%</span>
            <strong className="text-[var(--accent)]">{activeItem.deviation_percentile.toFixed(0)}th pct</strong>
          </div>
        </div>

        <div className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] space-y-1">
          <div className="text-[10px] uppercase text-[var(--tx-2)]">1-Week Build / Draw</div>
          <div className={`text-2xl font-bold ${activeItem.build_draw_1w >= 0 ? "text-sky-400" : "text-amber-400"}`}>
            {activeItem.build_draw_1w > 0 ? "+" : ""}{activeItem.build_draw_1w}
          </div>
          <div className="text-[11px] text-[var(--tx-2)]">
            {activeItem.build_draw_1w >= 0 ? "Weekly Inventory Build" : "Weekly Inventory Draw"}
          </div>
        </div>
      </div>

      {/* Lag & Data Provenance Disclosure */}
      <div className="p-3 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] text-xs font-mono text-[var(--tx-2)] flex items-center gap-2">
        <Clock className="w-4 h-4 text-sky-400 shrink-0" />
        <span><b>Data Provenance:</b> {fund.lag_label}</span>
      </div>
    </div>
  );
}
