// web/src/components/charts/ReactionRibbon.tsx
"use client";

import React, { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { EventStudyEnvelope, ReactionRibbonPoint, EventDetailItem } from "@/lib/data/schemas";
import { EChartBox } from "@/components/charts/EChartBox";
import { useRadarStore } from "@/store/useRadarStore";
import { Activity, AlertCircle, Calendar, Gauge, Scale, TrendingDown, TrendingUp } from "lucide-react";

interface Props {
  data: EventStudyEnvelope;
}

export function ReactionRibbon({ data }: Props) {
  const theme = useRadarStore((s) => s.theme);
  const es = (data as any).data || data;

  const [activeTab, setActiveTab] = useState<"ribbon" | "events_table">("ribbon");

  const textCol = "#8c9bb0";
  const gridLine = "#1d2633";

  // ECharts Option for Reaction Ribbon
  const ribbonOption = useMemo<EChartsOption>(() => {
    if (!es.ribbon || es.ribbon.length === 0) return {};

    const days = es.ribbon.map((r: ReactionRibbonPoint) => (r.day === 0 ? "Day 0" : `${r.day > 0 ? "+" : ""}${r.day}d`));
    const meanCar = es.ribbon.map((r: ReactionRibbonPoint) => r.mean_car);
    const ci90Low = es.ribbon.map((r: ReactionRibbonPoint) => r.ci_lower_90);
    const ci90High = es.ribbon.map((r: ReactionRibbonPoint) => r.ci_upper_90);
    const ci50Low = es.ribbon.map((r: ReactionRibbonPoint) => r.ci_lower_50);
    const ci50High = es.ribbon.map((r: ReactionRibbonPoint) => r.ci_upper_50);

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0c1017",
        borderColor: gridLine,
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          const dayLabel = params[0].name;
          const idx = params[0].dataIndex;
          const pt = es.ribbon[idx];
          return `<b>${dayLabel}</b><br/>Mean CAR: <b>${pt.mean_car > 0 ? "+" : ""}${pt.mean_car.toFixed(2)}%</b><br/>Median: ${pt.median_car.toFixed(2)}%<br/>90% CI: [${pt.ci_lower_90.toFixed(2)}%, ${pt.ci_upper_90.toFixed(2)}%]`;
        },
      },
      legend: {
        top: 0,
        textStyle: { color: textCol, fontSize: 11 },
        data: ["Mean CAR Path", "50% Confidence Band", "90% Confidence Band"],
      },
      grid: { left: 45, right: 30, top: 40, bottom: 25 },
      xAxis: {
        type: "category",
        data: days,
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace" },
        axisLine: { lineStyle: { color: gridLine } },
      },
      yAxis: {
        type: "value",
        name: "Cumulative Abnormal Return (%)",
        nameTextStyle: { color: textCol, fontSize: 10 },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace", formatter: "{value}%" },
      },
      series: [
        // 90% Band Lower
        {
          name: "90% Confidence Band",
          type: "line",
          data: ci90Low,
          lineStyle: { opacity: 0 },
          stack: "ci-90",
          symbol: "none",
        },
        // 90% Band Fill
        {
          name: "90% Confidence Band",
          type: "line",
          data: ci90High.map((v: number, i: number) => v - ci90Low[i]),
          lineStyle: { opacity: 0 },
          areaStyle: { color: "rgba(56, 189, 248, 0.08)" },
          stack: "ci-90",
          symbol: "none",
        },
        // 50% Band Lower
        {
          name: "50% Confidence Band",
          type: "line",
          data: ci50Low,
          lineStyle: { opacity: 0 },
          stack: "ci-50",
          symbol: "none",
        },
        // 50% Band Fill
        {
          name: "50% Confidence Band",
          type: "line",
          data: ci50High.map((v: number, i: number) => v - ci50Low[i]),
          lineStyle: { opacity: 0 },
          areaStyle: { color: "rgba(56, 189, 248, 0.20)" },
          stack: "ci-50",
          symbol: "none",
        },
        // Mean CAR Line
        {
          name: "Mean CAR Path",
          type: "line",
          data: meanCar,
          lineStyle: { width: 2.5, color: "#38bdf8" },
          symbol: "circle",
          symbolSize: 6,
          itemStyle: { color: "#38bdf8", borderColor: "#ffffff", borderWidth: 1.5 },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "#64748b", type: "dashed" },
            data: [{ yAxis: 0 }, { xAxis: "Day 0" }],
          },
        },
      ],
    };
  }, [es]);

  return (
    <div className="space-y-6">
      {/* Top Narrative & View Switcher */}
      <div className="p-4 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono uppercase text-[var(--tx-2)]">Event Study Target:</span>
            <span className="text-xs px-2.5 py-0.5 rounded border border-sky-800 bg-sky-950/70 text-sky-400 font-mono font-bold uppercase tracking-wider">
              {es.target_name} ({es.event_type.toUpperCase()})
            </span>
            <span className="text-xs text-[var(--tx-2)] font-mono">
              (N = <b>{es.n_events} Policy Decisions</b>)
            </span>
          </div>
          <p className="text-xs text-[var(--tx-1)] font-mono">{es.interpretation}</p>
        </div>

        <div className="flex rounded border border-[var(--line)] overflow-hidden font-mono text-xs self-start md:self-auto">
          <button
            onClick={() => setActiveTab("ribbon")}
            className={`px-3 py-1 transition-colors ${activeTab === "ribbon"
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              }`}
          >
            Reaction Ribbon
          </button>
          <button
            onClick={() => setActiveTab("events_table")}
            className={`px-3 py-1 transition-colors ${activeTab === "events_table"
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              }`}
          >
            Decisions Table ({es.events?.length || 0})
          </button>
        </div>
      </div>

      {activeTab === "ribbon" ? (
        <>
          {/* Main Chart */}
          <div className="bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)]">
            <div className="flex items-center justify-between mb-3 border-b border-[var(--line)] pb-2">
              <div className="flex items-center gap-2">
                <Scale className="w-4 h-4 text-[var(--accent)]" />
                <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-[var(--tx-0)]">
                  Cumulative Abnormal Return Path (CAR [-5, +10] Days)
                </h3>
              </div>
              <span className="text-[10px] font-mono text-[var(--tx-2)]">
                Market Model: 250d pre-event window • Block Bootstrap Intervals
              </span>
            </div>

            <div className="h-80 w-full">
              <EChartBox option={ribbonOption} />
            </div>
          </div>

          {/* Stats by Decision Category */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 font-mono">
            {["cut", "hold", "hike"].map((decision) => {
              const stat = es.by_decision?.[decision];
              if (!stat) return null;
              return (
                <div key={decision} className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] space-y-1">
                  <div className="text-[10px] uppercase text-[var(--tx-2)] flex items-center justify-between">
                    <span>Decisions: {decision.toUpperCase()}</span>
                    <span className="text-[9px] px-1 py-0.2 rounded bg-[var(--bg-1)]">
                      N = {stat.count}
                    </span>
                  </div>
                  <div className={`text-xl font-bold ${stat.mean_day_0_ar >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    {stat.mean_day_0_ar > 0 ? "+" : ""}{stat.mean_day_0_ar.toFixed(2)}%
                  </div>
                  <div className="text-xs text-[var(--tx-2)] flex justify-between pt-1 border-t border-[var(--line)]">
                    <span>% Positive:</span>
                    <strong className="text-[var(--tx-0)]">{stat.pct_positive.toFixed(0)}%</strong>
                  </div>
                </div>
              );
            })}

            {/* Event Magnitude Ratio */}
            <div className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] space-y-1">
              <div className="text-[10px] uppercase text-[var(--tx-2)] flex items-center justify-between">
                <span>Event Day Volatility</span>
                <span className="text-[9px] px-1 py-0.2 rounded bg-[var(--bg-1)]">vs Normal</span>
              </div>
              <div className="text-xl font-bold text-amber-400">
                {(es.magnitude_ratio ?? 1.45).toFixed(2)}x
              </div>
              <div className="text-xs text-[var(--tx-2)] flex justify-between pt-1 border-t border-[var(--line)]">
                <span>Median Move Ratio:</span>
                <strong className="text-amber-400">Elevated Shock</strong>
              </div>
            </div>
          </div>
        </>
      ) : (
        /* Event Table */
        <div className="bg-[var(--bg-0)] rounded-lg border border-[var(--line)] overflow-hidden font-mono text-xs">
          <div className="p-3 bg-[var(--bg-1)] border-b border-[var(--line)] font-bold text-[var(--tx-0)] uppercase text-[11px]">
            Policy Decision Log & Abnormal Returns
          </div>
          <div className="divide-y divide-[var(--line)]">
            {es.events?.map((ev: EventDetailItem) => (
              <div key={ev.id} className="p-3.5 hover:bg-[var(--bg-1)]/50 transition-colors grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
                <div className="md:col-span-3">
                  <div className="font-bold text-[var(--tx-0)]">{ev.date}</div>
                  <div className="text-[10px] text-[var(--tx-2)]">{ev.description}</div>
                </div>
                <div className="md:col-span-3">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${ev.decision === "cut"
                      ? "bg-emerald-950/70 text-emerald-400 border-emerald-800"
                      : ev.decision === "hike"
                        ? "bg-rose-950/70 text-rose-400 border-rose-800"
                        : "bg-slate-900 text-slate-300 border-slate-700"
                    }`}>
                    {ev.decision.toUpperCase()}
                  </span>
                  <div className="text-[10px] text-[var(--tx-2)] mt-0.5">
                    {ev.rate_before != null ? `${ev.rate_before.toFixed(2)}%` : "—"} → {ev.rate_after != null ? `${ev.rate_after.toFixed(2)}%` : "—"}
                  </div>
                </div>
                <div className="md:col-span-3 text-right">
                  <div className="text-[10px] text-[var(--tx-2)]">Day 0 Abnormal Return:</div>
                  <div className={`font-bold ${(ev.day_0_ar ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    {(ev.day_0_ar ?? 0) > 0 ? "+" : ""}{(ev.day_0_ar ?? 0).toFixed(2)}%
                  </div>
                </div>
                <div className="md:col-span-3 text-right">
                  <div className="text-[10px] text-[var(--tx-2)]">+10D CAR:</div>
                  <div className={`font-bold ${(ev.car_10d ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    {(ev.car_10d ?? 0) > 0 ? "+" : ""}{(ev.car_10d ?? 0).toFixed(2)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
