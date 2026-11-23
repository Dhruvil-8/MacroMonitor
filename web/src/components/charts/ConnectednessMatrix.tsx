// web/src/components/charts/ConnectednessMatrix.tsx
"use client";

import React, { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { ConnectednessEnvelope, ConnectednessCountryItem, TciPoint } from "@/lib/data/schemas";
import { EChartBox } from "@/components/charts/EChartBox";
import { useRadarStore } from "@/store/useRadarStore";
import { Activity, ArrowDownRight, ArrowUpRight, Globe, Layers, Network, ShieldCheck } from "lucide-react";

interface Props {
  data: ConnectednessEnvelope;
}

export function ConnectednessMatrix({ data }: Props) {
  const theme = useRadarStore((s) => s.theme);
  const conn = (data as any).data || data;

  const [activeTab, setActiveTab] = useState<"matrix" | "net_bars" | "tci_timeline">("matrix");

  const textCol = "#8c9bb0";
  const gridLine = "#1d2633";

  const labels: string[] = conn.labels || conn.countries?.map((c: ConnectednessCountryItem) => c.name) || [];

  // ECharts Option for Net Spillover Diverging Bars
  const netOption = useMemo<EChartsOption>(() => {
    if (!conn.countries || conn.countries.length === 0) return {};

    const countries = conn.countries.map((c: ConnectednessCountryItem) => c.name);
    const netVals = conn.countries.map((c: ConnectednessCountryItem) => c.net);

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: "#0c1017",
        borderColor: gridLine,
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          const item = params[0];
          const c = conn.countries[item.dataIndex];
          return `<b>${c.name}</b><br/>Net Spillover: <b>${c.net > 0 ? "+" : ""}${c.net.toFixed(2)}%</b><br/>Transmitted to Others: ${c.to_others.toFixed(2)}%<br/>Received from Others: ${c.from_others.toFixed(2)}%<br/>Role: <i>${c.role}</i>`;
        },
      },
      grid: { left: 80, right: 30, top: 20, bottom: 25 },
      xAxis: {
        type: "value",
        name: "Net Spillover % (To - From)",
        nameTextStyle: { color: textCol, fontSize: 10 },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace", formatter: "{value}%" },
      },
      yAxis: {
        type: "category",
        data: countries,
        axisLabel: { color: textCol, fontSize: 11, fontFamily: "monospace" },
        axisLine: { lineStyle: { color: gridLine } },
      },
      series: [
        {
          name: "Net Spillover",
          type: "bar",
          data: netVals.map((v: number) => ({
            value: v,
            itemStyle: {
              color: v >= 0 ? "#34d399" : "#f87171",
              borderRadius: v >= 0 ? [0, 3, 3, 0] : [3, 0, 0, 3],
            },
          })),
        },
      ],
    };
  }, [conn]);

  // ECharts Option for TCI Timeline
  const tciOption = useMemo<EChartsOption>(() => {
    if (!conn.tci_history || conn.tci_history.length === 0) return {};

    const dates = conn.tci_history.map((h: TciPoint) => h.date);
    const tcis = conn.tci_history.map((h: TciPoint) => h.tci);

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0c1017",
        borderColor: gridLine,
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) =>
          `<b>${params[0].name}</b><br/>Total Connectedness (TCI): <b>${params[0].value}%</b>`,
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
        name: "Total Connectedness Index (%)",
        nameTextStyle: { color: textCol, fontSize: 10 },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace", formatter: "{value}%" },
      },
      series: [
        {
          name: "TCI",
          type: "line",
          data: tcis,
          lineStyle: { width: 2.2, color: "#38bdf8" },
          showSymbol: false,
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "#64748b", type: "dashed" },
            data: [{ yAxis: conn.tci, label: { formatter: `Current TCI (${conn.tci}%)`, color: "#38bdf8" } }],
          },
        },
      ],
    };
  }, [conn]);

  return (
    <div className="space-y-6">
      {/* Top Narrative Strip */}
      <div className="p-4 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono uppercase text-[var(--tx-2)]">System Connectedness:</span>
            <span className="text-xs px-2.5 py-0.5 rounded border border-sky-800 bg-sky-950/70 text-sky-400 font-mono font-bold uppercase tracking-wider">
              TCI: {conn.tci}% (H={conn.horizon_weeks}w)
            </span>
            <span className="text-xs text-[var(--tx-2)] font-mono">
              (Order-Invariant Pesaran-Shin GFEVD)
            </span>
          </div>
          <p className="text-xs text-[var(--tx-1)] font-mono">{conn.interpretation}</p>
        </div>

        <div className="flex rounded border border-[var(--line)] overflow-hidden font-mono text-xs self-start md:self-auto">
          <button
            onClick={() => setActiveTab("matrix")}
            className={`px-3 py-1 transition-colors ${
              activeTab === "matrix"
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
            }`}
          >
            Spillover Matrix
          </button>
          <button
            onClick={() => setActiveTab("net_bars")}
            className={`px-3 py-1 transition-colors ${
              activeTab === "net_bars"
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
            }`}
          >
            Net Transmitters
          </button>
          <button
            onClick={() => setActiveTab("tci_timeline")}
            className={`px-3 py-1 transition-colors ${
              activeTab === "tci_timeline"
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
            }`}
          >
            TCI History
          </button>
        </div>
      </div>

      {/* Main View Area */}
      {activeTab === "matrix" ? (
        <div className="bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)] space-y-3 font-mono text-xs">
          <div className="flex items-center justify-between border-b border-[var(--line)] pb-2">
            <div className="flex items-center gap-2">
              <Network className="w-4 h-4 text-[var(--accent)]" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--tx-0)]">
                Diebold-Yilmaz Directed Spillover Matrix (% of Forecast Error Variance)
              </h3>
            </div>
            <span className="text-[10px] text-[var(--tx-2)]">Rows = From • Columns = To • Sums to 100%</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-center border-collapse">
              <thead>
                <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                  <th className="p-2 text-left">From \ To</th>
                  {labels.map((l: string) => (
                    <th key={l} className="p-2 font-bold text-[var(--tx-0)]">
                      {l}
                    </th>
                  ))}
                  <th className="p-2 text-sky-400 font-bold">From Others</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--line)]">
                {conn.matrix?.map((row: any, i: number) => {
                  const fromName = row.from_country || labels[i] || `Country ${i + 1}`;
                  const cMetric = conn.countries?.find((c: any) => c.name === fromName);
                  return (
                    <tr key={i} className="hover:bg-[var(--bg-1)]/40 transition-colors">
                      <td className="p-2 text-left font-bold text-[var(--tx-0)]">{fromName}</td>
                      {labels.map((colName: string, colIdx: number) => {
                        const val = Array.isArray(row) ? row[colIdx] : (row[colName] ?? 0);
                        const isDiag = fromName === colName;
                        return (
                          <td
                            key={colName}
                            className={`p-2 ${
                              isDiag
                                ? "bg-slate-800/40 text-slate-300 font-semibold"
                                : val > 15
                                ? "bg-rose-950/40 text-rose-300 font-bold"
                                : val > 8
                                ? "text-[var(--tx-0)]"
                                : "text-[var(--tx-2)]"
                            }`}
                          >
                            {val.toFixed(1)}%
                          </td>
                        );
                      })}
                      <td className="p-2 text-sky-400 font-bold">
                        {cMetric ? `${cMetric.from_others.toFixed(1)}%` : "—"}
                      </td>
                    </tr>
                  );
                })}
                {/* To Others Summary Row */}
                <tr className="border-t-2 border-[var(--line)] bg-[var(--bg-1)] text-emerald-400 font-bold">
                  <td className="p-2 text-left">To Others</td>
                  {labels.map((colName: string) => {
                    const cMetric = conn.countries?.find((c: any) => c.name === colName);
                    return (
                      <td key={colName} className="p-2">
                        {cMetric ? `${cMetric.to_others.toFixed(1)}%` : "—"}
                      </td>
                    );
                  })}
                  <td className="p-2 text-[var(--accent)] font-extrabold">TCI: {conn.tci}%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      ) : activeTab === "net_bars" ? (
        <div className="bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)]">
          <div className="flex items-center justify-between mb-3 border-b border-[var(--line)] pb-2">
            <div className="flex items-center gap-2">
              <Globe className="w-4 h-4 text-[var(--accent)]" />
              <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-[var(--tx-0)]">
                Net Directional Spillovers: Transmitters (+ / Green) vs. Receivers (- / Red)
              </h3>
            </div>
            <span className="text-[10px] font-mono text-[var(--tx-2)]">Net = (To Others) - (From Others)</span>
          </div>

          <div className="h-80 w-full">
            <EChartBox option={netOption} ariaLabel="Net Directional Spillovers Chart" />
          </div>
        </div>
      ) : (
        <div className="bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)]">
          <div className="flex items-center justify-between mb-3 border-b border-[var(--line)] pb-2">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-[var(--accent)]" />
              <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-[var(--tx-0)]">
                Total Connectedness Index (TCI) Historical Evolution
              </h3>
            </div>
            <span className="text-[10px] font-mono text-[var(--tx-2)]">Rolling 104-week window • H=10 weeks</span>
          </div>

          <div className="h-80 w-full">
            <EChartBox option={tciOption} ariaLabel="Total Connectedness Index Historical Evolution Chart" />
          </div>
        </div>
      )}

      {/* Institutional Caveats & Method Panel */}
      <div className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] text-xs font-mono text-[var(--tx-2)] flex items-start gap-2.5">
        <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <strong className="text-[var(--tx-0)]">Methodological Discipline:</strong>
          <p>
            Spillover decomposition utilizes order-invariant generalized forecast error variance decomposition (Pesaran & Shin 1998; Diebold & Yilmaz 2012).
            Shocks are statistical innovations from weekly USD-denominated returns and reflect transmission elasticity rather than causal structural policy claims.
          </p>
        </div>
      </div>
    </div>
  );
}
