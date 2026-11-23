// web/src/components/charts/DivergencePanel.tsx
"use client";

import React, { useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { DivergenceItem } from "@/lib/data/schemas";
import { AlertTriangle, ShieldCheck } from "lucide-react";
import { AuditData } from "../ui/AuditDrawer";
import { CusumChangeChart } from "./CusumChangeChart";

interface DivergencePanelProps {
  divergence: DivergenceItem;
  testedPairsCount?: number;
  onOpenAudit?: (data: AuditData) => void;
}

export const DivergencePanel: React.FC<DivergencePanelProps> = ({
  divergence,
  testedPairsCount = 10,
  onOpenAudit,
}) => {
  const [viewMode, setViewMode] = useState<"spread" | "cusum">("spread");

  const formattedDates = useMemo(() => {
    return divergence.history_dates.map((d) => {
      const parts = d.split("-");
      return parts.length >= 3 ? `${parts[1]}/${parts[2]}` : d;
    });
  }, [divergence.history_dates]);

  const option = useMemo(() => {
    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: {
        top: 15,
        right: 15,
        bottom: 22,
        left: 30,
        containLabel: true,
      },
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0c1017",
        borderColor: "#1d2633",
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return "";
          const idx = params[0].dataIndex;
          const rawDate = divergence.history_dates[idx] || params[0].axisValue;
          const val = params[0].value;
          return `<div style="font-weight:600;margin-bottom:4px;color:#cbd5e1;">${rawDate}</div>
            <div>Z-Spread: <b>${val > 0 ? "+" : ""}${Number(val).toFixed(2)}σ</b></div>`;
        },
      },
      xAxis: {
        type: "category",
        data: formattedDates,
        axisLine: { lineStyle: { color: "#2a3647" } },
        axisLabel: {
          color: "#6f7c90",
          fontSize: 10,
          interval: Math.max(1, Math.floor(divergence.history_dates.length / 5)),
        },
      },
      yAxis: {
        type: "value",
        splitLine: { lineStyle: { color: "#1d2633", opacity: 0.6 } },
        axisLabel: { color: "#6f7c90", fontSize: 10, formatter: "{value}σ" },
      },
      series: [
        {
          name: "Z-Divergence",
          type: "line",
          data: divergence.history_spread,
          lineStyle: {
            color: divergence.active ? "#e8765a" : "#6ea8ff",
            width: 2,
          },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: divergence.active ? "rgba(232, 118, 90, 0.22)" : "rgba(110, 168, 255, 0.18)" },
                { offset: 1, color: "rgba(0, 0, 0, 0.0)" },
              ],
            },
          },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "#2a3647", type: "dashed", width: 1 },
            data: [{ yAxis: 2.0 }, { yAxis: -2.0 }],
          },
          showSymbol: false,
        },
      ],
    };
  }, [divergence, formattedDates]);

  return (
    <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-[var(--tx-0)] font-mono">
              {divergence.asset_a.toUpperCase()} vs {divergence.asset_b.toUpperCase()}
            </h4>
            {divergence.active && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-[var(--neg)]/10 text-[var(--neg)] border border-[var(--neg)]/30 font-medium">
                <AlertTriangle className="w-3 h-3" />
                Diverging ({divergence.consecutive_days}d)
              </span>
            )}
            {divergence.cusum && divergence.cusum.shifts.length > 0 && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-[var(--warn)]/10 text-[var(--warn)] border border-[var(--warn)]/30 font-medium">
                CUSUM Shift ({divergence.cusum.shifts.length})
              </span>
            )}
          </div>
          <p className="text-xs text-[var(--tx-2)] mt-0.5">{divergence.rationale}</p>
        </div>

        <div className="flex items-center gap-2">
          {divergence.cusum && (
            <div className="flex rounded border border-[var(--line)] overflow-hidden font-mono text-[10px]">
              <button
                type="button"
                onClick={() => setViewMode("spread")}
                className={`px-2 py-0.5 transition-colors ${
                  viewMode === "spread"
                    ? "bg-[var(--accent)] text-white font-medium"
                    : "bg-[var(--bg-0)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                }`}
              >
                Spread
              </button>
              <button
                type="button"
                onClick={() => setViewMode("cusum")}
                className={`px-2 py-0.5 transition-colors ${
                  viewMode === "cusum"
                    ? "bg-[var(--accent)] text-white font-medium"
                    : "bg-[var(--bg-0)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                }`}
              >
                CUSUM
              </button>
            </div>
          )}

          {onOpenAudit && (
            <button
              type="button"
              onClick={() =>
                onOpenAudit({
                  metricName: `Divergence: ${divergence.asset_a.toUpperCase()} vs ${divergence.asset_b.toUpperCase()}`,
                  formula: "D_t = Z(\\text{mom}_{20, A}) - Z(\\text{mom}_{20, B})",
                  formulaPlain: "Difference between expanding standardized 20-day momentum rate of changes.",
                  parameters: { lookback: "20D", threshold_percentile: 95.0, consecutive_days: 3 },
                  inputs: [
                    { id: divergence.asset_a, name: divergence.asset_a.toUpperCase(), source: "Yahoo/FRED", unit: "index" },
                    { id: divergence.asset_b, name: divergence.asset_b.toUpperCase(), source: "Yahoo/FRED", unit: "index" },
                  ],
                  window: "20 business days",
                  currentValue: `${divergence.spread_z > 0 ? "+" : ""}${divergence.spread_z.toFixed(2)}σ`,
                  percentile: divergence.percentile,
                  codeVersion: "v1.0.0",
                  specUrl: "/methodology",
                })
              }
              className="p-1 rounded hover:bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--accent)]"
              title="Audit divergence metric"
            >
              <ShieldCheck className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Chart */}
      <div style={{ height: 140, width: "100%" }} className="w-full">
        {viewMode === "cusum" && divergence.cusum ? (
          <CusumChangeChart
            cusum={divergence.cusum}
            pairName={`${divergence.asset_a.toUpperCase()} vs ${divergence.asset_b.toUpperCase()}`}
            compact={true}
          />
        ) : (
          <ReactECharts
            option={option}
            style={{ height: "140px", width: "100%" }}
            opts={{ renderer: "canvas" }}
            notMerge={true}
            lazyUpdate={true}
          />
        )}
      </div>

      {/* Disclaimers & Multi-test caveat */}
      <div className="pt-2 border-t border-[var(--line)] flex flex-col sm:flex-row sm:items-center justify-between text-[11px] font-mono text-[var(--tx-2)] gap-1">
        <span>
          Current Spread: <strong className="text-[var(--tx-0)]">{divergence.spread_z > 0 ? "+" : ""}{divergence.spread_z.toFixed(2)}σ</strong> ({divergence.percentile.toFixed(0)}th pct)
          {divergence.cusum && (
            <span className="ml-2 text-[10px] text-[var(--tx-2)]">
              • CUSUM S⁺: {divergence.cusum.cusum_pos[divergence.cusum.cusum_pos.length - 1]?.toFixed(1) ?? "0.0"}σ | S⁻: {divergence.cusum.cusum_neg[divergence.cusum.cusum_neg.length - 1]?.toFixed(1) ?? "0.0"}σ
            </span>
          )}
        </span>
        <span className="text-[10px]">Statistical disagreement, not a trading signal • 1 of {testedPairsCount} pre-registered pairs</span>
      </div>
    </div>
  );
};
