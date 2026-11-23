// web/src/components/charts/MoveDecomposerWaterfall.tsx
"use client";

import React, { useState } from "react";
import ReactECharts from "echarts-for-react";
import { DecompositionData } from "@/lib/data/schemas";
import { Info, AlertCircle, CheckCircle2 } from "lucide-react";

interface MoveDecomposerWaterfallProps {
  decomposition: DecompositionData;
}

export const MoveDecomposerWaterfall: React.FC<MoveDecomposerWaterfallProps> = ({
  decomposition,
}) => {
  // Prepare items for waterfall
  const factorKeys = Object.keys(decomposition.contributions);
  const items = factorKeys.map((key) => {
    const val = decomposition.contributions[key] * 100.0;
    const interval = decomposition.intervals[key]
      ? [decomposition.intervals[key][0] * 100.0, decomposition.intervals[key][1] * 100.0]
      : [val, val];
    const label = key
      .replace(/_/g, " ")
      .replace("row equity", "Rest-of-World Equity")
      .replace("brent", "Brent Crude")
      .replace("copper", "Dr. Copper")
      .replace("gold", "Gold")
      .replace("dxy", "US Dollar Index")
      .replace("us10y", "US 10Y Yield")
      .replace("fx effect", "Currency Effect (FX)");

    return {
      name: label,
      rawKey: key,
      value: Number(val.toFixed(2)),
      ciLower: Number(interval[0].toFixed(2)),
      ciUpper: Number(interval[1].toFixed(2)),
      isResidual: false,
    };
  });

  // Append residual
  const resVal = decomposition.residual * 100.0;
  items.push({
    name: "Unexplained Residual",
    rawKey: "residual",
    value: Number(resVal.toFixed(2)),
    ciLower: Number(resVal.toFixed(2)),
    ciUpper: Number(resVal.toFixed(2)),
    isResidual: true,
  });

  // Sort: biggest drivers first, residual last
  const drivers = items.slice(0, -1).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const sortedItems = [...drivers, items[items.length - 1]];

  const categories = sortedItems.map((it) => it.name);
  const values = sortedItems.map((it) => it.value);

  // ECharts horizontal bar config with whiskers
  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: "#16181e",
      borderColor: "#262933",
      textStyle: { color: "#f1f5f9", fontSize: 12 },
      formatter: (params: any) => {
        const p = params[0];
        const it = sortedItems[p.dataIndex];
        const sign = it.value >= 0 ? "+" : "";
        return `
          <div class="font-sans font-medium text-xs text-slate-200">
            <span class="text-slate-400 block mb-1">${it.name}</span>
            <div class="font-mono text-sm font-semibold">${sign}${it.value.toFixed(2)}%</div>
            ${
              !it.isResidual
                ? `<div class="text-[11px] font-mono text-slate-400 mt-1">95% Interval: [${it.ciLower.toFixed(
                    2
                  )}%, ${it.ciUpper.toFixed(2)}%]</div>`
                : `<div class="text-[11px] font-mono text-slate-400 mt-1">Unexplained market divergence</div>`
            }
          </div>
        `;
      },
    },
    grid: {
      left: "28%",
      right: "10%",
      top: "8%",
      bottom: "8%",
      containLabel: false,
    },
    xAxis: {
      type: "value",
      axisLabel: {
        color: "#94a3b8",
        fontSize: 11,
        fontFamily: "JetBrains Mono, monospace",
        formatter: "{value}%",
      },
      splitLine: {
        lineStyle: { color: "rgba(255, 255, 255, 0.05)" },
      },
      axisLine: {
        lineStyle: { color: "#334155" },
      },
    },
    yAxis: {
      type: "category",
      data: categories,
      axisLabel: {
        color: "#cbd5e1",
        fontSize: 11,
        fontFamily: "Inter, sans-serif",
      },
      axisTick: { show: false },
      axisLine: {
        lineStyle: { color: "#334155" },
      },
    },
    series: [
      {
        name: "Contribution",
        type: "bar",
        data: sortedItems.map((it) => ({
          value: it.value,
          itemStyle: {
            color: it.isResidual
              ? "rgba(148, 163, 184, 0.45)"
              : it.value >= 0
              ? "#34d399"
              : "#f87171",
            borderRadius: [2, 2, 2, 2],
          },
        })),
        barWidth: 16,
      },
    ],
  };

  return (
    <div className="space-y-4">
      {/* Narrative Headline Strip */}
      <div className="p-3.5 rounded-md bg-[var(--bg-0)] border border-[var(--line)] text-xs text-[var(--tx-1)] flex items-start gap-3 leading-relaxed">
        <Info className="w-4 h-4 text-[var(--accent)] shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-serif text-sm text-[var(--tx-0)] font-medium">
            {decomposition.narrative}
          </p>
          <div className="flex items-center gap-3 text-[11px] font-mono text-[var(--tx-2)]">
            <span>
              Prior Window: {decomposition.window_dates.prior_start} to{" "}
              {decomposition.window_dates.prior_end}
            </span>
            <span>•</span>
            <span>
              Out-of-sample R²:{" "}
              <strong className="text-[var(--tx-0)]">
                {decomposition.r2_out_sample.toFixed(2)}
              </strong>
            </span>
            {(decomposition as any).estimator && (
              <span
                className="px-1.5 py-0.2 rounded text-[10px] font-bold uppercase tracking-wider bg-purple-950/50 text-purple-300 border border-purple-800/60"
                title={(decomposition as any).estimator_eval?.rationale || "State-space time-varying beta estimator"}
              >
                {(decomposition as any).estimator === "kalman" ? "⚡ Kalman Filter Betas" : "Rolling OLS"}
              </span>
            )}
            {decomposition.mostly_unexplained && (
              <span className="text-amber-400 font-semibold flex items-center gap-1">
                <AlertCircle className="w-3 h-3" /> Mostly Unexplained
              </span>
            )}
            {decomposition.regularized && (
              <span className="text-cyan-400 font-semibold">• Ridge Regularized</span>
            )}
          </div>
        </div>
      </div>

      {/* Attribution Provenance Badge */}
      <div className="flex items-center justify-between text-xs font-mono text-[var(--tx-2)]">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[var(--accent)]" />
          <span>Attribution Window: <strong>{decomposition.horizon_days} Trading Days</strong></span>
        </span>
        <span className="px-2 py-0.5 rounded bg-[var(--bg-0)] border border-[var(--line)] text-[11px] text-[var(--accent)] font-medium">
          USD Lens (Explicit FX Factor Included)
        </span>
      </div>

      {/* Horizontal Waterfall Chart */}
      <div className="rounded border border-[var(--line)] bg-[var(--bg-0)]/50 p-2">
        <ReactECharts option={option} style={{ height: "320px", width: "100%" }} />
      </div>

      {/* Total Move Summary Bar */}
      <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] flex items-center justify-between text-xs font-mono">
        <span className="text-[var(--tx-2)]">Total Move ({decomposition.target}):</span>
        <span
          className={`text-sm font-semibold ${
            decomposition.total_return >= 0 ? "text-emerald-400" : "text-rose-400"
          }`}
        >
          {decomposition.total_return >= 0 ? "+" : ""}
          {(decomposition.total_return * 100.0).toFixed(2)}%
        </span>
      </div>
    </div>
  );
};
