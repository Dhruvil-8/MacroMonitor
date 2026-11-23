// web/src/components/charts/CusumChangeChart.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { CusumData } from "@/lib/data/schemas";
import { AlertTriangle, CheckCircle2 } from "lucide-react";

export interface CusumChangeChartProps {
  cusum: CusumData;
  pairName: string;
  className?: string;
  compact?: boolean;
}

export const CusumChangeChart: React.FC<CusumChangeChartProps> = ({
  cusum,
  pairName,
  className = "",
  compact = false,
}) => {
  const latestPos = cusum.cusum_pos.length > 0 ? cusum.cusum_pos[cusum.cusum_pos.length - 1] : 0;
  const latestNeg = cusum.cusum_neg.length > 0 ? cusum.cusum_neg[cusum.cusum_neg.length - 1] : 0;
  const hasRecentShift = cusum.shifts.length > 0;

  const option = useMemo(() => {
    // Format dates for display
    const formattedDates = cusum.dates.map((d) => {
      const parts = d.split("-");
      return parts.length >= 3 ? `${parts[1]}/${parts[2]}` : d;
    });

    // Only plot markPoints for shifts that actually fall within cusum.dates
    const inWindowShifts = cusum.shifts.filter((s) => cusum.dates.includes(s.date));
    const upShifts = inWindowShifts.filter((s) => s.direction.toLowerCase().startsWith("up"));
    const downShifts = inWindowShifts.filter((s) => s.direction.toLowerCase().startsWith("down"));

    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: compact
        ? {
            top: 15,
            right: 15,
            bottom: 22,
            left: 30,
            containLabel: true,
          }
        : {
            top: 28,
            right: 25,
            bottom: 25,
            left: 35,
            containLabel: true,
          },
      legend: compact
        ? undefined
        : {
            data: ["Upward CUSUM (S⁺)", "Downward CUSUM (S⁻)"],
            top: 0,
            right: 10,
            textStyle: { color: "#8a99ad", fontSize: 10, fontFamily: "monospace" },
            icon: "roundRect",
          },
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0c1017",
        borderColor: "#1d2633",
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return "";
          const idx = params[0].dataIndex;
          const rawDate = cusum.dates[idx] || params[0].axisValue;
          const pos = params.find((p: any) => p.seriesName?.includes("S⁺"))?.value ?? 0;
          const neg = params.find((p: any) => p.seriesName?.includes("S⁻"))?.value ?? 0;
          const shift = cusum.shifts.find((s) => s.date === rawDate);

          let html = `<div style="font-weight:600;margin-bottom:4px;color:#cbd5e1;">${rawDate}</div>`;
          html += `<div><span style="color:#22c55e;">S⁺ (Upward):</span> <b>${Number(pos).toFixed(2)}σ</b></div>`;
          html += `<div><span style="color:#ef4444;">S⁻ (Downward):</span> <b>${Number(neg).toFixed(2)}σ</b></div>`;
          html += `<div style="color:#64748b;font-size:10px;margin-top:2px;">Threshold h = ${cusum.threshold.toFixed(1)}σ</div>`;
          if (shift) {
            html += `<div style="color:#f59e0b;font-weight:bold;margin-top:4px;">⚠️ Shift Flagged: ${shift.direction.toUpperCase()} (${shift.value.toFixed(2)}σ)</div>`;
          }
          return html;
        },
      },
      xAxis: {
        type: "category",
        data: formattedDates,
        axisLine: { lineStyle: { color: "#2a3647" } },
        axisLabel: {
          color: "#6f7c90",
          fontSize: 10,
          interval: Math.max(1, Math.floor(cusum.dates.length / 5)),
        },
      },
      yAxis: {
        type: "value",
        min: 0,
        splitLine: { lineStyle: { color: "#1d2633", opacity: 0.6 } },
        axisLabel: {
          color: "#6f7c90",
          fontSize: 10,
          formatter: "{value}σ",
        },
      },
      series: [
        {
          name: "Upward CUSUM (S⁺)",
          type: "line",
          data: cusum.cusum_pos,
          lineStyle: { color: "#22c55e", width: 2 },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(34, 197, 94, 0.25)" },
                { offset: 1, color: "rgba(34, 197, 94, 0.0)" },
              ],
            },
          },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "#e8765a", type: "dashed", width: 1 },
            label: {
              show: true,
              position: "insideEndTop",
              formatter: `h = ${cusum.threshold.toFixed(1)}σ`,
              color: "#e8765a",
              fontSize: 9,
              fontFamily: "monospace",
            },
            data: [{ yAxis: cusum.threshold }],
          },
          markPoint: upShifts.length > 0 ? {
            symbol: "pin",
            symbolSize: compact ? 22 : 28,
            itemStyle: { color: "#f59e0b" },
            label: {
              show: !compact,
              formatter: "Shift",
              fontSize: 8,
              color: "#000",
              fontWeight: "bold",
            },
            data: upShifts.map((s) => {
              const idx = cusum.dates.indexOf(s.date);
              return {
                name: "Upward Regime Shift",
                coord: [idx, s.value],
                value: s.value,
              };
            }),
          } : undefined,
          showSymbol: false,
        },
        {
          name: "Downward CUSUM (S⁻)",
          type: "line",
          data: cusum.cusum_neg,
          lineStyle: { color: "#ef4444", width: 2 },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(239, 68, 68, 0.25)" },
                { offset: 1, color: "rgba(239, 68, 68, 0.0)" },
              ],
            },
          },
          markPoint: downShifts.length > 0 ? {
            symbol: "pin",
            symbolSize: compact ? 22 : 28,
            itemStyle: { color: "#ef4444" },
            label: {
              show: !compact,
              formatter: "Shift",
              fontSize: 8,
              color: "#fff",
              fontWeight: "bold",
            },
            data: downShifts.map((s) => {
              const idx = cusum.dates.indexOf(s.date);
              return {
                name: "Downward Regime Shift",
                coord: [idx, s.value],
                value: s.value,
              };
            }),
          } : undefined,
          showSymbol: false,
        },
      ],
    };
  }, [cusum, compact]);

  // In compact mode (inside DivergencePanel card), render only the chart container
  if (compact) {
    return (
      <div style={{ width: "100%", height: 160 }} className={`w-full ${className}`}>
        <ReactECharts
          option={option}
          style={{ height: "160px", width: "100%" }}
          opts={{ renderer: "canvas" }}
          notMerge={true}
          lazyUpdate={true}
        />
      </div>
    );
  }

  // Full standalone mode
  return (
    <div className={`p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3 ${className}`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase tracking-widest text-[var(--accent)] font-semibold">
              C24 / Control Chart
            </span>
            <span className="text-xs font-semibold text-[var(--tx-0)] font-mono">
              {pairName}
            </span>
          </div>
          <p className="text-[11px] text-[var(--tx-2)] mt-0.5">
            Two-sided cumulative sum on standardized momentum innovations (slack k = 0.5, limit h = {cusum.threshold.toFixed(1)}σ)
          </p>
        </div>

        <div className="flex items-center gap-2">
          {hasRecentShift ? (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-[var(--warn)]/10 text-[var(--warn)] border border-[var(--warn)]/30 font-medium">
              <AlertTriangle className="w-3 h-3" />
              {cusum.shifts.length} Regime Shift{cusum.shifts.length > 1 ? "s" : ""}
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-[var(--pos)]/10 text-[var(--pos)] border border-[var(--pos)]/30 font-medium">
              <CheckCircle2 className="w-3 h-3" />
              In Control
            </span>
          )}
        </div>
      </div>

      <div style={{ width: "100%", height: 220 }} className="w-full">
        <ReactECharts
          option={option}
          style={{ height: "220px", width: "100%" }}
          opts={{ renderer: "canvas" }}
          notMerge={true}
          lazyUpdate={true}
        />
      </div>

      {/* Metric footer */}
      <div className="grid grid-cols-4 gap-2 pt-2 border-t border-[var(--line)] text-center font-mono">
        <div>
          <div className="text-[10px] text-[var(--tx-2)] uppercase">S⁺ (Up)</div>
          <div className={`text-xs font-semibold ${latestPos >= cusum.threshold ? "text-[var(--warn)]" : "text-[var(--pos)]"}`}>
            {latestPos.toFixed(2)}σ
          </div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--tx-2)] uppercase">S⁻ (Down)</div>
          <div className={`text-xs font-semibold ${latestNeg >= cusum.threshold ? "text-[var(--warn)]" : "text-[var(--neg)]"}`}>
            {latestNeg.toFixed(2)}σ
          </div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--tx-2)] uppercase">Limit h</div>
          <div className="text-xs font-semibold text-[var(--tx-0)]">
            {cusum.threshold.toFixed(1)}σ
          </div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--tx-2)] uppercase">Shifts</div>
          <div className={`text-xs font-semibold ${cusum.shifts.length > 0 ? "text-[var(--warn)]" : "text-[var(--tx-0)]"}`}>
            {cusum.shifts.length}
          </div>
        </div>
      </div>
    </div>
  );
};
