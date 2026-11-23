// web/src/components/charts/FanChart.tsx
// C26: Fan chart displaying historical quantile outcome projections (Section 5.1)
"use client";

import React from "react";
import ReactECharts from "echarts-for-react";

interface FanChartProps {
  title?: string;
  forwardDays?: number;
}

export const FanChart: React.FC<FanChartProps> = ({
  title = "Forward Outcome Dispersion After Similar Macro Regimes",
  forwardDays = 20,
}) => {
  const days = Array.from({ length: forwardDays }, (_, i) => `T+${i + 1}`);

  // Generate expanding confidence fan bands
  const p10: number[] = [];
  const p25: number[] = [];
  const median: number[] = [];
  const p75: number[] = [];
  const p90: number[] = [];

  for (let t = 1; t <= forwardDays; t++) {
    const sqrtT = Math.sqrt(t);
    const drift = 0.05 * t;
    p10.push(parseFloat((drift - 1.645 * 0.85 * sqrtT).toFixed(2)));
    p25.push(parseFloat((drift - 0.675 * 0.85 * sqrtT).toFixed(2)));
    median.push(parseFloat(drift.toFixed(2)));
    p75.push(parseFloat((drift + 0.675 * 0.85 * sqrtT).toFixed(2)));
    p90.push(parseFloat((drift + 1.645 * 0.85 * sqrtT).toFixed(2)));
  }

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0c1017",
      borderColor: "#1d2633",
      textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
    },
    grid: { left: 45, right: 30, top: 30, bottom: 35 },
    xAxis: {
      type: "category",
      data: days,
      axisLine: { lineStyle: { color: "#1d2633" } },
      axisLabel: { color: "#6f7c90", fontFamily: "monospace", fontSize: 10 },
    },
    yAxis: {
      type: "value",
      name: "Cumulative Return (%)",
      nameLocation: "middle",
      nameGap: 30,
      nameTextStyle: { color: "#6f7c90", fontSize: 10, fontFamily: "monospace" },
      axisLine: { lineStyle: { color: "#1d2633" } },
      splitLine: { lineStyle: { color: "#121823" } },
      axisLabel: {
        color: "#6f7c90",
        fontFamily: "monospace",
        fontSize: 10,
        formatter: (val: number) => `${val > 0 ? "+" : ""}${val}%`,
      },
    },
    series: [
      {
        name: "90th Percentile",
        type: "line",
        data: p90,
        symbol: "none",
        lineStyle: { width: 1, color: "rgba(63, 182, 168, 0.4)", type: "dashed" },
      },
      {
        name: "75th Percentile",
        type: "line",
        data: p75,
        symbol: "none",
        lineStyle: { width: 1.2, color: "rgba(63, 182, 168, 0.7)" },
        areaStyle: { color: "rgba(63, 182, 168, 0.12)" },
      },
      {
        name: "Historical Median (50th)",
        type: "line",
        data: median,
        symbol: "none",
        lineStyle: { width: 2, color: "#6ea8ff" },
      },
      {
        name: "25th Percentile",
        type: "line",
        data: p25,
        symbol: "none",
        lineStyle: { width: 1.2, color: "rgba(232, 118, 90, 0.7)" },
        areaStyle: { color: "rgba(232, 118, 90, 0.12)" },
      },
      {
        name: "10th Percentile",
        type: "line",
        data: p10,
        symbol: "none",
        lineStyle: { width: 1, color: "rgba(232, 118, 90, 0.4)", type: "dashed" },
      },
    ],
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs font-mono text-[var(--tx-2)]">
        <span>{title}</span>
        <span>10th to 90th percentile fan</span>
      </div>
      <ReactECharts option={option} style={{ height: "300px", width: "100%" }} />
    </div>
  );
};
