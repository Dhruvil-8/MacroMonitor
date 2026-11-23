// web/src/components/charts/DownsideCorrelationChart.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { DownsideCorrelationItem } from "@/lib/data/schemas";

interface DownsideCorrelationChartProps {
  items: DownsideCorrelationItem[];
  benchmarkName?: string;
}

export const DownsideCorrelationChart: React.FC<DownsideCorrelationChartProps> = ({
  items,
  benchmarkName = "S&P 500",
}) => {
  const option = useMemo(() => {
    if (!items || items.length === 0) return {};

    // Sort ascending so highest stress premium sits at the top in horizontal bar
    const sorted = [...items].reverse();
    const categories = sorted.map((d) => d.target_name);
    const allDayCorr = sorted.map((d) => d.all_day_corr.toFixed(2));
    const downsideCorr = sorted.map((d) => d.downside_corr.toFixed(2));

    return {
      backgroundColor: "transparent",
      animationDuration: 400,
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: "rgba(10, 14, 20, 0.95)",
        borderColor: "rgba(255, 255, 255, 0.12)",
        textStyle: { color: "#e6edf3", fontSize: 12, fontFamily: "var(--font-mono)" },
        formatter: (params: any[]) => {
          const idx = params[0]?.dataIndex ?? 0;
          const item = sorted[idx];
          if (!item) return "";
          return `
            <div style="font-weight: 600; margin-bottom: 6px; color: #58a6ff;">${item.target_name} vs. ${benchmarkName}</div>
            <div style="display: flex; justify-content: space-between; gap: 16px;"><span>Downside Stress Corr (Worst 10% Days):</span><b style="color: #f85149;">${item.downside_corr.toFixed(2)}</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px;"><span>Unconditional All-Day Corr:</span><b style="color: #6ea8ff;">${item.all_day_corr.toFixed(2)}</b></div>
            <div style="margin-top: 6px; padding-top: 4px; border-top: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: space-between; gap: 16px;">
              <span>Stress Correlation Premium (Δρ):</span>
              <b style="color: ${item.stress_premium >= 0 ? "#f85149" : "#3fb950"};">
                ${item.stress_premium >= 0 ? "+" : ""}${item.stress_premium.toFixed(2)}
              </b>
            </div>
            <div style="color: #6e7681; font-size: 10px; margin-top: 4px;">
              Stress sample N = ${item.n_stress} (benchmark return &le; ${(item.stress_threshold * 100).toFixed(1)}%)
            </div>
          `;
        },
      },
      legend: {
        data: ["Unconditional All-Day Correlation", "Downside Stress Correlation (Worst 10% Benchmark Days)"],
        textStyle: { color: "#8b949e", fontSize: 11 },
        top: 0,
        right: 10,
      },
      grid: {
        top: 35,
        right: 35,
        bottom: 25,
        left: 140,
      },
      xAxis: {
        type: "value",
        min: -0.2,
        max: 1.0,
        axisLine: { lineStyle: { color: "rgba(255, 255, 255, 0.15)" } },
        axisLabel: { color: "#8b949e", fontFamily: "var(--font-mono)", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(255, 255, 255, 0.05)" } },
      },
      yAxis: {
        type: "category",
        data: categories,
        axisLine: { lineStyle: { color: "rgba(255, 255, 255, 0.15)" } },
        axisLabel: { color: "#c9d1d9", fontFamily: "var(--font-mono)", fontSize: 11 },
      },
      series: [
        {
          name: "Unconditional All-Day Correlation",
          type: "bar",
          data: allDayCorr,
          itemStyle: { color: "#388bfd" },
          barGap: "20%",
          barCategoryGap: "30%",
        },
        {
          name: "Downside Stress Correlation (Worst 10% Benchmark Days)",
          type: "bar",
          data: downsideCorr,
          itemStyle: { color: "#f85149" },
        },
      ],
    };
  }, [items, benchmarkName]);

  return (
    <div style={{ width: "100%", height: 380 }}>
      <ReactECharts
        option={option}
        style={{ width: "100%", height: "100%" }}
        opts={{ renderer: "canvas" }}
      />
    </div>
  );
};
