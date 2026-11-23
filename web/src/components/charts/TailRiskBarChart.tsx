// web/src/components/charts/TailRiskBarChart.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { TailRiskMetric } from "@/lib/data/schemas";

interface TailRiskBarChartProps {
  tailMetrics: TailRiskMetric[];
  assetName: string;
}

export const TailRiskBarChart: React.FC<TailRiskBarChartProps> = ({
  tailMetrics,
  assetName,
}) => {
  const option = useMemo(() => {
    if (!tailMetrics || tailMetrics.length === 0) return {};

    const categories = tailMetrics.map((m) => `${(m.confidence * 100).toFixed(0)}% Confidence`);
    const histVar = tailMetrics.map((m) => (m.hist_var * 100).toFixed(2));
    const histEs = tailMetrics.map((m) => (m.hist_es * 100).toFixed(2));
    const gaussVar = tailMetrics.map((m) => (m.gauss_var * 100).toFixed(2));
    const gaussEs = tailMetrics.map((m) => (m.gauss_es * 100).toFixed(2));

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
          const m = tailMetrics[idx];
          if (!m) return "";
          return `
            <div style="font-weight: 600; margin-bottom: 6px; color: #58a6ff;">${assetName} — ${(m.confidence * 100).toFixed(0)}% Loss Risk</div>
            <div style="display: flex; justify-content: space-between; gap: 16px;"><span>Historical ES:</span><b style="color: #f85149;">${(m.hist_es * 100).toFixed(2)}%</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px;"><span>Historical VaR:</span><b style="color: #e8765a;">${(m.hist_var * 100).toFixed(2)}%</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px; color: #8b949e;"><span>Gaussian ES:</span><b>${(m.gauss_es * 100).toFixed(2)}%</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px; color: #8b949e;"><span>Gaussian VaR:</span><b>${(m.gauss_var * 100).toFixed(2)}%</b></div>
            <div style="margin-top: 6px; padding-top: 4px; border-top: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: space-between; gap: 16px;">
              <span>Tail Ratio (Hist / Gauss ES):</span><b style="color: #d29922;">${m.tail_ratio.toFixed(2)}x</b>
            </div>
          `;
        },
      },
      legend: {
        data: ["Historical Expected Shortfall", "Gaussian Expected Shortfall", "Historical VaR", "Gaussian VaR"],
        textStyle: { color: "#8b949e", fontSize: 11 },
        top: 0,
        right: 10,
      },
      grid: {
        top: 35,
        right: 25,
        bottom: 25,
        left: 45,
      },
      xAxis: {
        type: "category",
        data: categories,
        axisLine: { lineStyle: { color: "rgba(255, 255, 255, 0.15)" } },
        axisLabel: { color: "#8b949e", fontFamily: "var(--font-mono)", fontSize: 11 },
      },
      yAxis: {
        type: "value",
        axisLabel: {
          formatter: "{value}%",
          color: "#8b949e",
          fontFamily: "var(--font-mono)",
          fontSize: 10,
        },
        splitLine: { lineStyle: { color: "rgba(255, 255, 255, 0.05)" } },
      },
      series: [
        {
          name: "Historical Expected Shortfall",
          type: "bar",
          data: histEs,
          itemStyle: { color: "#f85149" },
          barGap: "20%",
          barCategoryGap: "35%",
        },
        {
          name: "Gaussian Expected Shortfall",
          type: "bar",
          data: gaussEs,
          itemStyle: { color: "rgba(248, 81, 73, 0.35)", borderColor: "#f85149", borderWidth: 1 },
        },
        {
          name: "Historical VaR",
          type: "bar",
          data: histVar,
          itemStyle: { color: "#d29922" },
        },
        {
          name: "Gaussian VaR",
          type: "bar",
          data: gaussVar,
          itemStyle: { color: "rgba(210, 153, 34, 0.35)", borderColor: "#d29922", borderWidth: 1 },
        },
      ],
    };
  }, [tailMetrics, assetName]);

  return (
    <div style={{ width: "100%", height: 320 }}>
      <ReactECharts
        option={option}
        style={{ width: "100%", height: "100%" }}
        opts={{ renderer: "canvas" }}
      />
    </div>
  );
};
