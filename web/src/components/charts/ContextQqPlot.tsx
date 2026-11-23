// web/src/components/charts/ContextQqPlot.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";

interface ContextQqPlotProps {
  theoretical: number[];
  empirical: number[];
  assetName: string;
}

export const ContextQqPlot: React.FC<ContextQqPlotProps> = ({
  theoretical,
  empirical,
  assetName,
}) => {
  const option = useMemo(() => {
    if (!theoretical || theoretical.length === 0 || !empirical || empirical.length === 0) return {};

    const points = theoretical.map((t, i) => [t, empirical[i]]);
    const minVal = Math.floor(Math.min(...theoretical, ...empirical));
    const maxVal = Math.ceil(Math.max(...theoretical, ...empirical));

    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      tooltip: {
        trigger: "item",
        backgroundColor: "rgba(10, 14, 20, 0.95)",
        borderColor: "rgba(255, 255, 255, 0.12)",
        textStyle: { color: "#e6edf3", fontSize: 11, fontFamily: "var(--font-mono)" },
        formatter: (p: any) => {
          const [t, e] = p.value;
          return `
            <div style="font-weight: 600; color: #58a6ff; margin-bottom: 4px;">${assetName} Quantile Comparison</div>
            <div style="display: flex; justify-content: space-between; gap: 16px;"><span>Theoretical (Normal):</span><b>${Number(t).toFixed(2)}σ</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px;"><span>Empirical (Observed):</span><b style="color: ${e < t ? '#f85149' : '#3fb950'};">${Number(e).toFixed(2)}σ</b></div>
            <div style="margin-top: 4px; font-size: 10px; color: #8b949e;">Departure: ${(e - t).toFixed(2)}σ</div>
          `;
        },
      },
      legend: {
        data: ["Empirical Quantiles", "Gaussian 45° Reference (Normal Ideal)"],
        textStyle: { color: "#8b949e", fontSize: 11, fontFamily: "monospace" },
        top: 0,
        right: 10,
        icon: "roundRect",
      },
      grid: {
        top: 35,
        right: 25,
        bottom: 30,
        left: 35,
        containLabel: true,
      },
      xAxis: {
        type: "value",
        name: "Theoretical Normal (σ)",
        nameLocation: "middle",
        nameGap: 24,
        nameTextStyle: { color: "#8b949e", fontSize: 10 },
        axisLine: { lineStyle: { color: "rgba(255, 255, 255, 0.15)" } },
        axisLabel: { color: "#8b949e", fontFamily: "var(--font-mono)", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(255, 255, 255, 0.05)" } },
      },
      yAxis: {
        type: "value",
        name: "Empirical Observed (σ)",
        nameTextStyle: { color: "#8b949e", fontSize: 10 },
        axisLine: { lineStyle: { color: "rgba(255, 255, 255, 0.15)" } },
        axisLabel: { color: "#8b949e", fontFamily: "var(--font-mono)", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(255, 255, 255, 0.05)" } },
      },
      series: [
        {
          name: "Gaussian 45° Reference (Normal Ideal)",
          type: "line",
          data: [
            [minVal, minVal],
            [maxVal, maxVal],
          ],
          lineStyle: { color: "#6e7681", type: "dashed", width: 1.5 },
          symbol: "none",
        },
        {
          name: "Empirical Quantiles",
          type: "scatter",
          data: points,
          itemStyle: {
            color: "#58a6ff",
            borderColor: "#0a0e14",
            borderWidth: 1.5,
          },
          symbolSize: 6,
        },
      ],
    };
  }, [theoretical, empirical, assetName]);

  return (
    <div className="space-y-2">
      <div className="w-full h-[280px]">
        <ReactECharts
          option={option}
          style={{ width: "100%", height: "100%" }}
          opts={{ renderer: "canvas" }}
          notMerge={true}
          lazyUpdate={true}
        />
      </div>
      <div className="flex justify-between items-center text-[11px] font-mono text-[var(--tx-2)] px-1 pt-1 border-t border-[var(--line)]">
        <span>Standardized Quantiles (50 points)</span>
        <span>Reference: Gaussian 45° Line</span>
      </div>
    </div>
  );
};
