// web/src/components/charts/ZScoreBandChart.tsx
// C21: Standardized z-score band chart with outlier thresholds (Section 5.1)
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";

interface ZScoreBandChartProps {
  seriesName?: string;
}

export const ZScoreBandChart: React.FC<ZScoreBandChartProps> = ({
  seriesName = "S&P 500 Daily Innovation Z-Score",
}) => {
  // Generate sample trailing 60-day expanding z-score series
  const { dates, zScores } = useMemo(() => {
    const ds: string[] = [];
    const zs: number[] = [];
    const baseDate = new Date("2026-08-25");
    let currentZ = 0.2;
    for (let i = 0; i < 60; i++) {
      const d = new Date(baseDate);
      d.setDate(d.getDate() + i);
      ds.push(d.toISOString().slice(5, 10));
      // Random walk with mean reversion
      const noise = (Math.random() - 0.5) * 1.2;
      currentZ = currentZ * 0.7 + noise;
      if (i === 58) currentZ = -2.85; // extreme outlier move
      if (i === 59) currentZ = -3.20;
      zs.push(parseFloat(currentZ.toFixed(2)));
    }
    return { dates: ds, zScores: zs };
  }, []);

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0c1017",
      borderColor: "#1d2633",
      textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
      formatter: (params: any) => {
        const item = params[0];
        const val = item.value;
        const alertTag = Math.abs(val) >= 2.0 ? `<span style="color:#e8765a;font-weight:bold"> [Outlier |z| >= 2.0]</span>` : "";
        return `
          <div style="font-weight:600;margin-bottom:2px">${item.axisValue}</div>
          <div style="color:#6f7c90">Standardized Z: <span style="color:${val >= 0 ? '#3fb6a8' : '#e8765a'}">${val >= 0 ? '+' : ''}${val}&sigma;</span>${alertTag}</div>
        `;
      },
    },
    grid: { left: 45, right: 30, top: 30, bottom: 35 },
    xAxis: {
      type: "category",
      data: dates,
      axisLine: { lineStyle: { color: "#1d2633" } },
      axisLabel: { color: "#6f7c90", fontFamily: "monospace", fontSize: 10 },
    },
    yAxis: {
      type: "value",
      min: -4.0,
      max: 4.0,
      axisLine: { lineStyle: { color: "#1d2633" } },
      splitLine: { lineStyle: { color: "#121823" } },
      axisLabel: {
        color: "#6f7c90",
        fontFamily: "monospace",
        fontSize: 10,
        formatter: (val: number) => `${val > 0 ? "+" : ""}${val}&sigma;`,
      },
    },
    series: [
      {
        name: seriesName,
        type: "line",
        data: zScores,
        smooth: false,
        symbol: "circle",
        symbolSize: 4,
        lineStyle: { width: 1.8, color: "#6ea8ff" },
        itemStyle: {
          color: (params: any) => (Math.abs(params.value) >= 2.0 ? "#e8765a" : "#6ea8ff"),
        },
        markLine: {
          symbol: "none",
          data: [
            { yAxis: 3.0, lineStyle: { color: "#e8765a", type: "dashed", width: 1 } },
            { yAxis: 2.0, lineStyle: { color: "#e6b94d", type: "dashed", width: 1 } },
            { yAxis: 0.0, lineStyle: { color: "#2f6fb4", type: "solid", width: 1 } },
            { yAxis: -2.0, lineStyle: { color: "#e6b94d", type: "dashed", width: 1 } },
            { yAxis: -3.0, lineStyle: { color: "#e8765a", type: "dashed", width: 1 } },
          ],
        },
        markArea: {
          itemStyle: { color: "rgba(232, 118, 90, 0.06)" },
          data: [
            [{ yAxis: 2.0 }, { yAxis: 4.0 }],
            [{ yAxis: -4.0 }, { yAxis: -2.0 }],
          ],
        },
      },
    ],
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs font-mono text-[var(--tx-2)]">
        <span>Standardized Innovation Z-Score (60D Lookback)</span>
        <span className="text-amber-400">&plusmn;2&sigma; / &plusmn;3&sigma; Threshold Bands</span>
      </div>
      <ReactECharts option={option} style={{ height: "300px", width: "100%" }} />
    </div>
  );
};
