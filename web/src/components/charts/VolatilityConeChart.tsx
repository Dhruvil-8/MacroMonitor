// web/src/components/charts/VolatilityConeChart.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { VolatilityConePoint } from "@/lib/data/schemas";

interface VolatilityConeChartProps {
  coneData: VolatilityConePoint[];
  assetName: string;
}

export const VolatilityConeChart: React.FC<VolatilityConeChartProps> = ({
  coneData,
  assetName,
}) => {
  const option = useMemo(() => {
    if (!coneData || coneData.length === 0) return {};

    const categories = coneData.map((d) => `${d.window}D`);
    const p10Values = coneData.map((d) => (d.p10 * 100).toFixed(1));
    const p25Values = coneData.map((d) => (d.p25 * 100).toFixed(1));
    const p50Values = coneData.map((d) => (d.p50 * 100).toFixed(1));
    const p75Values = coneData.map((d) => (d.p75 * 100).toFixed(1));
    const p90Values = coneData.map((d) => (d.p90 * 100).toFixed(1));
    const currentValues = coneData.map((d) => (d.current * 100).toFixed(1));

    // Confidence bands:
    // Base 1: p10
    // Band 1: p25 - p10
    // Band 2: p75 - p25 (IQR)
    // Band 3: p90 - p75
    const baseP10 = coneData.map((d) => d.p10 * 100);
    const bandP10ToP25 = coneData.map((d) => (d.p25 - d.p10) * 100);
    const bandP25ToP75 = coneData.map((d) => (d.p75 - d.p25) * 100);
    const bandP75ToP90 = coneData.map((d) => (d.p90 - d.p75) * 100);

    return {
      backgroundColor: "transparent",
      animationDuration: 400,
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(10, 14, 20, 0.95)",
        borderColor: "rgba(255, 255, 255, 0.12)",
        textStyle: { color: "#e6edf3", fontSize: 12, fontFamily: "var(--font-mono)" },
        formatter: (params: any[]) => {
          const idx = params[0]?.dataIndex ?? 0;
          const pt = coneData[idx];
          if (!pt) return "";
          return `
            <div style="font-weight: 600; margin-bottom: 6px; color: #58a6ff;">${assetName} — ${pt.window}-Day Window</div>
            <div style="display: flex; justify-content: space-between; gap: 16px;"><span>Current Vol:</span><b style="color: #f0883e;">${(pt.current * 100).toFixed(1)}% (${pt.percentile.toFixed(0)}th pct)</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px; color: #8b949e;"><span>Historical Max:</span><b>${(pt.max * 100).toFixed(1)}%</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px; color: #8b949e;"><span>90th Percentile:</span><b>${(pt.p90 * 100).toFixed(1)}%</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px; color: #8b949e;"><span>75th Percentile:</span><b>${(pt.p75 * 100).toFixed(1)}%</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px; color: #8b949e;"><span>Median (p50):</span><b>${(pt.p50 * 100).toFixed(1)}%</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px; color: #8b949e;"><span>25th Percentile:</span><b>${(pt.p25 * 100).toFixed(1)}%</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px; color: #8b949e;"><span>10th Percentile:</span><b>${(pt.p10 * 100).toFixed(1)}%</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px; color: #8b949e;"><span>Historical Min:</span><b>${(pt.min * 100).toFixed(1)}%</b></div>
          `;
        },
      },
      legend: {
        data: ["Current Realized", "Median (50th)", "IQR (25th-75th)", "80% Range (10th-90th)"],
        textStyle: { color: "#8b949e", fontSize: 11 },
        top: 0,
        right: 10,
      },
      grid: {
        top: 35,
        right: 25,
        bottom: 30,
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
        // Dummy stack for Base (p10)
        {
          name: "BaseP10",
          type: "line",
          stack: "cone_bands",
          data: baseP10,
          lineStyle: { opacity: 0 },
          itemStyle: { opacity: 0 },
          symbol: "none",
        },
        // Outer band (10th to 25th)
        {
          name: "80% Range (10th-90th)",
          type: "line",
          stack: "cone_bands",
          data: bandP10ToP25,
          lineStyle: { opacity: 0 },
          itemStyle: { opacity: 0 },
          areaStyle: { color: "rgba(56, 139, 253, 0.10)" },
          symbol: "none",
        },
        // Inner band IQR (25th to 75th)
        {
          name: "IQR (25th-75th)",
          type: "line",
          stack: "cone_bands",
          data: bandP25ToP75,
          lineStyle: { opacity: 0 },
          itemStyle: { opacity: 0 },
          areaStyle: { color: "rgba(56, 139, 253, 0.25)" },
          symbol: "none",
        },
        // Upper outer band (75th to 90th)
        {
          name: "Upper80",
          type: "line",
          stack: "cone_bands",
          data: bandP75ToP90,
          lineStyle: { opacity: 0 },
          itemStyle: { opacity: 0 },
          areaStyle: { color: "rgba(56, 139, 253, 0.10)" },
          symbol: "none",
        },
        // Median Line
        {
          name: "Median (50th)",
          type: "line",
          data: p50Values,
          lineStyle: { color: "#58a6ff", type: "dashed", width: 1.5 },
          itemStyle: { color: "#58a6ff" },
          symbol: "circle",
          symbolSize: 4,
        },
        // Current Realized Vol
        {
          name: "Current Realized",
          type: "line",
          data: currentValues,
          lineStyle: { color: "#f0883e", width: 2.5 },
          itemStyle: {
            color: "#f0883e",
            borderColor: "#0a0e14",
            borderWidth: 2,
          },
          symbol: "circle",
          symbolSize: 8,
          label: {
            show: true,
            position: "top",
            formatter: (p: any) => `${p.value}%`,
            color: "#f0883e",
            fontSize: 10,
            fontFamily: "var(--font-mono)",
            fontWeight: "bold",
          },
        },
      ],
    };
  }, [coneData, assetName]);

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
