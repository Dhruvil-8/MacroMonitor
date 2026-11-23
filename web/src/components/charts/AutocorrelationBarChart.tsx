// web/src/components/charts/AutocorrelationBarChart.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { AutocorrelationRiskData } from "@/lib/data/schemas";

interface AutocorrelationBarChartProps {
  acfData: AutocorrelationRiskData;
  assetName: string;
}

export const AutocorrelationBarChart: React.FC<AutocorrelationBarChartProps> = ({
  acfData,
  assetName,
}) => {
  const option = useMemo(() => {
    if (!acfData || !acfData.lags || acfData.lags.length === 0) return {};

    const categories = acfData.lags.map((k) => `Lag ${k}`);
    const bandVal = acfData.band;
    const bandPos = acfData.lags.map(() => bandVal.toFixed(3));
    const bandNeg = acfData.lags.map(() => (-bandVal).toFixed(3));

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
          const lag = acfData.lags[idx];
          const acfR = acfData.acf_returns[idx];
          const acfSq = acfData.acf_squared[idx];
          return `
            <div style="font-weight: 600; margin-bottom: 6px; color: #58a6ff;">${assetName} — Lag ${lag}</div>
            <div style="display: flex; justify-content: space-between; gap: 16px;"><span>Squared Returns r² (Vol Clustering):</span><b style="color: #f0883e;">${acfSq?.toFixed(3)}</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px;"><span>Raw Returns r (Linear Memory):</span><b style="color: #58a6ff;">${acfR?.toFixed(3)}</b></div>
            <div style="display: flex; justify-content: space-between; gap: 16px; color: #8b949e;"><span>95% N_eff Band (±1.96/√N):</span><b>±${bandVal.toFixed(3)}</b></div>
            <div style="color: #6e7681; font-size: 10px; margin-top: 4px;">Effective N = ${acfData.n_eff.toFixed(0)}</div>
          `;
        },
      },
      legend: {
        data: ["Squared Returns r² (Vol Clustering)", "Raw Returns r", "95% Confidence Band"],
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
        axisLabel: { color: "#8b949e", fontFamily: "var(--font-mono)", fontSize: 10, interval: 1 },
      },
      yAxis: {
        type: "value",
        axisLabel: {
          color: "#8b949e",
          fontFamily: "var(--font-mono)",
          fontSize: 10,
        },
        splitLine: { lineStyle: { color: "rgba(255, 255, 255, 0.05)" } },
      },
      series: [
        {
          name: "Squared Returns r² (Vol Clustering)",
          type: "bar",
          data: acfData.acf_squared,
          itemStyle: { color: "#f0883e" },
          barWidth: "30%",
        },
        {
          name: "Raw Returns r",
          type: "bar",
          data: acfData.acf_returns,
          itemStyle: { color: "#58a6ff" },
          barWidth: "30%",
        },
        {
          name: "95% Confidence Band",
          type: "line",
          data: bandPos,
          lineStyle: { color: "#6e7681", type: "dashed", width: 1 },
          symbol: "none",
        },
        {
          name: "LowerBand",
          type: "line",
          data: bandNeg,
          lineStyle: { color: "#6e7681", type: "dashed", width: 1 },
          symbol: "none",
        },
      ],
    };
  }, [acfData, assetName]);

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
