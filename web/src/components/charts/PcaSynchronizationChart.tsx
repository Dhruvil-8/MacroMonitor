// web/src/components/charts/PcaSynchronizationChart.tsx
"use client";

import React, { useState, useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { PcaFactorData } from "@/lib/data/schemas";

interface PcaSynchronizationChartProps {
  pcaData: PcaFactorData;
  latestShare: number;
}

export const PcaSynchronizationChart: React.FC<PcaSynchronizationChartProps> = ({
  pcaData,
  latestShare,
}) => {
  const [subTab, setSubTab] = useState<"sync" | "loadings" | "scree">("sync");

  const syncOption = useMemo(() => {
    if (!pcaData?.pc1_history || pcaData.pc1_history.length === 0) return {};
    const dates = pcaData.pc1_history.map((d) => d.date);
    const shares = pcaData.pc1_history.map((d) => d.pc1_share.toFixed(2));

    return {
      backgroundColor: "transparent",
      animationDuration: 400,
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(10, 14, 20, 0.95)",
        borderColor: "rgba(255, 255, 255, 0.12)",
        textStyle: { color: "#e6edf3", fontSize: 12, fontFamily: "var(--font-mono)" },
        formatter: (params: any[]) => {
          const pt = params[0];
          return `
            <div style="font-weight: 600; margin-bottom: 4px; color: #6ea8ff;">Global Market Synchronization</div>
            <div style="color: #8b949e;">Date: <b style="color: #e6edf3;">${pt.name}</b></div>
            <div style="color: #8b949e;">PC1 Variance Share: <b style="color: #58a6ff;">${pt.value}%</b></div>
            <div style="color: #6e7681; font-size: 10px; margin-top: 4px;">Higher values indicate loss of international diversification</div>
          `;
        },
      },
      grid: {
        top: 25,
        right: 25,
        bottom: 30,
        left: 45,
      },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: { lineStyle: { color: "rgba(255, 255, 255, 0.15)" } },
        axisLabel: {
          color: "#8b949e",
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          formatter: (v: string) => v.slice(0, 7),
        },
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
          name: "PC1 Share",
          type: "line",
          data: shares,
          lineStyle: { color: "#58a6ff", width: 2 },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(88, 166, 255, 0.25)" },
                { offset: 1, color: "rgba(88, 166, 255, 0.0)" },
              ],
            },
          },
          symbol: "none",
        },
      ],
    };
  }, [pcaData]);

  const loadingsOption = useMemo(() => {
    if (!pcaData?.loadings_pc1) return {};
    const keys = Object.keys(pcaData.loadings_pc1);
    const v1 = keys.map((k) => pcaData.loadings_pc1[k]);
    const v2 = keys.map((k) => pcaData.loadings_pc2?.[k] ?? 0);

    return {
      backgroundColor: "transparent",
      animationDuration: 400,
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: "rgba(10, 14, 20, 0.95)",
        borderColor: "rgba(255, 255, 255, 0.12)",
        textStyle: { color: "#e6edf3", fontSize: 12, fontFamily: "var(--font-mono)" },
      },
      legend: {
        data: ["PC1 (Global Equity Factor)", "PC2 (Regional Divergence)"],
        textStyle: { color: "#8b949e", fontSize: 11 },
        top: 0,
        right: 10,
      },
      grid: {
        top: 35,
        right: 25,
        bottom: 30,
        left: 55,
      },
      xAxis: {
        type: "category",
        data: keys.map((k) => k.toUpperCase()),
        axisLine: { lineStyle: { color: "rgba(255, 255, 255, 0.15)" } },
        axisLabel: { color: "#8b949e", fontFamily: "var(--font-mono)", fontSize: 10 },
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
          name: "PC1 (Global Equity Factor)",
          type: "bar",
          data: v1,
          itemStyle: { color: "#3fb6a8" },
        },
        {
          name: "PC2 (Regional Divergence)",
          type: "bar",
          data: v2,
          itemStyle: { color: "#a371f7" },
        },
      ],
    };
  }, [pcaData]);

  const screeOption = useMemo(() => {
    if (!pcaData?.scree || pcaData.scree.length === 0) return {};
    const labels = pcaData.scree.map((_, i) => `PC${i + 1}`);
    const vals = pcaData.scree;
    let cum = 0;
    const cumVals = vals.map((v) => {
      cum += v;
      return cum.toFixed(1);
    });

    return {
      backgroundColor: "transparent",
      animationDuration: 400,
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(10, 14, 20, 0.95)",
        borderColor: "rgba(255, 255, 255, 0.12)",
        textStyle: { color: "#e6edf3", fontSize: 12, fontFamily: "var(--font-mono)" },
      },
      legend: {
        data: ["Individual Explained Variance", "Cumulative Variance"],
        textStyle: { color: "#8b949e", fontSize: 11 },
        top: 0,
        right: 10,
      },
      grid: {
        top: 35,
        right: 45,
        bottom: 30,
        left: 45,
      },
      xAxis: {
        type: "category",
        data: labels,
        axisLine: { lineStyle: { color: "rgba(255, 255, 255, 0.15)" } },
        axisLabel: { color: "#8b949e", fontFamily: "var(--font-mono)", fontSize: 11 },
      },
      yAxis: [
        {
          type: "value",
          name: "Share",
          axisLabel: { formatter: "{value}%", color: "#8b949e", fontSize: 10 },
          splitLine: { lineStyle: { color: "rgba(255, 255, 255, 0.05)" } },
        },
        {
          type: "value",
          name: "Cumul",
          max: 100,
          axisLabel: { formatter: "{value}%", color: "#8b949e", fontSize: 10 },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: "Individual Explained Variance",
          type: "bar",
          data: vals,
          itemStyle: { color: "#58a6ff" },
          barWidth: "40%",
        },
        {
          name: "Cumulative Variance",
          type: "line",
          yAxisIndex: 1,
          data: cumVals,
          lineStyle: { color: "#f0883e", width: 2 },
          itemStyle: { color: "#f0883e" },
        },
      ],
    };
  }, [pcaData]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", gap: "6px" }}>
          <button
            onClick={() => setSubTab("sync")}
            style={{
              padding: "4px 10px",
              fontSize: "12px",
              fontFamily: "var(--font-mono)",
              borderRadius: "4px",
              border: "1px solid",
              borderColor: subTab === "sync" ? "#58a6ff" : "rgba(255,255,255,0.1)",
              backgroundColor: subTab === "sync" ? "rgba(88,166,255,0.15)" : "transparent",
              color: subTab === "sync" ? "#58a6ff" : "#8b949e",
              cursor: "pointer",
            }}
          >
            PC1 Synchronization History
          </button>
          <button
            onClick={() => setSubTab("loadings")}
            style={{
              padding: "4px 10px",
              fontSize: "12px",
              fontFamily: "var(--font-mono)",
              borderRadius: "4px",
              border: "1px solid",
              borderColor: subTab === "loadings" ? "#58a6ff" : "rgba(255,255,255,0.1)",
              backgroundColor: subTab === "loadings" ? "rgba(88,166,255,0.15)" : "transparent",
              color: subTab === "loadings" ? "#58a6ff" : "#8b949e",
              cursor: "pointer",
            }}
          >
            Factor Loadings (Sign-Aligned)
          </button>
          <button
            onClick={() => setSubTab("scree")}
            style={{
              padding: "4px 10px",
              fontSize: "12px",
              fontFamily: "var(--font-mono)",
              borderRadius: "4px",
              border: "1px solid",
              borderColor: subTab === "scree" ? "#58a6ff" : "rgba(255,255,255,0.1)",
              backgroundColor: subTab === "scree" ? "rgba(88,166,255,0.15)" : "transparent",
              color: subTab === "scree" ? "#58a6ff" : "#8b949e",
              cursor: "pointer",
            }}
          >
            Scree Plot
          </button>
        </div>

        <div style={{ fontSize: "12px", color: "#8b949e", fontFamily: "var(--font-mono)" }}>
          Latest Global PC1 Share: <b style={{ color: "#58a6ff" }}>{latestShare.toFixed(1)}%</b>
        </div>
      </div>

      <div style={{ width: "100%", height: 300 }}>
        {subTab === "sync" && (
          <ReactECharts option={syncOption} style={{ width: "100%", height: "100%" }} opts={{ renderer: "canvas" }} />
        )}
        {subTab === "loadings" && (
          <ReactECharts option={loadingsOption} style={{ width: "100%", height: "100%" }} opts={{ renderer: "canvas" }} />
        )}
        {subTab === "scree" && (
          <ReactECharts option={screeOption} style={{ width: "100%", height: "100%" }} opts={{ renderer: "canvas" }} />
        )}
      </div>
    </div>
  );
};
