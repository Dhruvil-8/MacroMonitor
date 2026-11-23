// web/src/components/charts/StressHistoryChart.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { CrisisEpisode } from "@/lib/data/schemas";

interface StressHistoryChartProps {
  dates: string[];
  history: number[];
  episodes: CrisisEpisode[];
}

export const StressHistoryChart: React.FC<StressHistoryChartProps> = ({
  dates,
  history,
  episodes,
}) => {
  const option = useMemo(() => {
    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0d131d",
        borderColor: "#232f42",
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "var(--font-mono)" },
        formatter: (params: any[]) => {
          if (!params || params.length === 0) return "";
          const p = params[0];
          return `<div>
            <div style="color:#6f7c90;font-size:10px;">${p.axisValue}</div>
            <div style="font-weight:600;margin-top:2px;">Stress Composite: <span style="color:#38bdf8;">${p.value}/100</span></div>
          </div>`;
        },
      },
      grid: {
        top: 20,
        left: 45,
        right: 20,
        bottom: 30,
      },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: { lineStyle: { color: "#232f42" } },
        axisLabel: {
          color: "#6f7c90",
          fontSize: 10,
          formatter: (val: string) => {
            const parts = val.split("-");
            return parts.length >= 2 ? `${parts[0].slice(2)}/${parts[1]}` : val;
          },
        },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        min: 0,
        max: 100,
        axisLabel: {
          color: "#6f7c90",
          fontSize: 10,
          formatter: (v: number) => `${v}`,
        },
        splitLine: {
          show: true,
          lineStyle: { color: "#232f42", opacity: 0.4 },
        },
      },
      series: [
        {
          name: "Financial Stress",
          type: "line",
          data: history,
          lineStyle: { width: 1.8, color: "#38bdf8" },
          areaStyle: {
            color: "#38bdf8",
            opacity: 0.08,
          },
          showSymbol: false,
          markLine: {
            silent: true,
            symbol: "none",
            data: [
              {
                yAxis: 40,
                lineStyle: { color: "#22c55e", type: "dashed", opacity: 0.5 },
                label: { show: true, position: "insideEndTop", formatter: "Calm", color: "#22c55e", fontSize: 9 },
              },
              {
                yAxis: 70,
                lineStyle: { color: "#f59e0b", type: "dashed", opacity: 0.5 },
                label: { show: true, position: "insideEndTop", formatter: "Elevated", color: "#f59e0b", fontSize: 9 },
              },
              {
                yAxis: 85,
                lineStyle: { color: "#ef4444", type: "dashed", opacity: 0.5 },
                label: { show: true, position: "insideEndTop", formatter: "Crisis", color: "#ef4444", fontSize: 9 },
              },
            ],
          },
        },
      ],
    };
  }, [dates, history, episodes]);

  return (
    <div className="h-[220px] w-full">
      <ReactECharts option={option} style={{ height: "100%", width: "100%" }} notMerge={true} />
    </div>
  );
};
