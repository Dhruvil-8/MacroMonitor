// web/src/components/charts/FiiDiiFlowChart.tsx
"use client";

import React, { useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

interface FiiDiiFlowChartProps {
  dates: string[];
  fiiDaily: number[];
  diiDaily: number[];
  fiiCum20d: number[];
  diiCum20d: number[];
}

export const FiiDiiFlowChart: React.FC<FiiDiiFlowChartProps> = ({
  dates,
  fiiDaily,
  diiDaily,
  fiiCum20d,
  diiCum20d,
}) => {
  const [viewMode, setViewMode] = useState<"daily" | "cumulative">("daily");

  const option = useMemo(() => {
    if (viewMode === "cumulative") {
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
            let html = `<div style="font-weight:600;margin-bottom:4px;border-bottom:1px solid #232f42;padding-bottom:2px;">${params[0].axisValue}</div>`;
            params.forEach((p) => {
              const sign = p.value >= 0 ? "+" : "";
              html += `<div style="display:flex;justify-content:space-between;gap:12px;margin-top:2px;">
                <span style="color:${p.color};">${p.seriesName}:</span>
                <span style="font-weight:600;">${sign}₹${p.value.toLocaleString()} Cr</span>
              </div>`;
            });
            return html;
          },
        },
        legend: {
          bottom: 0,
          textStyle: { color: "#a9b4c4", fontSize: 11 },
          itemWidth: 14,
          itemHeight: 8,
        },
        grid: {
          top: 20,
          left: 55,
          right: 25,
          bottom: 35,
        },
        xAxis: {
          type: "category",
          data: dates,
          axisLine: { lineStyle: { color: "#232f42" } },
          axisLabel: { color: "#6f7c90", fontSize: 10 },
          splitLine: { show: false },
        },
        yAxis: {
          type: "value",
          name: "₹ Crores (20D Sum)",
          nameTextStyle: { color: "#6f7c90", fontSize: 10, align: "right" },
          axisLabel: {
            color: "#6f7c90",
            fontSize: 10,
            formatter: (v: number) => `₹${(v / 1000).toFixed(0)}k Cr`,
          },
          splitLine: {
            show: true,
            lineStyle: { color: "#232f42", opacity: 0.5, type: "dashed" },
          },
        },
        series: [
          {
            name: "FII 20D Cumulative",
            type: "line",
            data: fiiCum20d,
            lineStyle: { width: 2, color: "#38bdf8" },
            showSymbol: false,
          },
          {
            name: "DII 20D Cumulative",
            type: "line",
            data: diiCum20d,
            lineStyle: { width: 2, color: "#a855f7" },
            showSymbol: false,
          },
        ],
      };
    }

    // Daily Bar Mode
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
          let html = `<div style="font-weight:600;margin-bottom:4px;border-bottom:1px solid #232f42;padding-bottom:2px;">${params[0].axisValue}</div>`;
          params.forEach((p) => {
            const sign = p.value >= 0 ? "+" : "";
            html += `<div style="display:flex;justify-content:space-between;gap:12px;margin-top:2px;">
              <span style="color:${p.color};">${p.seriesName}:</span>
              <span style="font-weight:600;">${sign}₹${p.value.toLocaleString()} Cr</span>
            </div>`;
          });
          return html;
        },
      },
      legend: {
        bottom: 0,
        textStyle: { color: "#a9b4c4", fontSize: 11 },
        itemWidth: 12,
        itemHeight: 8,
      },
      grid: {
        top: 20,
        left: 55,
        right: 25,
        bottom: 35,
      },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: { lineStyle: { color: "#232f42" } },
        axisLabel: { color: "#6f7c90", fontSize: 10 },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        name: "₹ Crores",
        nameTextStyle: { color: "#6f7c90", fontSize: 10, align: "right" },
        axisLabel: {
          color: "#6f7c90",
          fontSize: 10,
          formatter: (v: number) => `₹${v.toLocaleString()}`,
        },
        splitLine: {
          show: true,
          lineStyle: { color: "#232f42", opacity: 0.5, type: "dashed" },
        },
      },
      series: [
        {
          name: "FII Net Cash",
          type: "bar",
          data: fiiDaily,
          itemStyle: {
            color: (params: any) => (params.value >= 0 ? "#22c55e" : "#ef4444"),
          },
        },
        {
          name: "DII Net Cash",
          type: "bar",
          data: diiDaily,
          itemStyle: {
            color: (params: any) => (params.value >= 0 ? "#3b82f6" : "#f59e0b"),
          },
        },
      ],
    };
  }, [dates, fiiDaily, diiDaily, fiiCum20d, diiCum20d, viewMode]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5 bg-[#131b26] p-0.5 rounded border border-[#232f42]">
          <button
            onClick={() => setViewMode("daily")}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
              viewMode === "daily" ? "bg-[#232f42] text-[#e8edf5]" : "text-[#6f7c90]"
            }`}
          >
            Daily Net Flows
          </button>
          <button
            onClick={() => setViewMode("cumulative")}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
              viewMode === "cumulative" ? "bg-[#232f42] text-[#e8edf5]" : "text-[#6f7c90]"
            }`}
          >
            20-Day Cumulative
          </button>
        </div>
        <div className="text-[11px] font-mono text-[#6f7c90]">Unit: INR Crores (₹ Cr)</div>
      </div>

      <div className="h-[280px] w-full">
        <ReactECharts option={option} style={{ height: "100%", width: "100%" }} notMerge={true} />
      </div>
    </div>
  );
};
