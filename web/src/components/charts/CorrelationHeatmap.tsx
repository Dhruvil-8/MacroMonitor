// web/src/components/charts/CorrelationHeatmap.tsx
"use client";

import React, { useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

interface CorrelationHeatmapProps {
  benchmarkAssets: string[];
  heatmapNow: Record<string, Record<string, number>>;
  heatmap3m: Record<string, Record<string, number>>;
  heatmap1y: Record<string, Record<string, number>>;
}

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({
  benchmarkAssets,
  heatmapNow,
  heatmap3m,
  heatmap1y,
}) => {
  const [selectedSnapshot, setSelectedSnapshot] = useState<"now" | "3m" | "1y">("now");

  const currentMatrix = useMemo(() => {
    switch (selectedSnapshot) {
      case "3m":
        return heatmap3m;
      case "1y":
        return heatmap1y;
      case "now":
      default:
        return heatmapNow;
    }
  }, [selectedSnapshot, heatmapNow, heatmap3m, heatmap1y]);

  const heatmapData = useMemo(() => {
    const data: Array<[number, number, number]> = [];
    benchmarkAssets.forEach((rowAsset, rowIdx) => {
      benchmarkAssets.forEach((colAsset, colIdx) => {
        const val = currentMatrix[rowAsset]?.[colAsset] ?? (rowAsset === colAsset ? 1.0 : 0.0);
        data.push([colIdx, rowIdx, Number(val.toFixed(2))]);
      });
    });
    return data;
  }, [benchmarkAssets, currentMatrix]);

  const option = useMemo(() => {
    const labels = benchmarkAssets.map((s) => s.toUpperCase());
    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: {
        top: 25,
        right: 45,
        bottom: 50,
        left: 55,
      },
      tooltip: {
        position: "top",
        backgroundColor: "#0c1017",
        borderColor: "#1d2633",
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          const [colIdx, rowIdx, val] = params.data;
          const a = labels[rowIdx];
          const b = labels[colIdx];
          return `<strong>${a} / ${b}</strong>: ${val > 0 ? "+" : ""}${val.toFixed(2)}`;
        },
      },
      xAxis: {
        type: "category",
        data: labels,
        splitArea: { show: true, areaStyle: { color: ["rgba(18,24,35,0.3)", "transparent"] } },
        axisLabel: { color: "#a9b4c4", fontSize: 10, fontFamily: "monospace" },
        axisLine: { lineStyle: { color: "#2a3647" } },
      },
      yAxis: {
        type: "category",
        data: labels,
        splitArea: { show: true, areaStyle: { color: ["rgba(18,24,35,0.3)", "transparent"] } },
        axisLabel: { color: "#a9b4c4", fontSize: 10, fontFamily: "monospace" },
        axisLine: { lineStyle: { color: "#2a3647" } },
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: false,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        itemWidth: 10,
        itemHeight: 120,
        textStyle: { color: "#6f7c90", fontSize: 10, fontFamily: "monospace" },
        // CVD-safe diverging scale: -1 (coral/neg), 0 (neutral dark), +1 (teal/pos)
        inRange: {
          color: ["#e8765a", "#2a3647", "#3fb6a8"],
        },
      },
      series: [
        {
          name: "Correlation",
          type: "heatmap",
          data: heatmapData,
          label: {
            show: true,
            color: "#e8edf5",
            fontSize: 9,
            fontFamily: "monospace",
          },
        },
      ],
    };
  }, [benchmarkAssets, heatmapData]);

  return (
    <div className="space-y-3">
      {/* Snapshot Selector */}
      <div className="flex items-center justify-between pb-1 border-b border-[var(--line)]">
        <span className="text-xs font-mono text-[var(--tx-2)]">Pearson Return Matrix</span>
        <div className="flex rounded border border-[var(--line)] overflow-hidden text-xs">
          {(["now", "3m", "1y"] as const).map((period) => (
            <button
              key={period}
              onClick={() => setSelectedSnapshot(period)}
              className={`px-2.5 py-1 font-mono text-[11px] uppercase transition-colors ${
                selectedSnapshot === period
                  ? "bg-[var(--accent)] text-white font-medium"
                  : "bg-[var(--bg-0)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              }`}
            >
              {period === "now" ? "Current" : period === "3m" ? "3M Ago" : "1Y Ago"}
            </button>
          ))}
        </div>
      </div>

      {/* Heatmap Chart */}
      <div className="w-full h-[320px]">
        <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />
      </div>
    </div>
  );
};
