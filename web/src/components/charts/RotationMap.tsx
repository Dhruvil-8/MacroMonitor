// web/src/components/charts/RotationMap.tsx
"use client";

import React, { useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { RotationItem } from "@/lib/data/schemas";
import { Play, Pause } from "lucide-react";

interface RotationMapProps {
  benchmark: string;
  assets: RotationItem[];
}

export const RotationMap: React.FC<RotationMapProps> = ({
  benchmark,
  assets,
}) => {
  const [selectedGroup, setSelectedGroup] = useState<string>("all");
  const [showTails, setShowTails] = useState<boolean>(true);

  const groups = useMemo(() => {
    const set = new Set(assets.map((a) => a.group));
    return ["all", ...Array.from(set)];
  }, [assets]);

  const filteredAssets = useMemo(() => {
    if (selectedGroup === "all") return assets;
    return assets.filter((a) => a.group === selectedGroup);
  }, [assets, selectedGroup]);

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case "Leading":
        return "#3fb6a8"; // --pos (teal)
      case "Fading":
        return "#e6b94d"; // --warn (amber)
      case "Lagging":
        return "#e8765a"; // --neg (coral)
      case "Recovering":
      default:
        return "#6ea8ff"; // --accent (blue)
    }
  };

  const option = useMemo(() => {
    const seriesList: any[] = [];

    // Current Scatter points
    seriesList.push({
      name: "Current Position",
      type: "scatter",
      symbolSize: 14,
      data: filteredAssets.map((a) => ({
        name: a.name,
        value: [a.rs_ratio, a.rs_mom],
        item: a,
        itemStyle: {
          color: getPhaseColor(a.phase),
          borderColor: "#07090d",
          borderWidth: 1.5,
        },
      })),
      markLine: {
        silent: true,
        symbol: "none",
        lineStyle: { color: "#2a3647", type: "dashed" },
        data: [{ xAxis: 100.0 }, { yAxis: 100.0 }],
      },
    });

    // 10-day Trajectory Tails
    if (showTails) {
      filteredAssets.forEach((a) => {
        const lineCoords = a.tail_x.map((x, i) => [x, a.tail_y[i]]);
        seriesList.push({
          name: `${a.asset}_tail`,
          type: "line",
          data: lineCoords,
          symbol: "none",
          lineStyle: {
            color: getPhaseColor(a.phase),
            width: 1.2,
            opacity: 0.45,
          },
        });
      });
    }

    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: {
        top: 30,
        right: 35,
        bottom: 45,
        left: 45,
      },
      tooltip: {
        trigger: "item",
        backgroundColor: "#0c1017",
        borderColor: "#1d2633",
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          if (!params.data?.item) return "";
          const item = params.data.item as RotationItem;
          return `
            <strong>${item.name} (${item.asset.toUpperCase()})</strong><br/>
            Phase: <span style="color: ${getPhaseColor(item.phase)}">${item.phase}</span><br/>
            RS-Ratio: ${item.rs_ratio.toFixed(1)}<br/>
            RS-Momentum: ${item.rs_mom.toFixed(1)}
          `;
        },
      },
      xAxis: {
        type: "value",
        name: "RS-Ratio (vs Benchmark)",
        nameLocation: "middle",
        nameGap: 25,
        nameTextStyle: { color: "#6f7c90", fontSize: 10 },
        splitLine: { lineStyle: { color: "#1d2633", opacity: 0.5 } },
        axisLabel: { color: "#6f7c90", fontSize: 10, fontFamily: "monospace" },
      },
      yAxis: {
        type: "value",
        name: "RS-Momentum (10D)",
        nameLocation: "middle",
        nameGap: 30,
        nameTextStyle: { color: "#6f7c90", fontSize: 10 },
        splitLine: { lineStyle: { color: "#1d2633", opacity: 0.5 } },
        axisLabel: { color: "#6f7c90", fontSize: 10, fontFamily: "monospace" },
      },
      series: seriesList,
    };
  }, [filteredAssets, showTails]);

  return (
    <div className="space-y-3">
      {/* Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-1 border-b border-[var(--line)]">
        <div className="flex items-center gap-1.5 overflow-x-auto text-xs">
          {groups.map((grp) => (
            <button
              key={grp}
              onClick={() => setSelectedGroup(grp)}
              className={`px-2 py-0.5 rounded font-mono text-[10px] uppercase transition-colors ${
                selectedGroup === grp
                  ? "bg-[var(--accent)] text-white font-medium"
                  : "bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              }`}
            >
              {grp}
            </button>
          ))}
        </div>

        <button
          onClick={() => setShowTails(!showTails)}
          className="self-end sm:self-auto px-2 py-0.5 rounded border border-[var(--line)] bg-[var(--bg-0)] text-[11px] font-mono text-[var(--tx-2)] hover:text-[var(--tx-0)]"
        >
          {showTails ? "Hide 10D Tails" : "Show 10D Tails"}
        </button>
      </div>

      {/* Chart */}
      <div className="w-full h-[320px]">
        <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />
      </div>

      {/* Phase Legend */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-[var(--line)] text-xs font-mono text-[11px]">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[var(--pos)]" />
          <span className="text-[var(--tx-1)]">Leading (Top-Right)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[var(--warn)]" />
          <span className="text-[var(--tx-1)]">Fading (Bottom-Right)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[var(--neg)]" />
          <span className="text-[var(--tx-1)]">Lagging (Bottom-Left)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[var(--accent)]" />
          <span className="text-[var(--tx-1)]">Recovering (Top-Left)</span>
        </div>
      </div>
    </div>
  );
};
