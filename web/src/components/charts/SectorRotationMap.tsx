// web/src/components/charts/SectorRotationMap.tsx
"use client";

import React, { useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { SectorItem } from "@/lib/data/schemas";
import { Eye, EyeOff } from "lucide-react";

interface SectorRotationMapProps {
  benchmarkName: string;
  sectors: SectorItem[];
}

export const SectorRotationMap: React.FC<SectorRotationMapProps> = ({
  benchmarkName,
  sectors,
}) => {
  const [showTails, setShowTails] = useState<boolean>(true);

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case "Leading":
        return "#3fb6a8"; // Teal / Pos
      case "Weakening":
      case "Fading":
        return "#e6b94d"; // Amber / Warn
      case "Lagging":
        return "#e8765a"; // Coral / Neg
      case "Improving":
      case "Recovering":
      default:
        return "#6ea8ff"; // Blue / Accent
    }
  };

  const option = useMemo(() => {
    const seriesList: any[] = [];

    // 1. Current Position Scatter Points
    seriesList.push({
      name: "Current Position",
      type: "scatter",
      symbolSize: 14,
      data: sectors.map((s) => ({
        name: s.name,
        value: [s.rs_ratio ?? 100, s.rs_mom ?? (s as any).rs_momentum ?? 100],
        item: s,
        itemStyle: {
          color: getPhaseColor(s.phase),
          borderColor: "#07090d",
          borderWidth: 2,
        },
        label: {
          show: true,
          formatter: "{b}",
          position: "right",
          color: "#a9b4c4",
          fontSize: 10,
          fontFamily: "var(--font-sans)",
        },
      })),
      markLine: {
        silent: true,
        symbol: "none",
        lineStyle: { color: "#2a3647", type: "dashed", width: 1.2 },
        data: [{ xAxis: 100.0 }, { yAxis: 100.0 }],
      },
    });

    // 2. Trajectory Tails (Past 10 Days)
    if (showTails) {
      sectors.forEach((s) => {
        const trailPoints = s.trail || (s as any).tail;
        if (trailPoints && trailPoints.length > 1) {
          const coords = trailPoints.map((tp: any) => [tp.rs_ratio ?? tp.x, tp.rs_mom ?? tp.y]);
          seriesList.push({
            name: `${s.id}_trail`,
            type: "line",
            data: coords,
            symbol: "none",
            lineStyle: {
              color: getPhaseColor(s.phase),
              width: 1.2,
              opacity: 0.45,
            },
          });
        }
      });
    }

    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: {
        top: 40,
        right: 80,
        bottom: 50,
        left: 55,
      },
      tooltip: {
        trigger: "item",
        backgroundColor: "#0c1017",
        borderColor: "#1d2633",
        textStyle: { color: "#e8edf5", fontSize: 12 },
        formatter: (params: any) => {
          if (!params.data || !params.data.item) return "";
          const s = params.data.item as SectorItem;
          const [rs, mom] = params.data.value;
          const color = getPhaseColor(s.phase);
          return `
            <div class="font-sans">
              <div class="font-semibold text-xs border-b border-[#1d2633] pb-1 mb-1 text-[#e8edf5]">${s.name}</div>
              <div class="flex items-center space-x-1.5 mb-1.5">
                <span class="w-2 h-2 rounded-full" style="background-color: ${color}"></span>
                <span class="text-xs font-semibold" style="color: ${color}">${s.phase}</span>
              </div>
              <div class="text-[11px] text-[#8b98a5]">RS Ratio (vs ${benchmarkName}): <span class="font-mono text-[#e8edf5] font-semibold">${rs.toFixed(2)}</span></div>
              <div class="text-[11px] text-[#8b98a5]">RS Momentum: <span class="font-mono text-[#e8edf5] font-semibold">${mom.toFixed(2)}</span></div>
              <div class="text-[11px] text-[#8b98a5]">Index Weight: <span class="font-mono text-[#e8edf5] font-semibold">${(s.weight * 100).toFixed(1)}%</span></div>
            </div>
          `;
        },
      },
      xAxis: {
        type: "value",
        name: `Relative Strength vs ${benchmarkName}`,
        nameLocation: "center",
        nameGap: 30,
        nameTextStyle: { color: "#8b98a5", fontSize: 11 },
        scale: true,
        splitLine: { lineStyle: { color: "#161d28" } },
        axisLabel: { color: "#8b98a5", fontSize: 10 },
      },
      yAxis: {
        type: "value",
        name: "Momentum (Rate of Change)",
        nameLocation: "center",
        nameGap: 35,
        nameTextStyle: { color: "#8b98a5", fontSize: 11 },
        scale: true,
        splitLine: { lineStyle: { color: "#161d28" } },
        axisLabel: { color: "#8b98a5", fontSize: 10 },
      },
      series: seriesList,
    };
  }, [sectors, showTails, benchmarkName]);

  return (
    <div className="space-y-2">
      {/* Quadrant Legend & Tail Toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs border-b border-[#1d2633] pb-2">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#3fb6a8]"></span>
            <span className="text-[#a9b4c4]">Leading</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#e6b94d]"></span>
            <span className="text-[#a9b4c4]">Weakening</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#e8765a]"></span>
            <span className="text-[#a9b4c4]">Lagging</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#6ea8ff]"></span>
            <span className="text-[#a9b4c4]">Improving</span>
          </div>
        </div>

        <button
          type="button"
          onClick={() => setShowTails(!showTails)}
          className="flex items-center space-x-1 px-2.5 py-1 rounded bg-[#121823] hover:bg-[#1d2633] text-[#8b98a5] hover:text-[#e8edf5] text-xs transition"
        >
          {showTails ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          <span>{showTails ? "Hide 10D Tails" : "Show 10D Tails"}</span>
        </button>
      </div>

      <div className="h-[380px] w-full">
        <ReactECharts
          option={option}
          style={{ height: "100%", width: "100%" }}
          opts={{ renderer: "svg" }}
        />
      </div>
    </div>
  );
};
