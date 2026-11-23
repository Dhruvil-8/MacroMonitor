// web/src/components/charts/RollingBetaChart.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { DriverSensitivityItem } from "@/lib/data/schemas";

interface RollingBetaChartProps {
  driverItem: DriverSensitivityItem;
  targetName: string;
  driverName: string;
}

export const RollingBetaChart: React.FC<RollingBetaChartProps> = ({
  driverItem,
  targetName,
  driverName,
}) => {
  const history = driverItem.history || [];

  const option = useMemo(() => {
    if (history.length === 0) {
      return {};
    }

    const dates = history.map((h) => h.date);
    const betas = history.map((h) => h.beta);
    const ciLows = history.map((h) => h.ci_lower);
    const ciHighs = history.map((h) => h.ci_upper);

    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: {
        top: 25,
        right: 25,
        bottom: 35,
        left: 55,
      },
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0c1017",
        borderColor: "#1d2633",
        textStyle: { color: "#e8edf5", fontSize: 12 },
        formatter: (params: any) => {
          if (!params || params.length === 0) return "";
          const date = params[0].axisValue;
          const idx = params[0].dataIndex;
          const b = betas[idx];
          const low = ciLows[idx];
          const high = ciHighs[idx];
          return `
            <div class="font-sans">
              <div class="font-semibold text-xs border-b border-[#1d2633] pb-1 mb-1 text-[#e8edf5]">${date}</div>
              <div class="text-[11px] text-[#8b98a5]">Rolling Beta: <span class="font-mono text-[#e8edf5] font-semibold">${b.toFixed(4)}</span></div>
              <div class="text-[11px] text-[#8b98a5]">95% Confidence Band: <span class="font-mono text-[#e8edf5]">[${low.toFixed(4)}, ${high.toFixed(4)}]</span></div>
            </div>
          `;
        },
      },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: { lineStyle: { color: "#1d2633" } },
        axisLabel: { color: "#8b98a5", fontSize: 10 },
      },
      yAxis: {
        type: "value",
        scale: true,
        splitLine: { lineStyle: { color: "#161d28" } },
        axisLabel: { color: "#8b98a5", fontSize: 10 },
      },
      series: [
        // CI Upper Band (for area fill)
        {
          name: "CI High",
          type: "line",
          data: ciHighs,
          lineStyle: { opacity: 0 },
          stack: "confidence-band",
          symbol: "none",
        },
        // CI Ribbon fill
        {
          name: "CI Low",
          type: "line",
          data: ciLows.map((low, i) => ciHighs[i] - low),
          lineStyle: { opacity: 0 },
          areaStyle: { color: "rgba(110, 168, 255, 0.12)" },
          stack: "confidence-band",
          symbol: "none",
        },
        // Main Beta Line
        {
          name: "Beta Estimate",
          type: "line",
          data: betas,
          symbol: "none",
          lineStyle: { color: "#6ea8ff", width: 2 },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "#e8edf5", type: "dashed", width: 1 },
            data: [{ yAxis: 0.0 }],
          },
        },
      ],
    };
  }, [history]);

  if (history.length === 0) {
    return (
      <div className="h-[240px] flex items-center justify-center text-xs text-[var(--tx-3)] font-mono">
        Insufficient historical rolling windows for this pair.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-[var(--tx-2)] border-b border-[var(--line)] pb-2">
        <span>
          Rolling Beta Drift: <strong className="text-[var(--tx-0)]">{targetName}</strong> vs{" "}
          <strong className="text-[var(--accent)]">{driverName}</strong>
        </span>
        <span className="font-mono text-[11px] text-[var(--tx-3)]">
          Current β = {driverItem.beta.toFixed(3)} (Stability: {(driverItem.stability * 100).toFixed(0)}%)
        </span>
      </div>

      <div className="h-[240px] w-full">
        <ReactECharts
          option={option}
          style={{ height: "100%", width: "100%" }}
          opts={{ renderer: "svg" }}
        />
      </div>
    </div>
  );
};
