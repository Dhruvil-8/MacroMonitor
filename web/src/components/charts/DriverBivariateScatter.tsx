// web/src/components/charts/DriverBivariateScatter.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { DriverScatterData } from "@/lib/data/schemas";

interface DriverBivariateScatterProps {
  scatterData: DriverScatterData;
  targetName: string;
  driverName: string;
}

export const DriverBivariateScatter: React.FC<DriverBivariateScatterProps> = ({
  scatterData,
  targetName,
  driverName,
}) => {
  const option = useMemo(() => {
    if (!scatterData || scatterData.scatter.length === 0) {
      return {};
    }

    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: {
        top: 25,
        right: 25,
        bottom: 45,
        left: 55,
      },
      tooltip: {
        trigger: "item",
        backgroundColor: "#0c1017",
        borderColor: "#1d2633",
        textStyle: { color: "#e8edf5", fontSize: 12 },
        formatter: (params: any) => {
          if (!params || !params.value) return "";
          const [x, y] = params.value;
          return `
            <div class="font-sans">
              <div class="text-[11px] text-[#8b98a5]">${driverName} Move: <span class="font-mono text-[#e8edf5] font-semibold">${x.toFixed(2)}%</span></div>
              <div class="text-[11px] text-[#8b98a5]">${targetName} Return: <span class="font-mono text-[#e8edf5] font-semibold">${y.toFixed(2)}%</span></div>
            </div>
          `;
        },
      },
      xAxis: {
        type: "value",
        name: `${driverName} Daily Move (%)`,
        nameLocation: "center",
        nameGap: 28,
        nameTextStyle: { color: "#8b98a5", fontSize: 10 },
        splitLine: { lineStyle: { color: "#161d28" } },
        axisLabel: { color: "#8b98a5", fontSize: 10, formatter: "{value}%" },
      },
      yAxis: {
        type: "value",
        name: `${targetName} Daily Return (%)`,
        nameLocation: "center",
        nameGap: 35,
        nameTextStyle: { color: "#8b98a5", fontSize: 10 },
        splitLine: { lineStyle: { color: "#161d28" } },
        axisLabel: { color: "#8b98a5", fontSize: 10, formatter: "{value}%" },
      },
      series: [
        // Daily Scatter Points
        {
          name: "Observations",
          type: "scatter",
          symbolSize: 6,
          data: scatterData.scatter,
          itemStyle: {
            color: "rgba(110, 168, 255, 0.45)",
          },
        },
        // Fitted OLS Regression Line
        {
          name: "Fitted OLS Line",
          type: "line",
          data: scatterData.line,
          symbol: "none",
          lineStyle: {
            color: "#3fb6a8",
            width: 2,
          },
        },
        // Upper Confidence Band
        {
          name: "CI Upper",
          type: "line",
          data: scatterData.band_upper,
          symbol: "none",
          lineStyle: {
            color: "rgba(63, 182, 168, 0.4)",
            type: "dashed",
            width: 1,
          },
        },
        // Lower Confidence Band
        {
          name: "CI Lower",
          type: "line",
          data: scatterData.band_lower,
          symbol: "none",
          lineStyle: {
            color: "rgba(63, 182, 168, 0.4)",
            type: "dashed",
            width: 1,
          },
        },
      ],
    };
  }, [scatterData, targetName, driverName]);

  if (!scatterData || scatterData.scatter.length === 0) {
    return (
      <div className="h-[240px] flex items-center justify-center text-xs text-[var(--tx-3)] font-mono">
        No bivariate scatter data available for this pair.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-[var(--tx-2)] border-b border-[var(--line)] pb-2">
        <span>
          Bivariate Fit (C09): <strong className="text-[var(--tx-0)]">{targetName}</strong> vs{" "}
          <strong className="text-[var(--accent)]">{driverName}</strong>
        </span>
        <div className="flex items-center space-x-3 font-mono text-[11px] text-[var(--tx-3)]">
          <span>β = {scatterData.beta.toFixed(3)}</span>
          <span>R² = {(scatterData.r2 * 100).toFixed(1)}%</span>
          <span>t = {scatterData.t_stat.toFixed(1)}</span>
        </div>
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
