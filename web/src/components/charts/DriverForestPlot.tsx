// web/src/components/charts/DriverForestPlot.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { DriverTargetMatrixRow } from "@/lib/data/schemas";

interface DriverForestPlotProps {
  targetRow: DriverTargetMatrixRow;
  selectedDriverId: string;
  onSelectDriver: (driverId: string) => void;
}

const DRIVER_LABELS: Record<string, string> = {
  row_equity: "Rest-of-World Equity",
  brent: "Brent Crude Oil",
  copper: "Copper (Cyclical)",
  gold: "Gold (Safe Haven)",
  dxy: "US Dollar Index",
  us10y: "US 10Y Yield",
};

export const DriverForestPlot: React.FC<DriverForestPlotProps> = ({
  targetRow,
  selectedDriverId,
  onSelectDriver,
}) => {
  const driverKeys = Object.keys(targetRow.drivers);

  const categories = useMemo(() => {
    return driverKeys.map((k) => DRIVER_LABELS[k] || k);
  }, [driverKeys]);

  const option = useMemo(() => {
    const errorBarData: any[] = [];
    const pointData: any[] = [];

    driverKeys.forEach((k, idx) => {
      const item = targetRow.drivers[k];
      const isSelected = k === selectedDriverId;
      const isSignificant = item.significant;

      let color = item.beta >= 0 ? "#3fb6a8" : "#e8765a";
      if (!isSignificant) color = "#6f7c90";

      // Whisker interval [ci_lower, ci_upper, category_index]
      errorBarData.push([item.ci_lower, item.ci_upper, idx]);

      pointData.push({
        name: DRIVER_LABELS[k] || k,
        value: [item.beta, idx],
        itemStyle: {
          color: isSelected ? "#6ea8ff" : color,
          borderColor: "#07090d",
          borderWidth: 2,
        },
      });
    });

    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: {
        top: 25,
        right: 40,
        bottom: 30,
        left: 155,
      },
      tooltip: {
        trigger: "item",
        backgroundColor: "#0c1017",
        borderColor: "#1d2633",
        textStyle: { color: "#e8edf5", fontSize: 12 },
        formatter: (params: any) => {
          const idx = params.value ? params.value[1] : params.dataIndex;
          if (idx === undefined || idx < 0 || idx >= driverKeys.length) return "";
          const k = driverKeys[idx];
          const item = targetRow.drivers[k];
          return `
            <div class="font-sans">
              <div class="font-semibold text-xs border-b border-[#1d2633] pb-1 mb-1 text-[#e8edf5]">
                ${DRIVER_LABELS[k] || k} Sensitivity
              </div>
              <div class="text-[11px] text-[#8b98a5]">Estimated Beta: <span class="font-mono text-[#e8edf5] font-semibold">${item.beta.toFixed(4)}</span></div>
              <div class="text-[11px] text-[#8b98a5]">95% HAC Confidence Interval: <span class="font-mono text-[#e8edf5]">[${item.ci_lower.toFixed(4)}, ${item.ci_upper.toFixed(4)}]</span></div>
              <div class="text-[11px] text-[#8b98a5]">T-Statistic: <span class="font-mono text-[#e8edf5]">${item.t_stat.toFixed(2)} (p = ${item.p_value.toFixed(3)})</span></div>
              <div class="text-[11px] text-[#8b98a5]">Stability: <span class="font-mono text-[#e8edf5]">${(item.stability * 100).toFixed(0)}% same sign</span></div>
              <div class="text-[11px] text-[#8b98a5]">Economic Prior: <span class="font-mono text-[#e8edf5] uppercase">${item.prior}</span> (${item.prior_aligned ? "Aligned ✓" : "Divergent ✕"})</div>
            </div>
          `;
        },
      },
      xAxis: {
        type: "value",
        name: "Beta Sensitivity (β)",
        nameLocation: "center",
        nameGap: 25,
        nameTextStyle: { color: "#8b98a5", fontSize: 10 },
        splitLine: { lineStyle: { color: "#161d28" } },
        axisLabel: { color: "#8b98a5", fontSize: 10 },
      },
      yAxis: {
        type: "category",
        data: categories,
        inverse: true,
        axisLine: { lineStyle: { color: "#1d2633" } },
        axisTick: { show: false },
        axisLabel: {
          color: "#a9b4c4",
          fontSize: 11,
        },
      },
      series: [
        // Whiskers using custom renderItem
        {
          type: "custom",
          name: "95% Confidence Interval",
          renderItem: (params: any, api: any) => {
            const categoryIndex = api.value(2);
            const low = api.coord([api.value(0), categoryIndex]);
            const high = api.coord([api.value(1), categoryIndex]);
            const halfHeight = 5;

            return {
              type: "group",
              children: [
                {
                  type: "line",
                  shape: { x1: low[0], y1: low[1], x2: high[0], y2: high[1] },
                  style: api.style({ stroke: "#6f7c90", lineWidth: 1.5 }),
                },
                {
                  type: "line",
                  shape: { x1: low[0], y1: low[1] - halfHeight, x2: low[0], y2: low[1] + halfHeight },
                  style: api.style({ stroke: "#6f7c90", lineWidth: 1.5 }),
                },
                {
                  type: "line",
                  shape: { x1: high[0], y1: high[1] - halfHeight, x2: high[0], y2: high[1] + halfHeight },
                  style: api.style({ stroke: "#6f7c90", lineWidth: 1.5 }),
                },
              ],
            };
          },
          encode: { x: [0, 1], y: 2 },
          data: errorBarData,
        },
        // Point Estimate
        {
          name: "Beta Estimate",
          type: "scatter",
          symbolSize: 10,
          data: pointData,
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "#e8edf5", type: "dashed", width: 1 },
            data: [{ xAxis: 0.0 }],
          },
        },
      ],
    };
  }, [targetRow, driverKeys, categories, selectedDriverId]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-[var(--tx-2)] border-b border-[var(--line)] pb-2">
        <span>
          Showing 95% Newey-West HAC intervals for <strong className="text-[var(--tx-0)]">{targetRow.target_name}</strong>
        </span>
        <span className="font-mono text-[11px] text-[var(--tx-3)]">
          R² = {(targetRow.r2 * 100).toFixed(1)}% (N = {targetRow.n_obs})
        </span>
      </div>

      <div className="h-[280px] w-full">
        <ReactECharts
          option={option}
          style={{ height: "100%", width: "100%" }}
          opts={{ renderer: "svg" }}
          onEvents={{
            click: (params: any) => {
              const idx = params.dataIndex;
              if (idx !== undefined && idx >= 0 && idx < driverKeys.length) {
                onSelectDriver(driverKeys[idx]);
              }
            },
          }}
        />
      </div>
    </div>
  );
};
