// web/src/components/charts/RidgelineDensityChart.tsx
// C19: Ridgeline of return distributions across markets or regimes (Section 5.1 & 5.3)
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";

interface SeriesDensity {
  name: string;
  mean: number;
  std: number;
  current: number;
}

const SERIES_CONFIG: SeriesDensity[] = [
  { name: "S&P 500 (US)", mean: 0.04, std: 1.15, current: -3.70 },
  { name: "Nasdaq 100 (US)", mean: 0.05, std: 1.45, current: -4.10 },
  { name: "DAX (Germany)", mean: 0.03, std: 1.25, current: -2.60 },
  { name: "FTSE 100 (UK)", mean: 0.02, std: 0.95, current: -1.02 },
  { name: "Nikkei 225 (Japan)", mean: 0.03, std: 1.35, current: -5.06 },
  { name: "Nifty 50 (India)", mean: 0.06, std: 1.05, current: 1.12 },
  { name: "Crude Oil (WTI)", mean: 0.01, std: 2.45, current: 2.15 },
  { name: "Gold (COMEX)", mean: 0.03, std: 1.10, current: 0.45 },
];

export const RidgelineDensityChart: React.FC = () => {
  // Generate overlapping normal-KDE densities for each series
  const xValues = useMemo(() => {
    const xs: number[] = [];
    for (let x = -5.0; x <= 5.0; x += 0.2) {
      xs.push(parseFloat(x.toFixed(1)));
    }
    return xs;
  }, []);

  const seriesData = useMemo(() => {
    return SERIES_CONFIG.map((s, idx) => {
      const offset = (SERIES_CONFIG.length - 1 - idx) * 0.45; // vertical offset for ridgeline
      const lineData = xValues.map((x) => {
        const z = (x - s.mean) / s.std;
        const density = (1.0 / (s.std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z);
        return [x, parseFloat((offset + density * 0.9).toFixed(3))];
      });
      return {
        name: s.name,
        offset,
        data: lineData,
        current: s.current,
      };
    });
  }, [xValues]);

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "item",
      backgroundColor: "#0c1017",
      borderColor: "#1d2633",
      textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
      formatter: (params: any) => {
        return `
          <div style="font-weight:600;margin-bottom:2px">${params.seriesName}</div>
          <div style="color:#6f7c90">Return Coord: <span style="color:#6ea8ff">${params.value[0]}%</span></div>
          <div style="color:#6f7c90">Density Height: <span style="color:#3fb6a8">${params.value[1]}</span></div>
        `;
      },
    },
    grid: { left: 140, right: 30, top: 20, bottom: 40 },
    xAxis: {
      type: "value",
      min: -5,
      max: 5,
      name: "Daily Return (%)",
      nameLocation: "middle",
      nameGap: 25,
      nameTextStyle: { color: "#6f7c90", fontSize: 10, fontFamily: "monospace" },
      axisLine: { lineStyle: { color: "#1d2633" } },
      splitLine: { lineStyle: { color: "#121823" } },
      axisLabel: {
        color: "#a9b4c4",
        fontFamily: "monospace",
        fontSize: 10,
        formatter: (val: number) => `${val > 0 ? "+" : ""}${val}%`,
      },
    },
    yAxis: {
      type: "value",
      axisLine: { show: false },
      splitLine: { show: false },
      axisLabel: { show: false },
    },
    series: seriesData.map((s, idx) => ({
      name: s.name,
      type: "line",
      smooth: true,
      symbol: "none",
      data: s.data,
      lineStyle: { width: 1.5, color: "#6ea8ff" },
      areaStyle: {
        color: {
          type: "linear",
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: "rgba(110, 168, 255, 0.45)" },
            { offset: 1, color: "rgba(15, 23, 42, 0.05)" },
          ],
        },
      },
      markPoint: {
        symbol: "pin",
        symbolSize: 22,
        itemStyle: { color: s.current >= 0 ? "#3fb6a8" : "#e8765a" },
        data: [
          {
            name: "Current",
            coord: [s.current, s.offset + 0.15],
            label: { show: false },
          },
        ],
      },
    })),
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs font-mono text-[var(--tx-2)]">
        <span>Cross-Asset Return Ridgelines (8 Markets)</span>
        <span>Pins mark latest 1D move</span>
      </div>
      <ReactECharts option={option} style={{ height: "340px", width: "100%" }} />
    </div>
  );
};
