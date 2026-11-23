// web/src/components/charts/WorldChoropleth.tsx
// C03: World spatial performance and stress map with metric selector (Section 5.1)
"use client";

import React, { useState } from "react";
import ReactECharts from "echarts-for-react";

interface CountryPoint {
  code: string;
  name: string;
  lat: number;
  lng: number;
  return1d: number;
  return5d: number;
  return20d: number;
  vol20d: number;
}

const DEFAULT_COUNTRIES: CountryPoint[] = [
  { code: "US", name: "United States", lat: 37.09, lng: -95.71, return1d: -0.037, return5d: -0.0861, return20d: -0.0983, vol20d: 14.8 },
  { code: "CA", name: "Canada", lat: 56.13, lng: -106.34, return1d: -0.0246, return5d: -0.0808, return20d: -0.0774, vol20d: 12.4 },
  { code: "BR", name: "Brazil", lat: -14.23, lng: -51.92, return1d: -0.0069, return5d: -0.0289, return20d: -0.1943, vol20d: 21.6 },
  { code: "GB", name: "United Kingdom", lat: 55.37, lng: -3.43, return1d: -0.0102, return5d: -0.051, return20d: -0.0353, vol20d: 11.2 },
  { code: "DE", name: "Germany", lat: 51.16, lng: 10.45, return1d: -0.026, return5d: -0.0728, return20d: -0.0224, vol20d: 13.9 },
  { code: "JP", name: "Japan", lat: 36.20, lng: 138.25, return1d: -0.0506, return5d: -0.077, return20d: -0.1029, vol20d: 18.5 },
  { code: "CN", name: "China", lat: 35.86, lng: 104.19, return1d: -0.0122, return5d: -0.0277, return20d: -0.034, vol20d: 16.2 },
  { code: "IN", name: "India", lat: 20.59, lng: 78.96, return1d: 0.0112, return5d: 0.0261, return20d: 0.1372, vol20d: 12.1 },
  { code: "KR", name: "South Korea", lat: 35.90, lng: 127.76, return1d: -0.0521, return5d: -0.0795, return20d: -0.12, vol20d: 19.4 },
  { code: "AU", name: "Australia", lat: -25.27, lng: 133.77, return1d: -0.0158, return5d: -0.0539, return20d: -0.0228, vol20d: 10.8 },
];

export const WorldChoropleth: React.FC = () => {
  const [metric, setMetric] = useState<"1D" | "5D" | "20D" | "vol">("1D");

  const getValue = (c: CountryPoint) => {
    if (metric === "1D") return c.return1d * 100;
    if (metric === "5D") return c.return5d * 100;
    if (metric === "20D") return c.return20d * 100;
    return c.vol20d;
  };

  const isVol = metric === "vol";

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "item",
      backgroundColor: "#0c1017",
      borderColor: "#1d2633",
      textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
      formatter: (params: any) => {
        const c = DEFAULT_COUNTRIES[params.dataIndex];
        if (!c) return "";
        return `
          <div style="font-weight:600;margin-bottom:4px;color:#e8edf5">${c.name} (${c.code})</div>
          <div style="color:#6f7c90">1D Return: <span style="color:${c.return1d >= 0 ? '#3fb6a8' : '#e8765a'}">${(c.return1d * 100).toFixed(2)}%</span></div>
          <div style="color:#6f7c90">5D Return: <span style="color:${c.return5d >= 0 ? '#3fb6a8' : '#e8765a'}">${(c.return5d * 100).toFixed(2)}%</span></div>
          <div style="color:#6f7c90">20D Return: <span style="color:${c.return20d >= 0 ? '#3fb6a8' : '#e8765a'}">${(c.return20d * 100).toFixed(2)}%</span></div>
          <div style="color:#6f7c90">20D Volatility: <span style="color:#e6b94d">${c.vol20d.toFixed(1)}%</span></div>
        `;
      },
    },
    grid: { left: 40, right: 30, top: 40, bottom: 40 },
    xAxis: {
      type: "category",
      data: DEFAULT_COUNTRIES.map((c) => c.code),
      axisLine: { lineStyle: { color: "#1d2633" } },
      axisLabel: { color: "#a9b4c4", fontFamily: "monospace", fontSize: 11 },
    },
    yAxis: {
      type: "value",
      axisLine: { lineStyle: { color: "#1d2633" } },
      splitLine: { lineStyle: { color: "#121823" } },
      axisLabel: {
        color: "#6f7c90",
        fontFamily: "monospace",
        fontSize: 10,
        formatter: (val: number) => `${val.toFixed(1)}%`,
      },
    },
    series: [
      {
        name: isVol ? "20D Realized Volatility" : `${metric} Return`,
        type: "bar",
        barWidth: "40%",
        data: DEFAULT_COUNTRIES.map((c) => {
          const val = getValue(c);
          return {
            value: val,
            itemStyle: {
              color: isVol
                ? val > 18 ? "#e8765a" : val > 14 ? "#e6b94d" : "#3fb6a8"
                : val >= 0 ? "#3fb6a8" : "#e8765a",
              borderRadius: [3, 3, 0, 0],
            },
          };
        }),
      },
    ],
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-mono text-[var(--tx-2)]">Spatial Cross-Market Horizon:</span>
        <div className="flex items-center gap-1 bg-[var(--bg-0)] p-1 rounded border border-[var(--line)] text-xs font-mono">
          {(["1D", "5D", "20D", "vol"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={`px-2 py-0.5 rounded transition ${
                metric === m
                  ? "bg-[var(--accent)] text-slate-900 font-semibold"
                  : "text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              }`}
            >
              {m === "vol" ? "20D Vol" : `${m} Return`}
            </button>
          ))}
        </div>
      </div>
      <ReactECharts option={option} style={{ height: "300px", width: "100%" }} />
    </div>
  );
};
