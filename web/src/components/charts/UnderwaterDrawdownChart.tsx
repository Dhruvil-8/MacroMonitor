// web/src/components/charts/UnderwaterDrawdownChart.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { DrawdownEpisodeItem, DrawdownSeriesPoint } from "@/lib/data/schemas";

interface UnderwaterDrawdownChartProps {
  series: DrawdownSeriesPoint[];
  episodes: DrawdownEpisodeItem[];
  assetName: string;
}

export const UnderwaterDrawdownChart: React.FC<UnderwaterDrawdownChartProps> = ({
  series,
  episodes,
  assetName,
}) => {
  const option = useMemo(() => {
    if (!series || series.length === 0) return {};

    const dates = series.map((d) => d.date);
    const ddValues = series.map((d) => (d.drawdown * 100).toFixed(2));

    return {
      backgroundColor: "transparent",
      animationDuration: 400,
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(10, 14, 20, 0.95)",
        borderColor: "rgba(255, 255, 255, 0.12)",
        textStyle: { color: "#e6edf3", fontSize: 12, fontFamily: "var(--font-mono)" },
        formatter: (params: any[]) => {
          const pt = params[0];
          return `
            <div style="font-weight: 600; margin-bottom: 4px; color: #f85149;">${assetName} Underwater Drawdown</div>
            <div style="color: #8b949e;">Date: <b style="color: #e6edf3;">${pt.name}</b></div>
            <div style="color: #8b949e;">Drawdown from Peak: <b style="color: #f85149;">${pt.value}%</b></div>
          `;
        },
      },
      grid: {
        top: 20,
        right: 25,
        bottom: 30,
        left: 55,
      },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: { lineStyle: { color: "rgba(255, 255, 255, 0.15)" } },
        axisLabel: {
          color: "#8b949e",
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          formatter: (v: string) => v.slice(0, 7),
        },
      },
      yAxis: {
        type: "value",
        max: 0,
        axisLabel: {
          formatter: "{value}%",
          color: "#8b949e",
          fontFamily: "var(--font-mono)",
          fontSize: 10,
        },
        splitLine: { lineStyle: { color: "rgba(255, 255, 255, 0.05)" } },
      },
      series: [
        {
          name: "Underwater Drawdown",
          type: "line",
          data: ddValues,
          lineStyle: { color: "#f85149", width: 1.5 },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(248, 81, 73, 0.05)" },
                { offset: 1, color: "rgba(248, 81, 73, 0.45)" },
              ],
            },
          },
          symbol: "none",
        },
      ],
    };
  }, [series, assetName]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{ width: "100%", height: 260 }}>
        <ReactECharts
          option={option}
          style={{ width: "100%", height: "100%" }}
          opts={{ renderer: "canvas" }}
        />
      </div>

      {episodes && episodes.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#8b949e", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Top 5 Deepest Historical Drawdowns
          </div>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: 12,
              fontFamily: "var(--font-mono)",
              textAlign: "left",
            }}
          >
            <thead>
              <tr style={{ borderBottom: "1px solid rgba(255, 255, 255, 0.1)", color: "#8b949e" }}>
                <th style={{ padding: "6px 8px" }}>Peak Date</th>
                <th style={{ padding: "6px 8px" }}>Trough Date</th>
                <th style={{ padding: "6px 8px" }}>Recovery Date</th>
                <th style={{ padding: "6px 8px", textAlign: "right" }}>Depth</th>
                <th style={{ padding: "6px 8px", textAlign: "right" }}>Duration</th>
                <th style={{ padding: "6px 8px", textAlign: "right" }}>Recovery Days</th>
                <th style={{ padding: "6px 8px" }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {episodes.map((ep, idx) => (
                <tr
                  key={idx}
                  style={{
                    borderBottom: "1px solid rgba(255, 255, 255, 0.05)",
                    backgroundColor: idx % 2 === 0 ? "rgba(255, 255, 255, 0.01)" : "transparent",
                  }}
                >
                  <td style={{ padding: "6px 8px", color: "#c9d1d9" }}>{ep.peak_date}</td>
                  <td style={{ padding: "6px 8px", color: "#c9d1d9" }}>{ep.trough_date}</td>
                  <td style={{ padding: "6px 8px", color: ep.recovery_date === "Active" ? "#f0883e" : "#c9d1d9" }}>
                    {ep.recovery_date || "Active"}
                  </td>
                  <td style={{ padding: "6px 8px", textAlign: "right", color: "#f85149", fontWeight: "bold" }}>
                    {(ep.depth * 100).toFixed(2)}%
                  </td>
                  <td style={{ padding: "6px 8px", textAlign: "right", color: "#8b949e" }}>
                    {ep.duration_days} d
                  </td>
                  <td style={{ padding: "6px 8px", textAlign: "right", color: "#8b949e" }}>
                    {ep.recovery_days !== null ? `${ep.recovery_days} d` : "—"}
                  </td>
                  <td style={{ padding: "6px 8px" }}>
                    <span
                      style={{
                        display: "inline-block",
                        padding: "2px 6px",
                        borderRadius: "3px",
                        fontSize: 10,
                        backgroundColor:
                          ep.status === "Recovered"
                            ? "rgba(63, 185, 80, 0.15)"
                            : "rgba(240, 136, 62, 0.15)",
                        color: ep.status === "Recovered" ? "#3fb950" : "#f0883e",
                      }}
                    >
                      {ep.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
