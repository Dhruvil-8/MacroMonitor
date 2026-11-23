// web/src/components/charts/CurveRibbon.tsx
"use client";

import React, { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { YieldCurveEnvelope } from "@/lib/data/schemas";
import { EChartBox } from "@/components/charts/EChartBox";

interface Props {
  data: YieldCurveEnvelope;
  height?: number;
}

export function CurveRibbon({ data, height = 320 }: Props) {
  const yc = (data as any).data || data;
  const textCol = "#8c9bb0";
  const gridLine = "#1d2633";

  const option = useMemo<EChartsOption>(() => {
    if (!yc.curves || yc.curves.length === 0) return {};

    const colors = ["#38bdf8", "#34d399", "#fbbf24", "#94a3b8"];
    const seriesList: any[] = [];
    const legendData: string[] = [];

    yc.curves.forEach((c: any, idx: number) => {
      const color = colors[idx % colors.length];
      legendData.push(c.label);

      // Actual points (scatter/dots)
      seriesList.push({
        name: c.label,
        type: "scatter",
        symbolSize: 8,
        itemStyle: { color, borderColor: "#ffffff", borderWidth: 1.5 },
        data: c.points.map((p: any) => [p.months, p.yield_pct]),
        tooltip: {
          formatter: (params: any) =>
            `<b>${c.label} (${c.date})</b><br/>${params.value[0]} Months: <b>${params.value[1]}%</b>`,
        },
      });

      // Smooth fitted curve line (Nelson-Siegel)
      if (c.smooth_fitted && c.smooth_fitted.length > 0) {
        seriesList.push({
          name: `${c.label} (NS Fit)`,
          type: "line",
          showSymbol: false,
          smooth: true,
          lineStyle: { color, width: idx === 0 ? 2.5 : 1.5, type: idx === 0 ? "solid" : "dashed" },
          data: c.smooth_fitted.map((p: any) => [p.months, p.yield_pct]),
        });
      }
    });

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "item",
        backgroundColor: "#0c1017",
        borderColor: gridLine,
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
      },
      legend: {
        top: 0,
        textStyle: { color: textCol, fontSize: 11 },
        data: legendData,
      },
      grid: { left: 45, right: 30, top: 40, bottom: 35 },
      xAxis: {
        type: "value",
        name: "Maturity (Months)",
        nameTextStyle: { color: textCol, fontSize: 10 },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace" },
      },
      yAxis: {
        type: "value",
        name: "Yield (%)",
        nameTextStyle: { color: textCol, fontSize: 10 },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace", formatter: "{value}%" },
      },
      series: seriesList,
    };
  }, [yc]);

  return (
    <div className="w-full bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)]">
      <div className="flex items-center justify-between mb-3 border-b border-[var(--line)] pb-2">
        <h4 className="text-xs font-bold uppercase tracking-wider font-mono text-[var(--tx-0)]">
          Nelson-Siegel Yield Curve Ribbon
        </h4>
        <span className="text-[10px] font-mono text-[var(--tx-2)]">As of: {yc.as_of}</span>
      </div>
      <div style={{ height }}>
        <EChartBox option={option} height={height} />
      </div>
    </div>
  );
}
