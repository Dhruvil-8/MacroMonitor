// web/src/components/charts/NetSpilloverBars.tsx
"use client";

import React, { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { ConnectednessEnvelope, ConnectednessCountryItem } from "@/lib/data/schemas";
import { EChartBox } from "@/components/charts/EChartBox";

interface Props {
  data: ConnectednessEnvelope;
  height?: number;
}

export function NetSpilloverBars({ data, height = 300 }: Props) {
  const conn = (data as any).data || data;
  const textCol = "#8c9bb0";
  const gridLine = "#1d2633";

  const option = useMemo<EChartsOption>(() => {
    if (!conn.countries || conn.countries.length === 0) return {};

    const countries = conn.countries.map((c: ConnectednessCountryItem) => c.name);
    const netVals = conn.countries.map((c: ConnectednessCountryItem) => c.net);

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: "#0c1017",
        borderColor: gridLine,
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          const item = params[0];
          const c = conn.countries[item.dataIndex];
          return `<b>${c.name}</b><br/>Net Spillover: <b>${c.net > 0 ? "+" : ""}${c.net.toFixed(2)}%</b><br/>Transmitted to Others: ${c.to_others.toFixed(2)}%<br/>Received from Others: ${c.from_others.toFixed(2)}%<br/>Role: <i>${c.role}</i>`;
        },
      },
      grid: { left: 80, right: 30, top: 20, bottom: 25 },
      xAxis: {
        type: "value",
        name: "Net Spillover % (To - From)",
        nameTextStyle: { color: textCol, fontSize: 10 },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace", formatter: "{value}%" },
      },
      yAxis: {
        type: "category",
        data: countries,
        axisLabel: { color: textCol, fontSize: 11, fontFamily: "monospace" },
        axisLine: { lineStyle: { color: gridLine } },
      },
      series: [
        {
          name: "Net Spillover",
          type: "bar",
          data: netVals.map((v: number) => ({
            value: v,
            itemStyle: {
              color: v >= 0 ? "#34d399" : "#f87171",
              borderRadius: v >= 0 ? [0, 3, 3, 0] : [3, 0, 0, 3],
            },
          })),
        },
      ],
    };
  }, [conn]);

  return (
    <div className="w-full bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)]">
      <div className="flex items-center justify-between mb-3 border-b border-[var(--line)] pb-2">
        <h4 className="text-xs font-bold uppercase tracking-wider font-mono text-[var(--tx-0)]">
          Net Cross-Country Spillover Balance
        </h4>
        <span className="text-[10px] font-mono text-[var(--tx-2)]">Diebold-Yilmaz GFEVD (Horizon = 10W)</span>
      </div>
      <div style={{ height }}>
        <EChartBox option={option} height={height} />
      </div>
    </div>
  );
}
