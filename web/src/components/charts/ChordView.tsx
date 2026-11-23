// web/src/components/charts/ChordView.tsx
"use client";

import React, { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { ConnectednessEnvelope, ConnectednessCountryItem } from "@/lib/data/schemas";
import { EChartBox } from "@/components/charts/EChartBox";

interface Props {
  data: ConnectednessEnvelope;
  height?: number;
}

export function ChordView({ data, height = 400 }: Props) {
  const conn = (data as any).data || data;

  const option = useMemo<EChartsOption>(() => {
    if (!conn.matrix || !conn.countries || conn.countries.length === 0) return {};

    const countries: ConnectednessCountryItem[] = conn.countries;
    const names = countries.map((c) => c.name);

    // Nodes for circular graph layout
    const nodes = countries.map((c) => {
      const isTransmitter = c.net >= 0;
      return {
        name: c.name,
        symbolSize: Math.max(20, Math.min(45, 20 + Math.abs(c.net) * 1.5)),
        itemStyle: {
          color: isTransmitter ? "#34d399" : "#f87171",
          borderColor: "#ffffff",
          borderWidth: 1.5,
        },
        label: {
          show: true,
          position: "right" as const,
          color: "#e8edf5",
          fontSize: 11,
          fontFamily: "monospace",
        },
      };
    });

    // Links representing directed spillovers
    const links: any[] = [];
    const matrix: number[][] = conn.matrix;

    for (let i = 0; i < names.length; i++) {
      for (let j = 0; j < names.length; j++) {
        if (i === j) continue;
        const val = matrix[i]?.[j] ?? 0;
        if (val > 1.0) {
          // meaningful spillover threshold
          links.push({
            source: names[i],
            target: names[j],
            value: Number(val.toFixed(2)),
            lineStyle: {
              width: Math.max(1, Math.min(6, val / 4)),
              curveness: 0.3,
              opacity: 0.5,
              color: "#38bdf8",
            },
          });
        }
      }
    }

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "item",
        backgroundColor: "#0c1017",
        borderColor: "#1d2633",
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          if (params.dataType === "edge") {
            return `Spillover: <b>${params.data.source} → ${params.data.target}</b><br/>Variance Contribution: <b>${params.data.value}%</b>`;
          }
          const c = countries.find((x) => x.name === params.name);
          return `<b>${params.name}</b><br/>Role: <i>${c?.role ?? "neutral"}</i><br/>Net Spillover: <b>${c?.net ? (c.net > 0 ? "+" : "") + c.net.toFixed(2) : "0"}%</b>`;
        },
      },
      series: [
        {
          type: "graph",
          layout: "circular",
          circular: {
            rotateLabel: true,
          },
          data: nodes,
          links: links,
          roam: false,
          label: {
            position: "right",
            formatter: "{b}",
          },
          lineStyle: {
            color: "source",
            curveness: 0.3,
          },
          emphasis: {
            focus: "adjacency",
            lineStyle: {
              width: 5,
              opacity: 1,
            },
          },
        },
      ],
    };
  }, [conn]);

  return (
    <div className="w-full bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)] font-mono">
      <div className="flex items-center justify-between mb-3 border-b border-[var(--line)] pb-2">
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider text-[var(--tx-0)]">
            Circular Connectedness &amp; Transmission Graph
          </h4>
          <p className="text-[10px] text-[var(--tx-2)]">
            Directed pairwise forecast error variance spillovers between launch economies
          </p>
        </div>
        <span className="text-[10px] text-[var(--tx-2)]">Total Connectedness: {conn.tci}%</span>
      </div>
      <div style={{ height }}>
        <EChartBox option={option} height={height} />
      </div>
    </div>
  );
}
