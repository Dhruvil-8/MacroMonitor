// web/src/components/charts/SectorContributionWaterfall.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { SectorItem } from "@/lib/data/schemas";
import { Info } from "lucide-react";

interface SectorContributionWaterfallProps {
  benchmarkName: string;
  indexReturn: number;
  residual: number;
  sectors: SectorItem[];
  horizon: string;
  weightsSource: string;
  asOf: string;
}

export const SectorContributionWaterfall: React.FC<SectorContributionWaterfallProps> = ({
  benchmarkName,
  indexReturn,
  residual,
  sectors,
  horizon,
  weightsSource,
  asOf,
}) => {
  // Sort sectors by contribution for selected horizon (descending)
  const sortedSectors = useMemo(() => {
    return [...sectors].sort((a, b) => {
      const ca = a.contributions?.[horizon] ?? 0;
      const cb = b.contributions?.[horizon] ?? 0;
      return cb - ca;
    });
  }, [sectors, horizon]);

  const categories = useMemo(() => {
    const names = sortedSectors.map((s) => s.name);
    return [...names, "Residual (Unmodeled / Drift)", `${benchmarkName} Total`];
  }, [sortedSectors, benchmarkName]);

  const values = useMemo(() => {
    const vals = sortedSectors.map((s) => (s.contributions?.[horizon] ?? 0) * 100);
    vals.push(residual * 100);
    vals.push(indexReturn * 100);
    return vals;
  }, [sortedSectors, horizon, residual, indexReturn]);

  const option = useMemo(() => {
    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: {
        top: 20,
        right: 45,
        bottom: 25,
        left: 170,
        containLabel: false,
      },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: "#0c1017",
        borderColor: "#1d2633",
        textStyle: { color: "#e8edf5", fontSize: 12 },
        formatter: (params: any) => {
          if (!params || params.length === 0) return "";
          const p = params[0];
          const val = p.value as number;
          const idx = p.dataIndex;
          if (idx < sortedSectors.length) {
            const s = sortedSectors[idx];
            const weight = (s.weight * 100).toFixed(1);
            const ret = ((s.returns[horizon] ?? 0) * 100).toFixed(2);
            return `
              <div class="font-sans">
                <div class="font-semibold text-xs border-b border-[#1d2633] pb-1 mb-1">${s.name}</div>
                <div class="text-[11px] text-[#8b98a5]">Contribution: <span class="font-mono text-[#e8edf5] font-semibold">${val >= 0 ? "+" : ""}${val.toFixed(2)}% (${(val * 100).toFixed(0)} bps)</span></div>
                <div class="text-[11px] text-[#8b98a5]">Sector Return: <span class="font-mono text-[#e8edf5]">${ret >= "0" ? "+" : ""}${ret}%</span></div>
                <div class="text-[11px] text-[#8b98a5]">Factsheet Weight: <span class="font-mono text-[#e8edf5]">${weight}%</span></div>
              </div>
            `;
          } else if (idx === sortedSectors.length) {
            return `
              <div class="font-sans">
                <div class="font-semibold text-xs border-b border-[#1d2633] pb-1 mb-1">Residual (Unmodeled / Drift)</div>
                <div class="text-[11px] text-[#8b98a5]">Residual: <span class="font-mono text-[#e8edf5] font-semibold">${val >= 0 ? "+" : ""}${val.toFixed(2)}% (${(val * 100).toFixed(0)} bps)</span></div>
                <div class="text-[10px] text-[#6f7c90] mt-1 max-w-[200px]">Accounts for intra-month weighting drift and non-sector components.</div>
              </div>
            `;
          } else {
            return `
              <div class="font-sans">
                <div class="font-semibold text-xs border-b border-[#1d2633] pb-1 mb-1">${benchmarkName} Total Return</div>
                <div class="text-[11px] text-[#8b98a5]">Index Move: <span class="font-mono text-[#e8edf5] font-semibold">${val >= 0 ? "+" : ""}${val.toFixed(2)}%</span></div>
              </div>
            `;
          }
        },
      },
      xAxis: {
        type: "value",
        axisLabel: {
          color: "#8b98a5",
          fontSize: 10,
          formatter: "{value}%",
        },
        splitLine: { lineStyle: { color: "#161d28" } },
      },
      yAxis: {
        type: "category",
        data: categories,
        inverse: true,
        axisLabel: {
          color: "#a9b4c4",
          fontSize: 11,
          formatter: (name: string) => {
            if (name.includes("Total")) return `{bold|${name}}`;
            if (name.includes("Residual")) return `{italic|${name}}`;
            return name;
          },
          rich: {
            bold: { fontWeight: "bold", color: "#e8edf5" },
            italic: { fontStyle: "italic", color: "#6f7c90" },
          },
        },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: "#1d2633" } },
      },
      series: [
        {
          name: "Contribution",
          type: "bar",
          data: values.map((val, idx) => {
            let color = val >= 0 ? "#3fb6a8" : "#e8765a";
            if (idx === sortedSectors.length) {
              color = "#6f7c90"; // Neutral gray for residual
            } else if (idx === sortedSectors.length + 1) {
              color = "#6ea8ff"; // Accent blue for total index move
            }
            return {
              value: Number(val.toFixed(3)),
              itemStyle: {
                color,
                borderRadius: [0, 2, 2, 0],
              },
            };
          }),
        },
      ],
    };
  }, [sortedSectors, categories, values, horizon, benchmarkName]);

  const height = Math.max(300, categories.length * 28 + 60);

  return (
    <div className="space-y-3">
      {/* Editorial Disclaimer Banner */}
      <div className="flex items-start space-x-2 text-[11px] text-[#8b98a5] bg-[#0c1017] border border-[#1d2633] rounded px-3 py-2">
        <Info className="w-3.5 h-3.5 text-[#6ea8ff] mt-0.5 flex-shrink-0" />
        <div>
          <span className="font-semibold text-[#e8edf5]">Approximate Contribution Model:</span>{" "}
          Contribution is estimated as <code className="text-[#6ea8ff] font-mono text-[10px]">w_s × r_s</code> using published index weights from{" "}
          <span className="text-[#e8edf5]">{weightsSource}</span> (as of {asOf}). The explicit residual represents intra-month weighting drift and unclassified components, guaranteeing exact mathematical reconciliation to the index return.
        </div>
      </div>

      <div style={{ height: `${height}px` }} className="w-full">
        <ReactECharts
          option={option}
          style={{ height: "100%", width: "100%" }}
          opts={{ renderer: "svg" }}
        />
      </div>
    </div>
  );
};
