"use client";

import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { AnaloguePresetDetailData } from "@/lib/data/schemas";
import { EChartBox } from "@/components/charts/EChartBox";

interface FanChartProps {
  fans: AnaloguePresetDetailData["outcome_fans"];
  selectedAsset: string;
  height?: number;
}

export function FanChart({ fans, selectedAsset, height = 340 }: FanChartProps) {
  const rows = useMemo(
    () =>
      fans
        .filter((f) => f.asset === selectedAsset)
        .sort((a, b) => a.horizon_days - b.horizon_days),
    [fans, selectedAsset]
  );

  const option = useMemo<EChartsOption>(() => {
    const textCol = "#8c9bb0";
    const gridCol = "#1d2633";
    const accent = "#38bdf8";
    const amber = "#f59e0b";
    const unit = rows[0]?.unit ?? "%";

    const horizons = ["Today", ...rows.map((r) => `${r.horizon} (${r.horizon_days}D)`)];
    const p10 = [0, ...rows.map((r) => r.cond_p10)];
    const p25 = [0, ...rows.map((r) => r.cond_p25)];
    const p50 = [0, ...rows.map((r) => r.cond_p50)];
    const p75 = [0, ...rows.map((r) => r.cond_p75)];
    const p90 = [0, ...rows.map((r) => r.cond_p90)];
    const baseMed = [0, ...rows.map((r) => r.base_p50)];

    // Stackable band series
    const bandOuterSpan = p90.map((v, i) => Number((v - p10[i]).toFixed(3)));
    const bandInnerSpan = p75.map((v, i) => Number((v - p25[i]).toFixed(3)));

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0c1017",
        borderColor: gridCol,
        textStyle: { color: "#e8edf5", fontSize: 12 },
        formatter: (params: unknown) => {
          const arr = params as { dataIndex: number; axisValue: string }[];
          const idx = arr?.[0]?.dataIndex ?? 0;
          if (idx === 0) return `<strong>Baseline Origin (t=0)</strong>`;
          const r = rows[idx - 1];
          if (!r) return "";
          return `<div style="font-family:monospace;font-size:11px;">
            <strong>${r.asset} · ${r.horizon} Horizon</strong> (N=${r.n_matches} matches vs ${r.n_base} base)<br/>
            p10 / p25 / p50 / p75 / p90: ${r.cond_p10} / ${r.cond_p25} / <b>${r.cond_p50}</b> / ${r.cond_p75} / ${r.cond_p90}${r.unit}<br/>
            Unconditional Base Median: ${r.base_p50}${r.unit}<br/>
            Median Diff (90% Boot CI): ${r.diff_median >= 0 ? "+" : ""}${r.diff_median}${r.unit} [${r.diff_ci90_low}, ${r.diff_ci90_high}]
          </div>`;
        },
      },
      legend: {
        top: 4,
        textStyle: { color: textCol, fontSize: 11 },
        data: ["Analogue Median (p50)", "Unconditional Base Median", "p25–p75 IQR", "p10–p90 Band"],
      },
      grid: { left: 52, right: 24, top: 44, bottom: 32 },
      xAxis: {
        type: "category",
        data: horizons,
        axisLabel: { color: textCol, fontSize: 11 },
        axisLine: { lineStyle: { color: gridCol } },
      },
      yAxis: {
        type: "value",
        name: `Forward Change (${unit})`,
        nameTextStyle: { color: textCol, fontSize: 11 },
        axisLabel: { color: textCol, fontSize: 11, formatter: `{value}${unit}` },
        splitLine: { lineStyle: { color: gridCol, type: "dashed" } },
      },
      series: [
        { name: "_p10_base", type: "line", stack: "outer", symbol: "none", lineStyle: { opacity: 0 }, data: p10, silent: true },
        {
          name: "p10–p90 Band",
          type: "line",
          stack: "outer",
          symbol: "none",
          lineStyle: { opacity: 0 },
          areaStyle: { color: accent, opacity: 0.14 },
          data: bandOuterSpan,
        },
        { name: "_p25_base", type: "line", stack: "inner", symbol: "none", lineStyle: { opacity: 0 }, data: p25, silent: true },
        {
          name: "p25–p75 IQR",
          type: "line",
          stack: "inner",
          symbol: "none",
          lineStyle: { opacity: 0 },
          areaStyle: { color: accent, opacity: 0.25 },
          data: bandInnerSpan,
        },
        {
          name: "Analogue Median (p50)",
          type: "line",
          symbol: "circle",
          symbolSize: 7,
          lineStyle: { color: accent, width: 2.5 },
          itemStyle: { color: accent },
          data: p50,
          markLine: { silent: true, symbol: "none", lineStyle: { color: gridCol }, data: [{ yAxis: 0 }] },
        },
        {
          name: "Unconditional Base Median",
          type: "line",
          symbol: "diamond",
          symbolSize: 7,
          lineStyle: { color: amber, width: 2, type: "dashed" },
          itemStyle: { color: amber },
          data: baseMed,
        },
      ],
    };
  }, [rows]);

  return <EChartBox option={option} height={height} ariaLabel={`Conditional Analogue Outcome Fan for ${selectedAsset}`} />;
}
