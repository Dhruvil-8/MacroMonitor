"use client";

import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { StressEnvelope, StressEpisode, StressSeriesPoint } from "@/lib/data/schemas";
import { downloadCsv } from "@/lib/formatters";
import { useRadarStore } from "@/store/useRadarStore";
import { ChartPanel } from "@/components/insight/ChartPanel";
import { EChartBox } from "@/components/charts/EChartBox";

const COMP_LABELS: Record<string, string> = {
  equity: "Equity (VIX & DD)",
  rates: "Rates (10Y Vol)",
  fx: "FX Basket Vol",
  commodity: "Commodity Vol",
  correlation: "Cross-Asset Corr",
  credit: "Credit Proxy (HYG)",
};

interface Props {
  data: StressEnvelope;
}

export function StressStackChart({ data }: Props) {
  const theme = useRadarStore((s) => s.theme);
  const [showEpisodes, setShowEpisodes] = useState(true);

  const st = (data as any).data || data;
  const series: StressSeriesPoint[] = st.series || [];
  const latest = st.latest;
  const episodes: StressEpisode[] = st.episodes || [];
  const componentKeys: string[] = st.component_keys || Object.keys(COMP_LABELS);

  const option = useMemo<EChartsOption>(() => {
    const textCol = "#8c9bb0";
    const gridLine = "#1d2633";
    const dates = series.map((s) => s.date);
    const comp = series.map((s) => s.composite);

    const eq = series.map((s) => +(Number(s.components?.equity ?? (s as any).equity ?? 0) / 6).toFixed(2));
    const rt = series.map((s) => +(Number(s.components?.rates ?? (s as any).rates ?? 0) / 6).toFixed(2));
    const fx = series.map((s) => +(Number(s.components?.fx ?? (s as any).fx ?? 0) / 6).toFixed(2));
    const cm = series.map((s) => +(Number(s.components?.commodity ?? (s as any).commodity ?? 0) / 6).toFixed(2));
    const co = series.map((s) => +(Number(s.components?.correlation ?? (s as any).correlation ?? 0) / 6).toFixed(2));
    const cr = series.map((s) => +(Number(s.components?.credit ?? (s as any).credit ?? 0) / 6).toFixed(2));

    const markAreas: [{ name: string; xAxis: string; itemStyle: { color: string } }, { xAxis: string }][] = [];
    if (showEpisodes) {
      for (const ep of episodes) {
        const startD = dates.find((d) => d >= ep.start);
        const endD = [...dates].reverse().find((d) => d <= ep.end);
        if (startD && endD) {
          markAreas.push([
            { name: ep.label, xAxis: startD, itemStyle: { color: "rgba(245, 158, 11, 0.12)" } },
            { xAxis: endD },
          ]);
        }
      }
    }

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0c1017",
        borderColor: gridLine,
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
      },
      legend: {
        top: 0,
        textStyle: { color: textCol, fontSize: 11 },
      },
      grid: { left: 48, right: 20, top: 38, bottom: 32 },
      xAxis: {
        type: "category",
        data: dates,
        axisLabel: { color: textCol, fontFamily: "monospace", fontSize: 10 },
        axisLine: { lineStyle: { color: gridLine } },
      },
      yAxis: {
        type: "value",
        name: "Stress Percentile (0-100)",
        nameTextStyle: { color: textCol, fontSize: 10 },
        min: 0,
        max: 100,
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: { color: textCol, fontFamily: "monospace", fontSize: 11 },
      },
      series: [
        { name: "Equity", type: "line", stack: "contrib", areaStyle: { opacity: 0.35 }, lineStyle: { width: 0 }, symbol: "none", data: eq, itemStyle: { color: "#38BDF8" } },
        { name: "Rates", type: "line", stack: "contrib", areaStyle: { opacity: 0.35 }, lineStyle: { width: 0 }, symbol: "none", data: rt, itemStyle: { color: "#818CF8" } },
        { name: "FX", type: "line", stack: "contrib", areaStyle: { opacity: 0.35 }, lineStyle: { width: 0 }, symbol: "none", data: fx, itemStyle: { color: "#2DD4BF" } },
        { name: "Commodity", type: "line", stack: "contrib", areaStyle: { opacity: 0.35 }, lineStyle: { width: 0 }, symbol: "none", data: cm, itemStyle: { color: "#FACC15" } },
        { name: "Correlation", type: "line", stack: "contrib", areaStyle: { opacity: 0.35 }, lineStyle: { width: 0 }, symbol: "none", data: co, itemStyle: { color: "#F472B6" } },
        { name: "Credit", type: "line", stack: "contrib", areaStyle: { opacity: 0.35 }, lineStyle: { width: 0 }, symbol: "none", data: cr, itemStyle: { color: "#F97316" } },
        {
          name: "Composite",
          type: "line",
          data: comp,
          symbol: "none",
          lineStyle: { width: 2.2, color: "#F8FAFC" },
          itemStyle: { color: "#F8FAFC" },
          markArea: {
            silent: true,
            label: { show: false },
            data: markAreas,
          },
        },
      ],
    };
  }, [series, episodes, showEpisodes]);

  const insights = useMemo(() => {
    if (!latest) return ["Stress composite unavailable."];
    const compObj = latest.components || {};
    const entries = Object.entries(compObj).filter(([, v]) => v !== null && v !== undefined) as [string, number][];
    const highest = [...entries].sort((a, b) => b[1] - a[1])[0];
    const lowest = [...entries].sort((a, b) => a[1] - b[1])[0];
    const lines = [
      `Composite cross-asset stress sits at ${(latest.composite ?? 50).toFixed(1)}/100 (${latest.tercile ?? "—"} tercile).`,
    ];
    if (highest && lowest) {
      lines.push(
        `Highest sub-component is ${COMP_LABELS[highest[0]] ?? highest[0]} (${highest[1].toFixed(1)}th pct); lowest is ${COMP_LABELS[lowest[0]] ?? lowest[0]} (${lowest[1].toFixed(1)}th pct).`
      );
    }
    if (latest.missing_components && latest.missing_components.length > 0) {
      lines.push(`Missing components omitted without imputation: ${latest.missing_components.join(", ")}.`);
    }
    return lines;
  }, [latest]);

  return (
    <ChartPanel
      title="Multi-Asset Financial Stress Composite Breakdown"
      subtitle="Equal-weighted composite of 6 expanding-percentile stress components with historical episode shading"
      infoText="Shaded bands highlight pre-registered macro stress episodes (2015 China, 2018 Q4, 2020 Pandemic, 2022 Rates, 2024 Carry Unwind). Stacked bands represent each component's equal-weight contribution to the composite line. Missing components are never imputed."
      insights={insights}
      asOf={st.as_of}
      windowLabel="20D Vol / 1Y DD / 60D Corr"
      sourcesLabel="VIX, SPX, US10Y, FX, WTI, GOLD, COPPER, HYG"
      n={series.length}
      nEff={Math.max(1, Math.round(series.length / 6))}
      headerRightExtra={
        <button
          type="button"
          onClick={() => setShowEpisodes((v) => !v)}
          aria-pressed={showEpisodes}
          className={`px-2.5 py-1.5 text-xs font-medium rounded-[6px] border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
            showEpisodes
              ? "border-[#38bdf8] bg-[#121823] text-[#e8edf5]"
              : "border-[#1d2633] text-[#8c9bb0] hover:bg-[#121823]"
          }`}
        >
          {showEpisodes ? "Episodes: On" : "Episodes: Off"}
        </button>
      }
      getTableData={() => ({
        headers: ["Date", "Composite", "Tercile", "Equity", "Rates", "FX", "Commodity", "Correlation", "Credit"],
        rows: series.map((r) => [
          r.date,
          r.composite.toFixed(1),
          r.composite >= 66.7 ? "High" : r.composite >= 33.3 ? "Moderate" : "Low",
          r.components.equity !== null && r.components.equity !== undefined ? r.components.equity.toFixed(1) : "—",
          r.components.rates !== null && r.components.rates !== undefined ? r.components.rates.toFixed(1) : "—",
          r.components.fx !== null && r.components.fx !== undefined ? r.components.fx.toFixed(1) : "—",
          r.components.commodity !== null && r.components.commodity !== undefined ? r.components.commodity.toFixed(1) : "—",
          r.components.correlation !== null && r.components.correlation !== undefined ? r.components.correlation.toFixed(1) : "—",
          r.components.credit !== null && r.components.credit !== undefined ? r.components.credit.toFixed(1) : "—",
        ]),
      })}
      audit={{
        title: "Multi-Asset Stress Stack",
        formula: "composite_t = mean(pct_equity, pct_rates, pct_fx, pct_commodity, pct_corr, pct_credit); missing components dropped",
        parameters: { tercile_low: 33.33, tercile_high: 66.67, min_expanding_obs: 750 },
        inputs: ["VIX", "SPX", "US10Y", "DXY", "WTI", "GOLD", "HYG"].map((id) => ({
          id,
          lastUpdate: st.as_of,
        })),
        percentileSummary: `Latest composite ${latest?.composite.toFixed(1) ?? "—"}/100 (${latest?.tercile ?? "—"} tercile).`,
        n: series.length,
        nEff: Math.max(1, Math.round(series.length / 6)),
        asOf: st.as_of,
        codeVersion: st.code_version,
        paramsHash: st.params_hash,
        specAnchor: "stress_dashboard",
      }}
      onDownloadCsv={() =>
        downloadCsv(
          `stress_composite_${st.as_of}.csv`,
          ["date", "composite", "tercile", "equity", "rates", "fx", "commodity", "correlation", "credit"],
          series.map((r) => [
            r.date,
            r.composite,
            r.composite >= 66.7 ? "High" : r.composite >= 33.3 ? "Moderate" : "Low",
            r.components.equity,
            r.components.rates,
            r.components.fx,
            r.components.commodity,
            r.components.correlation,
            r.components.credit,
          ])
        )
      }
    >
      <div className="flex flex-col gap-4">
        <EChartBox
          option={option}
          height={320}
          ariaLabel="Stacked area and line chart of the multi-asset stress composite and its six components"
        />
        {latest && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 pt-1">
            {componentKeys.map((key) => {
              const val = latest.components[key];
              const isHigh = (val ?? 0) >= 66.7;
              const isLow = (val ?? 100) < 33.3;
              return (
                <div
                  key={key}
                  className="rounded-[6px] border border-[#1d2633] bg-[#07090d] px-3 py-2"
                >
                  <div className="text-[11px] text-[#6f7c90] truncate">{COMP_LABELS[key] ?? key}</div>
                  <div className="flex items-baseline justify-between mt-1 font-mono tabular-nums">
                    <span className="text-sm font-semibold text-[#e8edf5]">
                      {val !== null && val !== undefined ? val.toFixed(1) : "N/A"}
                    </span>
                    <span
                      className={`text-[10px] font-medium ${
                        isHigh
                          ? "text-[#f59e0b]"
                          : isLow
                          ? "text-[#38bdf8]"
                          : "text-[#6f7c90]"
                      }`}
                    >
                      {val === null || val === undefined ? "Dropped" : isHigh ? "▲ High" : isLow ? "▼ Low" : "· Mid"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </ChartPanel>
  );
}
