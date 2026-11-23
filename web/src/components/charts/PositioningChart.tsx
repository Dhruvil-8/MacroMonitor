"use client";

import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { CotContract, PositioningEnvelope } from "@/lib/data/schemas";
import { downloadCsv, fmtNum, fmtPct, fmtPercentile } from "@/lib/formatters";
import { useRadarStore } from "@/store/useRadarStore";
import { ChartPanel } from "@/components/insight/ChartPanel";
import { EChartBox } from "@/components/charts/EChartBox";

interface Props {
  data: PositioningEnvelope;
  selectedContractId?: string;
  onSelectContractId?: (id: string) => void;
}

export function PositioningChart({ data, selectedContractId, onSelectContractId }: Props) {
  const theme = useRadarStore((s) => s.theme);
  const isDark = theme === "dark";
  const pos = (data as any).data || data;
  const contracts: CotContract[] = pos.contracts || [];
  const [internalSelectedId, setInternalSelectedId] = useState<string>(contracts[0]?.id ?? "ES");

  const effectiveSelectedId = selectedContractId ?? internalSelectedId;
  const handleSelectId = (id: string) => {
    if (onSelectContractId) {
      onSelectContractId(id);
    } else {
      setInternalSelectedId(id);
    }
  };

  const active = useMemo(
    () => contracts.find((c: CotContract) => c.id === effectiveSelectedId) ?? contracts[0],
    [contracts, effectiveSelectedId]
  );

  const option = useMemo<EChartsOption>(() => {
    if (!active) return {};
    const textCol = "#8c9bb0";
    const gridLine = "#1d2633";
    const dates = active.series.map((s) => s.date);

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0c1017",
        borderColor: gridLine,
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
      },
      legend: { top: 0, textStyle: { color: textCol, fontSize: 11 } },
      grid: { left: 54, right: 50, top: 34, bottom: 32 },
      xAxis: {
        type: "category",
        data: dates,
        axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace" },
        axisLine: { lineStyle: { color: gridLine } },
      },
      yAxis: [
        {
          type: "value",
          name: "Net % of OI",
          nameTextStyle: { color: textCol, fontSize: 10 },
          splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
          axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace", formatter: "{value}%" },
        },
        {
          type: "value",
          name: "Percentile",
          min: 0,
          max: 100,
          nameTextStyle: { color: textCol, fontSize: 10 },
          splitLine: { show: false },
          axisLabel: { color: textCol, fontSize: 10, fontFamily: "monospace", formatter: "{value}%" },
        },
      ],
      series: [
        {
          name: "Non-Comm Net % of OI",
          type: "bar",
          yAxisIndex: 0,
          data: active.series.map((s) => ({
            value: s.net_pct_oi !== null && s.net_pct_oi !== undefined ? Number((s.net_pct_oi * 100).toFixed(2)) : null,
            itemStyle: {
              color:
                (s.net_pct_oi ?? 0) >= 0
                  ? "rgba(45, 212, 191, 0.65)"
                  : "rgba(244, 63, 94, 0.65)",
            },
          })),
        },
        {
          name: "Expanding Percentile",
          type: "line",
          yAxisIndex: 1,
          showSymbol: false,
          lineStyle: { width: 1.8, color: "#38bdf8" },
          data: active.series.map((s) => s.percentile),
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: textCol, type: "dotted" },
            data: [{ yAxis: 10 }, { yAxis: 50 }, { yAxis: 90 }],
          },
        },
      ],
    };
  }, [active]);

  if (!active) return null;

  const insights = [
    `${active.name} (${active.id}): non-commercial net speculative positioning stood at ${active.net_speculative?.toLocaleString() ?? "—"} contracts (${fmtPct(active.net_pct_oi, 2, true)} of open interest as of ${active.latest_date ?? "—"}).`,
    `Expanding historical rank is at the ${fmtPercentile(active.percentile)} (Z-score ${fmtNum(active.z_score, 2, true)}, 1W change ${fmtPct(active.chg_1w_pct_oi, 2, true)} of OI).`,
    `Extreme percentile readings reflect crowded historical non-commercial positioning rather than directional forecasts.`,
  ];

  return (
    <ChartPanel
      title="CFTC Commitments of Traders Positioning (Futures Only)"
      subtitle="Weekly non-commercial speculative net positioning as % of open interest and expanding historical percentile across 7 major futures contracts"
      infoText="Data sourced directly from CFTC Socrata Legacy Futures-Only reports (publicreporting.cftc.gov). Net Speculative = Non-Commercial Long - Non-Commercial Short, divided by total Open Interest. Percentiles and Z-scores use strictly prior weekly reports (min 52 weeks)."
      insights={insights}
      asOf={active.latest_date ?? pos.as_of}
      windowLabel="Weekly · Trailing 3Y / Expanding"
      sourcesLabel="CFTC Public Reporting API (Legacy COT)"
      n={active.n}
      nEff={active.n_eff}
      headerRightExtra={
        <select
          aria-label="Select CFTC futures contract"
          value={active.id}
          onChange={(e) => handleSelectId(e.target.value)}
          className="px-2.5 py-1.5 text-xs rounded-[6px] border border-[#1d2633] bg-[#07090d] text-[#e8edf5] cursor-pointer"
        >
          {contracts.map((c) => (
            <option key={c.id} value={c.id}>
              {c.id} — {c.name}
            </option>
          ))}
        </select>
      }
      getTableData={() => ({
        headers: ["Date", "Open Interest", "NonComm Long", "NonComm Short", "Net Speculative", "Net % OI", "Percentile", "Z-Score"],
        rows: active.series.map((r) => [
          r.date,
          r.open_interest?.toLocaleString() ?? "—",
          r.noncomm_long?.toLocaleString() ?? "—",
          r.noncomm_short?.toLocaleString() ?? "—",
          r.net_speculative?.toLocaleString() ?? "—",
          fmtPct(r.net_pct_oi, 2, true),
          fmtPercentile(r.percentile),
          fmtNum(r.z_score, 2, true),
        ]),
      })}
      audit={{
        title: `CFTC COT Positioning: ${active.name}`,
        formula: "NetPctOI_t = (NonCommLong_t - NonCommShort_t) / OI_t; pct_t = expanding_pct(NetPctOI, min_obs=52)",
        parameters: { cftc_contract_code: active.cftc_code, min_expanding_weeks: 52, report_type: "Legacy Futures Only" },
        inputs: [{ id: `CFTC_${active.id}`, lastUpdate: active.latest_date }],
        percentileSummary: `Latest Net % OI ${fmtPct(active.net_pct_oi, 2, true)} ranks at ${fmtPercentile(active.percentile)} (Z=${fmtNum(active.z_score, 2, true)}).`,
        n: active.n,
        nEff: active.n_eff,
        asOf: active.latest_date ?? pos.as_of,
        codeVersion: pos.code_version,
        paramsHash: pos.params_hash,
        specAnchor: "positioning",
      }}
      onDownloadCsv={() =>
        downloadCsv(
          `cftc_cot_${active.id}_${active.latest_date ?? pos.as_of}.csv`,
          ["date", "open_interest", "noncomm_long", "noncomm_short", "net_speculative", "net_pct_oi", "percentile", "z_score"],
          active.series.map((r) => [
            r.date,
            r.open_interest,
            r.noncomm_long,
            r.noncomm_short,
            r.net_speculative,
            r.net_pct_oi,
            r.percentile,
            r.z_score,
          ])
        )
      }
    >
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        <div className="xl:col-span-5 overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-[#1d2633] text-[#6f7c90] text-left">
                <th className="py-2 pr-2 font-medium">Contract</th>
                <th className="py-2 px-2 font-medium text-right">Net % OI</th>
                <th className="py-2 px-2 font-medium text-right">1W Chg</th>
                <th className="py-2 px-2 font-medium text-right">Pct</th>
                <th className="py-2 pl-2 font-medium text-right">Z-Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1d2633] font-mono tabular-nums">
              {contracts.map((c) => {
                const isSel = c.id === active.id;
                const isExtreme = (c.percentile ?? 50) >= 90 || (c.percentile ?? 50) <= 10;
                return (
                  <tr
                    key={c.id}
                    onClick={() => handleSelectId(c.id)}
                    className={`cursor-pointer transition-colors ${isSel
                      ? "bg-[#121823]"
                      : "hover:bg-[#121823]/50"
                      }`}
                  >
                    <td className="py-2 pr-2 font-sans">
                      <span className="font-mono font-semibold text-[#38bdf8]">{c.id}</span>
                      <span className="text-[#8c9bb0] ml-1.5">{c.name}</span>
                    </td>
                    <td className="py-2 px-2 text-right font-semibold">{fmtPct(c.net_pct_oi, 1, true)}</td>
                    <td className="py-2 px-2 text-right text-[#6f7c90]">{fmtPct(c.chg_1w_pct_oi, 1, true)}</td>
                    <td
                      className={`py-2 px-2 text-right font-semibold ${isExtreme ? "text-[#f59e0b]" : "text-[#e8edf5]"
                        }`}
                    >
                      {isExtreme ? "▲ " : ""}
                      {fmtPercentile(c.percentile)}
                    </td>
                    <td className="py-2 pl-2 text-right">{fmtNum(c.z_score, 2, true)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="mt-2 text-[11px] font-mono text-[#6f7c90]">
            Click any contract row to switch time-series view. Report date: {active.latest_date ?? "—"} · OI:{" "}
            {active.open_interest?.toLocaleString() ?? "—"}.
          </div>
        </div>

        <div className="xl:col-span-7">
          <EChartBox
            option={option}
            height={300}
            ariaLabel={`Weekly CFTC non-commercial net positioning as % of open interest and expanding percentile for ${active.name}`}
          />
        </div>
      </div>
    </ChartPanel>
  );
}
