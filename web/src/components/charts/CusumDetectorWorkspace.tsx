// web/src/components/charts/CusumDetectorWorkspace.tsx
"use client";

import React, { useMemo, useState, useEffect } from "react";
import ReactECharts from "echarts-for-react";
import { DivergenceItem } from "@/lib/data/schemas";
import { AlertTriangle, CheckCircle2, ShieldCheck, Activity, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { AuditData } from "../ui/AuditDrawer";

interface CusumDetectorWorkspaceProps {
  divergences: DivergenceItem[];
  onOpenAudit?: (data: AuditData) => void;
}

export const CusumDetectorWorkspace: React.FC<CusumDetectorWorkspaceProps> = ({
  divergences,
  onOpenAudit,
}) => {
  // Pairs that have cusum data
  const validPairs = useMemo(() => {
    return divergences.filter((d) => d.cusum && d.cusum.dates.length > 0);
  }, [divergences]);

  const [selectedPairId, setSelectedPairId] = useState<string>("");

  useEffect(() => {
    if (!selectedPairId && validPairs.length > 0) {
      setSelectedPairId(validPairs[0].id);
    }
  }, [validPairs, selectedPairId]);

  // Keep selected pair valid
  const currentPair = useMemo(() => {
    return validPairs.find((p) => p.id === selectedPairId) || validPairs[0];
  }, [validPairs, selectedPairId]);

  // Aggregate stats across all pairs
  const totalShiftsCount = useMemo(() => {
    return validPairs.reduce((acc, p) => acc + (p.cusum?.shifts.length || 0), 0);
  }, [validPairs]);

  const pairsWithShifts = useMemo(() => {
    return validPairs.filter((p) => (p.cusum?.shifts.length || 0) > 0);
  }, [validPairs]);

  const cusum = currentPair?.cusum;
  const latestPos = cusum && cusum.cusum_pos.length > 0 ? cusum.cusum_pos[cusum.cusum_pos.length - 1] : 0;
  const latestNeg = cusum && cusum.cusum_neg.length > 0 ? cusum.cusum_neg[cusum.cusum_neg.length - 1] : 0;
  const shifts = cusum?.shifts || [];
  const threshold = cusum?.threshold ?? 4.0;

  const chartOption = useMemo(() => {
    if (!cusum || cusum.dates.length === 0) return {};

    const formattedDates = cusum.dates.map((d) => {
      const parts = d.split("-");
      return parts.length >= 3 ? `${parts[1]}/${parts[2]}` : d;
    });

    // Only plot shifts that fall within the current 60-day window
    const inWindowShifts = shifts.filter((s) => cusum.dates.includes(s.date));
    const upShifts = inWindowShifts.filter((s) => s.direction.toLowerCase().startsWith("up"));
    const downShifts = inWindowShifts.filter((s) => s.direction.toLowerCase().startsWith("down"));

    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: {
        top: 35,
        right: 35,
        bottom: 30,
        left: 45,
        containLabel: true,
      },
      legend: {
        data: ["Upward CUSUM (S⁺)", "Downward CUSUM (S⁻)"],
        top: 4,
        right: 15,
        textStyle: { color: "#8a99ad", fontSize: 11, fontFamily: "monospace" },
        icon: "roundRect",
      },
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0c1017",
        borderColor: "#1d2633",
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return "";
          const idx = params[0].dataIndex;
          const rawDate = cusum.dates[idx] || params[0].axisValue;
          const pos = params.find((p: any) => p.seriesName?.includes("S⁺"))?.value ?? 0;
          const neg = params.find((p: any) => p.seriesName?.includes("S⁻"))?.value ?? 0;
          const shift = shifts.find((s) => s.date === rawDate);

          let html = `<div style="font-weight:600;margin-bottom:6px;color:#cbd5e1;border-bottom:1px solid #1d2633;padding-bottom:4px;">Date: ${rawDate}</div>`;
          html += `<div style="margin-bottom:2px;"><span style="color:#22c55e;">S⁺ (Upward Accumulation):</span> <b>${Number(pos).toFixed(2)}σ</b></div>`;
          html += `<div style="margin-bottom:2px;"><span style="color:#ef4444;">S⁻ (Downward Accumulation):</span> <b>${Number(neg).toFixed(2)}σ</b></div>`;
          html += `<div style="color:#64748b;font-size:10px;margin-top:4px;">Decision Limit h = ${threshold.toFixed(1)}σ (slack k = 0.5σ)</div>`;
          if (shift) {
            html += `<div style="color:#f59e0b;font-weight:bold;margin-top:6px;background:rgba(245,158,11,0.1);padding:3px 6px;border-radius:4px;">⚠️ Regime Shift Flagged: ${shift.direction.toUpperCase()} (${shift.value.toFixed(2)}σ)</div>`;
          }
          return html;
        },
      },
      xAxis: {
        type: "category",
        data: formattedDates,
        axisLine: { lineStyle: { color: "#2a3647" } },
        axisLabel: {
          color: "#6f7c90",
          fontSize: 10,
          interval: Math.max(1, Math.floor(cusum.dates.length / 8)),
        },
      },
      yAxis: {
        type: "value",
        min: 0,
        splitLine: { lineStyle: { color: "#1d2633", opacity: 0.6 } },
        axisLabel: {
          color: "#6f7c90",
          fontSize: 10,
          formatter: "{value}σ",
        },
      },
      series: [
        {
          name: "Upward CUSUM (S⁺)",
          type: "line",
          data: cusum.cusum_pos,
          lineStyle: { color: "#22c55e", width: 2 },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(34, 197, 94, 0.28)" },
                { offset: 1, color: "rgba(34, 197, 94, 0.0)" },
              ],
            },
          },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "#e8765a", type: "dashed", width: 1.5 },
            label: {
              show: true,
              position: "insideEndTop",
              formatter: `Decision Limit h = ${threshold.toFixed(1)}σ`,
              color: "#e8765a",
              fontSize: 10,
              fontFamily: "monospace",
            },
            data: [{ yAxis: threshold }],
          },
          markPoint: upShifts.length > 0 ? {
            symbol: "pin",
            symbolSize: 32,
            itemStyle: { color: "#f59e0b" },
            label: {
              show: true,
              formatter: "Shift",
              fontSize: 9,
              color: "#000",
              fontWeight: "bold",
            },
            data: upShifts.map((s) => {
              const idx = cusum.dates.indexOf(s.date);
              return {
                name: "Upward Shift",
                coord: [idx, s.value],
                value: s.value,
              };
            }),
          } : undefined,
          showSymbol: false,
        },
        {
          name: "Downward CUSUM (S⁻)",
          type: "line",
          data: cusum.cusum_neg,
          lineStyle: { color: "#ef4444", width: 2 },
          areaStyle: {
            color: {
              type: "linear",
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(239, 68, 68, 0.28)" },
                { offset: 1, color: "rgba(239, 68, 68, 0.0)" },
              ],
            },
          },
          markPoint: downShifts.length > 0 ? {
            symbol: "pin",
            symbolSize: 32,
            itemStyle: { color: "#ef4444" },
            label: {
              show: true,
              formatter: "Shift",
              fontSize: 9,
              color: "#fff",
              fontWeight: "bold",
            },
            data: downShifts.map((s) => {
              const idx = cusum.dates.indexOf(s.date);
              return {
                name: "Downward Shift",
                coord: [idx, s.value],
                value: s.value,
              };
            }),
          } : undefined,
          showSymbol: false,
        },
      ],
    };
  }, [cusum, shifts, threshold]);

  if (!currentPair || !cusum) {
    return (
      <div className="p-8 text-center rounded-lg bg-[var(--bg-1)] border border-[var(--line)] font-mono text-xs text-[var(--tx-2)]">
        Loading CUSUM Change-Point Detector data...
      </div>
    );
  }

  return (
    <div className="p-4 sm:p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[var(--line)]">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase tracking-widest text-[var(--accent)] font-semibold">
              C24 / Control Chart
            </span>
            <h3 className="text-base font-bold text-[var(--tx-0)] font-mono">
              CUSUM Change-Point Detector (Page&apos;s Two-Sided)
            </h3>
          </div>
          <p className="text-xs text-[var(--tx-2)] mt-0.5">
            Sequential cumulative sum on standardized momentum innovations (slack k = 0.5&sigma;, limit h = {threshold.toFixed(1)}&sigma;) to separate genuine structural breaks from transient noise.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono bg-[var(--bg-2)] border border-[var(--line)] text-[var(--tx-1)]">
            <Activity className="w-3.5 h-3.5 text-[var(--accent)]" />
            <span>{totalShiftsCount} Shifts Flagged Across {validPairs.length} Pairs</span>
          </span>

          {onOpenAudit && (
            <button
              type="button"
              onClick={() =>
                onOpenAudit({
                  metricName: `CUSUM Control Chart: ${currentPair.asset_a.toUpperCase()} vs ${currentPair.asset_b.toUpperCase()}`,
                  formula: "S_t^+ = \\max(0, S_{t-1}^+ + z_t - k), \\quad S_t^- = \\max(0, S_{t-1}^- - z_t - k)",
                  formulaPlain: "Two-sided cumulative sum tracking upward and downward drift beyond reference slack k = 0.5σ. Signals regime shift when S_t ≥ h = 4.0σ.",
                  parameters: {
                    slack_k: 0.5,
                    decision_interval_h: threshold,
                    horizon: "60 trading days",
                    innovation_type: "standardized momentum spread innovations",
                  },
                  inputs: [
                    { id: currentPair.asset_a, name: currentPair.asset_a.toUpperCase(), source: "Yahoo/FRED", unit: "index" },
                    { id: currentPair.asset_b, name: currentPair.asset_b.toUpperCase(), source: "Yahoo/FRED", unit: "index" },
                  ],
                  window: "60 business days",
                  currentValue: `S⁺=${latestPos.toFixed(2)}σ, S⁻=${latestNeg.toFixed(2)}σ`,
                  percentile: currentPair.percentile,
                  codeVersion: "v1.0.0",
                  specUrl: "/methodology",
                })
              }
              className="p-1.5 rounded hover:bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--accent)] transition-colors"
              title="Audit CUSUM mathematical specification"
            >
              <ShieldCheck className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Interactive Pair Selector Pills */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-thin">
        <span className="text-[11px] font-mono text-[var(--tx-2)] uppercase mr-1 whitespace-nowrap">
          Pair:
        </span>
        {validPairs.map((pair) => {
          const isSelected = pair.id === currentPair.id;
          const shiftCount = pair.cusum?.shifts.length || 0;
          return (
            <button
              key={pair.id}
              type="button"
              onClick={() => setSelectedPairId(pair.id)}
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-mono font-medium transition-all whitespace-nowrap ${
                isSelected
                  ? "bg-[var(--accent)] text-white shadow-sm"
                  : "bg-[var(--bg-2)] text-[var(--tx-1)] hover:bg-[var(--bg-0)] hover:text-[var(--tx-0)] border border-[var(--line)]"
              }`}
            >
              <span>{pair.asset_a.toUpperCase()} / {pair.asset_b.toUpperCase()}</span>
              {shiftCount > 0 && (
                <span
                  className={`w-2 h-2 rounded-full ${
                    isSelected ? "bg-amber-300" : "bg-[var(--warn)]"
                  }`}
                  title={`${shiftCount} historical regime shift(s)`}
                />
              )}
            </button>
          );
        })}
      </div>

      {/* Active Pair Summary & KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-3 rounded-md bg-[var(--bg-0)] border border-[var(--line)]">
          <div className="text-[10px] font-mono text-[var(--tx-2)] uppercase tracking-wider flex items-center gap-1">
            <ArrowUpRight className="w-3 h-3 text-[var(--pos)]" /> Upper Drift S⁺
          </div>
          <div className={`text-lg font-bold font-mono mt-0.5 ${latestPos >= threshold ? "text-[var(--warn)]" : "text-[var(--pos)]"}`}>
            {latestPos.toFixed(2)}&sigma;
          </div>
          <div className="text-[10px] text-[var(--tx-2)] mt-0.5">
            {latestPos >= threshold ? "Threshold breached" : "Below limit"}
          </div>
        </div>

        <div className="p-3 rounded-md bg-[var(--bg-0)] border border-[var(--line)]">
          <div className="text-[10px] font-mono text-[var(--tx-2)] uppercase tracking-wider flex items-center gap-1">
            <ArrowDownRight className="w-3 h-3 text-[var(--neg)]" /> Lower Drift S⁻
          </div>
          <div className={`text-lg font-bold font-mono mt-0.5 ${latestNeg >= threshold ? "text-[var(--warn)]" : "text-[var(--neg)]"}`}>
            {latestNeg.toFixed(2)}&sigma;
          </div>
          <div className="text-[10px] text-[var(--tx-2)] mt-0.5">
            {latestNeg >= threshold ? "Threshold breached" : "Below limit"}
          </div>
        </div>

        <div className="p-3 rounded-md bg-[var(--bg-0)] border border-[var(--line)]">
          <div className="text-[10px] font-mono text-[var(--tx-2)] uppercase tracking-wider">
            Decision Limit h
          </div>
          <div className="text-lg font-bold font-mono mt-0.5 text-[var(--tx-0)]">
            {threshold.toFixed(1)}&sigma;
          </div>
          <div className="text-[10px] text-[var(--tx-2)] mt-0.5 font-mono">
            Slack k = 0.50&sigma;
          </div>
        </div>

        <div className="p-3 rounded-md bg-[var(--bg-0)] border border-[var(--line)]">
          <div className="text-[10px] font-mono text-[var(--tx-2)] uppercase tracking-wider">
            Process Status
          </div>
          <div className="flex items-center gap-1.5 mt-1">
            {shifts.length > 0 ? (
              <>
                <AlertTriangle className="w-4 h-4 text-[var(--warn)] shrink-0" />
                <span className="text-xs font-bold font-mono text-[var(--warn)]">
                  {shifts.length} Shift{shifts.length > 1 ? "s" : ""} Flagged
                </span>
              </>
            ) : (
              <>
                <CheckCircle2 className="w-4 h-4 text-[var(--pos)] shrink-0" />
                <span className="text-xs font-bold font-mono text-[var(--pos)]">
                  In Control
                </span>
              </>
            )}
          </div>
          <div className="text-[10px] text-[var(--tx-2)] mt-0.5 font-mono">
            {shifts.length > 0 ? `Latest: ${shifts[shifts.length - 1].date}` : "No breaks detected"}
          </div>
        </div>
      </div>

      {/* Main ECharts Canvas with explicit height */}
      <div style={{ width: "100%", height: 280 }} className="w-full pt-1">
        <ReactECharts
          option={chartOption}
          style={{ height: "280px", width: "100%" }}
          opts={{ renderer: "canvas" }}
          notMerge={true}
          lazyUpdate={true}
        />
      </div>

      {/* Flagged Shifts History Feed across all pairs */}
      {pairsWithShifts.length > 0 && (
        <div className="pt-3 border-t border-[var(--line)]">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-mono font-semibold text-[var(--tx-0)] uppercase tracking-wider">
              Cross-Asset Structural Shift Log
            </span>
            <span className="text-[10px] font-mono text-[var(--tx-2)]">
              {pairsWithShifts.length} of {validPairs.length} pairs exhibit persistent innovation drift
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
            {pairsWithShifts.map((p) => {
              const latestShift = p.cusum?.shifts[p.cusum.shifts.length - 1];
              const isCurrent = p.id === currentPair.id;
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => setSelectedPairId(p.id)}
                  className={`text-left p-2.5 rounded border transition-colors ${
                    isCurrent
                      ? "bg-[var(--accent)]/10 border-[var(--accent)]"
                      : "bg-[var(--bg-0)] border-[var(--line)] hover:border-[var(--tx-2)]"
                  }`}
                >
                  <div className="flex items-center justify-between text-xs font-mono font-semibold text-[var(--tx-0)]">
                    <span>{p.asset_a.toUpperCase()} / {p.asset_b.toUpperCase()}</span>
                    <span className="text-[10px] text-[var(--warn)] font-normal flex items-center gap-0.5">
                      <AlertTriangle className="w-2.5 h-2.5" />
                      {p.cusum?.shifts.length} shift(s)
                    </span>
                  </div>
                  {latestShift && (
                    <div className="text-[10px] font-mono text-[var(--tx-2)] mt-1">
                      {latestShift.date}: <span className="text-[var(--tx-1)] font-semibold">{latestShift.direction.toUpperCase()}</span> ({latestShift.value.toFixed(1)}&sigma;)
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
