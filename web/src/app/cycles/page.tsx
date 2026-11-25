// web/src/app/cycles/page.tsx
"use client";

import React, { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { ChartCard } from "@/components/charts/ChartCard";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { HealthDot } from "@/components/ui/HealthDot";
import { loadCycles } from "@/lib/data/client";
import { CyclesEnvelope } from "@/lib/data/schemas";

const QUADRANT_BADGES: Record<string, { color: string; summary: string }> = {
  Recovery: {
    color: "text-emerald-400 border-emerald-500/40 bg-emerald-950/40",
    summary: "Growth accelerating while inflation cools — historically favorable for Equities & Credit.",
  },
  Overheat: {
    color: "text-amber-400 border-amber-500/40 bg-amber-950/40",
    summary: "Both growth and inflation impulses rising — historically strongest for Commodities & Cyclicals.",
  },
  Stagflation: {
    color: "text-rose-400 border-rose-500/40 bg-rose-950/40",
    summary: "Growth slowing while inflation remains elevated — historically defensive for Gold & Cash.",
  },
  Reflation: {
    color: "text-sky-400 border-sky-500/40 bg-sky-950/40",
    summary: "Growth & inflation cooling together — historically favorable for Sovereign Bonds & Duration.",
  },
};

export default function CyclesPage() {
  const [env, setEnv] = useState<CyclesEnvelope | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadCycles()
      .then((res) => {
        if (res.status === "error") setError(res.error);
        else setEnv(res.data);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const data = env?.data;

  const spiralOption = useMemo(() => {
    if (!data || data.trail.length === 0) return {};
    const coords = data.trail.map((pt) => [pt.inflation_z, pt.growth_z]);
    const latest = data.trail[data.trail.length - 1];

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "item",
        backgroundColor: "rgba(10, 14, 20, 0.95)",
        borderColor: "rgba(255,255,255,0.12)",
        textStyle: { color: "#e6edf3", fontSize: 11, fontFamily: "monospace" },
        formatter: (p: any) => {
          const idx = p.dataIndex;
          const pt = data.trail[idx];
          if (!pt) return "";
          return `<div><b>${pt.date} (${pt.quadrant})</b></div>
                  <div>Growth Z: ${pt.growth_z >= 0 ? "+" : ""}${pt.growth_z}σ</div>
                  <div>Inflation Z: ${pt.inflation_z >= 0 ? "+" : ""}${pt.inflation_z}σ</div>`;
        },
      },
      grid: { top: 35, right: 35, bottom: 45, left: 45, containLabel: true },
      xAxis: {
        type: "value",
        name: "Inflation Impulse (σ)",
        nameLocation: "middle",
        nameGap: 28,
        min: -3,
        max: 3,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.2)" } },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "monospace" },
      },
      yAxis: {
        type: "value",
        name: "Growth Impulse (σ)",
        min: -3,
        max: 3,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.2)" } },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "monospace" },
      },
      series: [
        {
          name: "24M Regime Trail",
          type: "line",
          data: coords,
          smooth: 0.2,
          symbol: "circle",
          symbolSize: (val: any, params: any) => (params.dataIndex === coords.length - 1 ? 12 : 6),
          lineStyle: { color: "#58a6ff", width: 2 },
          itemStyle: {
            color: (params: any) => (params.dataIndex === coords.length - 1 ? "#f0883e" : "#58a6ff"),
          },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "rgba(255,255,255,0.3)", type: "dashed", width: 1 },
            data: [{ xAxis: 0 }, { yAxis: 0 }],
          },
          markArea: {
            silent: true,
            itemStyle: { color: "transparent" },
            label: { color: "rgba(255,255,255,0.35)", fontSize: 11, fontFamily: "monospace", fontWeight: "bold" },
            data: [
              [{ name: "RECOVERY\n(Growth ↑ · Infl ↓)", xAxis: -2.9, yAxis: 2.6 }, { xAxis: -0.2, yAxis: 2.6 }],
              [{ name: "OVERHEAT\n(Growth ↑ · Infl ↑)", xAxis: 0.2, yAxis: 2.6 }, { xAxis: 2.9, yAxis: 2.6 }],
              [{ name: "REFLATION\n(Growth ↓ · Infl ↓)", xAxis: -2.9, yAxis: -2.6 }, { xAxis: -0.2, yAxis: -2.6 }],
              [{ name: "STAGFLATION\n(Growth ↓ · Infl ↑)", xAxis: 0.2, yAxis: -2.6 }, { xAxis: 2.9, yAxis: -2.6 }],
            ],
          },
        },
        {
          name: "Latest",
          type: "effectScatter",
          coordinateSystem: "cartesian2d",
          data: [[latest.inflation_z, latest.growth_z]],
          symbolSize: 14,
          rippleEffect: { brushType: "stroke", scale: 2.8 },
          itemStyle: { color: "#f0883e" },
        },
      ],
    };
  }, [data]);

  if (error) {
    return (
      <div className="p-5 rounded-lg border border-rose-500/40 bg-[var(--bg-1)] text-rose-400 text-xs font-mono">
        Failed to load Macro Cycles: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-96 rounded-lg border border-[var(--line)] bg-[var(--bg-1)] animate-pulse flex items-center justify-center font-mono text-xs text-[var(--tx-2)]">
        Loading 4-Quadrant Growth-Inflation Regime Engine...
      </div>
    );
  }

  const badgeInfo = QUADRANT_BADGES[data.current_quadrant] || QUADRANT_BADGES.Reflation;

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-[var(--line)]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Regime &amp; Cycle Analytics
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--tx-0)] mt-0.5">
            Growth–Inflation Macro Cycle Clock
          </h1>
          <p className="text-xs text-[var(--tx-1)] mt-1">
            Market-implied 4-quadrant economic regime tracking 60-day standardized growth and inflation impulses without publication lag.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot status="ok" label="Daily Implied Regime" />
          <DataAgeBadge asOf={env?.as_of || ""} />
        </div>
      </div>

      {/* Current Regime Banner */}
      <div className={`p-4 rounded-lg border flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${badgeInfo.color}`}>
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wider opacity-80">Current Implied Quadrant</div>
          <div className="text-2xl font-bold tracking-tight mt-0.5">{data.current_quadrant}</div>
          <p className="text-xs opacity-90 mt-1">{badgeInfo.summary}</p>
        </div>
        <div className="flex gap-4 font-mono text-xs shrink-0 tabular-nums">
          <div className="p-2.5 rounded bg-[var(--bg-0)]/60 border border-[var(--line)]">
            <div className="text-[10px] text-[var(--tx-2)]">Growth Impulse</div>
            <div className="text-base font-bold text-[var(--tx-0)]">
              {data.current_growth_z >= 0 ? "+" : ""}
              {data.current_growth_z.toFixed(2)}σ
            </div>
          </div>
          <div className="p-2.5 rounded bg-[var(--bg-0)]/60 border border-[var(--line)]">
            <div className="text-[10px] text-[var(--tx-2)]">Inflation Impulse</div>
            <div className="text-base font-bold text-[var(--tx-0)]">
              {data.current_inflation_z >= 0 ? "+" : ""}
              {data.current_inflation_z.toFixed(2)}σ
            </div>
          </div>
        </div>
      </div>

      {/* Cycle Gauges */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {data.gauges.map((g) => (
          <div key={g.id} className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-1.5">
            <div className="flex items-center justify-between text-[11px] font-mono text-[var(--tx-2)]">
              <span>{g.name}</span>
              <span>{g.percentile.toFixed(0)}th pct</span>
            </div>
            <div className="text-xl font-mono font-bold text-[var(--tx-0)] tabular-nums">
              {g.value.toFixed(2)}%
            </div>
            <p className="text-[11px] text-[var(--tx-3)] leading-snug">{g.note}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7">
          <ChartCard
            eyebrow="4-Quadrant Cycle Trail"
            title="24-Month Growth vs. Inflation Spiral"
            subtitle="Orange ripple marks the current monthly realization"
            asOf={env?.as_of}
            sourceText="Copper/Gold, XLY/XLP, 10Y-2Y, HY OAS, 10Y Breakeven, Brent"
            dataTable={
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                    <th className="py-1.5 px-2">Month</th>
                    <th className="py-1.5 px-2">Quadrant</th>
                    <th className="py-1.5 px-2 text-right">Growth Z</th>
                    <th className="py-1.5 px-2 text-right">Inflation Z</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)]">
                  {data.trail.map((pt) => (
                    <tr key={pt.date}>
                      <td className="py-1.5 px-2">{pt.date}</td>
                      <td className="py-1.5 px-2">{pt.quadrant}</td>
                      <td className="py-1.5 px-2 text-right">{pt.growth_z.toFixed(2)}σ</td>
                      <td className="py-1.5 px-2 text-right">{pt.inflation_z.toFixed(2)}σ</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          >
            <ReactECharts option={spiralOption} style={{ height: "380px", width: "100%" }} />
          </ChartCard>
        </div>

        <div className="lg:col-span-5">
          <ChartCard
            eyebrow="Empirical Asset Playbook"
            title="Annualized Returns by Quadrant"
            subtitle="Realized historical performance across all daily observations in each regime"
            asOf={env?.as_of}
            sourceText="2000–Present Empirical Sample"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                    <th className="py-2 px-2 font-sans">Quadrant</th>
                    <th className="py-2 px-2 text-right">Share</th>
                    <th className="py-2 px-2 text-right">SPX</th>
                    <th className="py-2 px-2 text-right">Nifty</th>
                    <th className="py-2 px-2 text-right">Gold</th>
                    <th className="py-2 px-2 text-right">Brent</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)]">
                  {data.quadrant_returns.map((q) => {
                    const active = q.quadrant === data.current_quadrant;
                    const fmt = (v: number | null | undefined) =>
                      v === null || v === undefined ? (
                        <span className="text-[var(--tx-3)]">—</span>
                      ) : (
                        <span className={v >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]"}>
                          {v >= 0 ? "+" : ""}
                          {v.toFixed(1)}%
                        </span>
                      );
                    return (
                      <tr key={q.quadrant} className={active ? "bg-[var(--bg-2)]/80 font-semibold" : ""}>
                        <td className="py-2.5 px-2 font-sans text-[var(--tx-0)]">
                          {q.quadrant} {active && <span className="text-[10px] font-mono text-[var(--accent)]">●</span>}
                        </td>
                        <td className="py-2.5 px-2 text-right text-[var(--tx-2)]">{q.share_pct}%</td>
                        <td className="py-2.5 px-2 text-right">{fmt(q.returns.spx)}</td>
                        <td className="py-2.5 px-2 text-right">{fmt(q.returns.nifty50)}</td>
                        <td className="py-2.5 px-2 text-right">{fmt(q.returns.gold)}</td>
                        <td className="py-2.5 px-2 text-right">{fmt(q.returns.brent)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="text-[11px] text-[var(--tx-3)] mt-3 leading-relaxed font-sans">
                Descriptive historical sample averages conditioned on market-implied growth and inflation regimes. Not forward return guarantees.
              </p>
            </div>
          </ChartCard>
        </div>
      </div>
    </div>
  );
}
