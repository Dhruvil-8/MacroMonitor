// web/src/app/compare/page.tsx
"use client";

import React, { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { ChartCard } from "@/components/charts/ChartCard";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { HealthDot } from "@/components/ui/HealthDot";
import { fetchManifest, loadCompactSeries, loadDrivers } from "@/lib/data/client";
import { CompactSeriesData, DriverTargetMatrixRow } from "@/lib/data/schemas";
import { GitCompare, ShieldAlert, Plus, X, RotateCcw } from "lucide-react";

const LINE_COLORS = ["#58a6ff", "#3fb6a8", "#f0883e", "#a371f7"];

const HORIZON_DAYS: Record<string, number> = {
  "3M": 63,
  "6M": 126,
  "1Y": 252,
  "3Y": 756,
  "5Y": 1260,
  MAX: 99999,
};

function computePearson(x: number[], y: number[]): number | null {
  const n = Math.min(x.length, y.length);
  if (n < 15) return null;
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    syy += y[i] * y[i];
    sxy += x[i] * y[i];
  }
  const denom = Math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
  if (denom <= 1e-12) return null;
  return (n * sxy - sx * sy) / denom;
}

export default function ComparePage() {
  const [activeTab, setActiveTab] = useState<"compare" | "stress">("compare");
  const [availableIds, setAvailableIds] = useState<string[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>(["spx", "nifty50", "gold"]);
  const [addCandidate, setAddCandidate] = useState<string>("dax");
  const [horizon, setHorizon] = useState<string>("1Y");
  const [seriesCache, setSeriesCache] = useState<Record<string, CompactSeriesData>>({});
  const [asOf, setAsOf] = useState<string>("");
  const [loadingSeries, setLoadingSeries] = useState<boolean>(false);

  // Portfolio stress calculator state
  const [driverRows, setDriverRows] = useState<DriverTargetMatrixRow[]>([]);
  const [weights, setWeights] = useState<Record<string, number>>({
    spx: 40,
    nifty50: 40,
    dax: 20,
  });
  const [shocks, setShocks] = useState({
    row_equity: 0, // %
    brent: 10,     // %
    copper: 0,     // %
    gold: 0,       // %
    dxy: 2,        // %
    us10y: 50,     // bps
  });

  useEffect(() => {
    fetchManifest().then((res) => {
      if (res.status !== "error") {
        setAsOf(res.asOf);
        const ids = res.data.manifest.datasets
          .map((d) => d.path)
          .filter((p) => p.startsWith("/v1/series/") && p.endsWith(".json"))
          .map((p) => p.replace("/v1/series/", "").replace(".json", ""))
          .sort();
        setAvailableIds(ids);
      }
    });

    loadDrivers().then((res) => {
      if (res.status !== "error") {
        setDriverRows(res.data.data.matrix_250d);
      }
    });
  }, []);

  useEffect(() => {
    const missing = selectedIds.filter((id) => !seriesCache[id]);
    if (missing.length === 0) return;

    let active = true;
    setLoadingSeries(true);
    Promise.all(missing.map((id) => loadCompactSeries(id)))
      .then((results) => {
        if (!active) return;
        setSeriesCache((prev) => {
          const next = { ...prev };
          for (const res of results) {
            if (res.status !== "error") {
              next[res.data.data.id] = res.data.data;
            }
          }
          return next;
        });
      })
      .finally(() => {
        if (active) setLoadingSeries(false);
      });

    return () => {
      active = false;
    };
  }, [selectedIds, seriesCache]);

  const handleAddAsset = () => {
    if (!addCandidate || selectedIds.includes(addCandidate) || selectedIds.length >= 4) return;
    setSelectedIds([...selectedIds, addCandidate]);
  };

  const handleRemoveAsset = (id: string) => {
    if (selectedIds.length <= 1) return;
    setSelectedIds(selectedIds.filter((x) => x !== id));
  };

  // Align selected series on common dates and rebase to 100
  const comparison = useMemo(() => {
    const loaded = selectedIds.map((id) => seriesCache[id]).filter(Boolean);
    if (loaded.length === 0) {
      return { dates: [] as string[], series: [], correlations: [] as { a: string; b: string; corr: number | null }[], stats: [] };
    }

    // Build date -> value maps
    const maps = loaded.map((s) => {
      const m = new Map<string, number>();
      for (let i = 0; i < s.d.length; i++) {
        const val = s.v[i];
        if (val !== null && !isNaN(val)) {
          m.set(s.d[i], val);
        }
      }
      return m;
    });

    // Intersect dates preserving order from first series
    let commonDates = loaded[0].d.filter((dt) => maps.every((m) => m.has(dt)));
    const limit = HORIZON_DAYS[horizon] ?? 252;
    if (commonDates.length > limit) {
      commonDates = commonDates.slice(commonDates.length - limit);
    }

    if (commonDates.length < 2) {
      return { dates: [], series: [], correlations: [], stats: [] };
    }

    const rebasedSeries = loaded.map((s, idx) => {
      const m = maps[idx];
      const baseVal = m.get(commonDates[0])!;
      const isYieldOrSpread =
        s.unit === "percent" || s.unit === "bps" || s.id.includes("us10y") || s.id.includes("us2y") || s.id.includes("spread") || baseVal <= 0;

      const vals = commonDates.map((dt) => {
        const v = m.get(dt)!;
        if (isYieldOrSpread) {
          return parseFloat((100 + (v - baseVal)).toFixed(2));
        }
        return parseFloat(((v / baseVal) * 100).toFixed(2));
      });

      const lastRaw = m.get(commonDates[commonDates.length - 1])!;
      const totalChange = isYieldOrSpread ? lastRaw - baseVal : ((lastRaw / baseVal) - 1) * 100;

      // Daily returns for correlation
      const dailyRets: number[] = [];
      for (let i = 1; i < commonDates.length; i++) {
        const prev = m.get(commonDates[i - 1])!;
        const curr = m.get(commonDates[i])!;
        dailyRets.push(isYieldOrSpread ? curr - prev : Math.log(curr / prev));
      }

      return {
        id: s.id,
        name: s.name,
        isYieldOrSpread,
        baseVal,
        lastRaw,
        totalChange,
        rebased: vals,
        dailyRets,
        color: LINE_COLORS[idx % LINE_COLORS.length],
      };
    });

    const correlations: { a: string; b: string; corr: number | null }[] = [];
    for (let i = 0; i < rebasedSeries.length; i++) {
      for (let j = i + 1; j < rebasedSeries.length; j++) {
        correlations.push({
          a: rebasedSeries[i].id.toUpperCase(),
          b: rebasedSeries[j].id.toUpperCase(),
          corr: computePearson(rebasedSeries[i].dailyRets, rebasedSeries[j].dailyRets),
        });
      }
    }

    return {
      dates: commonDates,
      series: rebasedSeries,
      correlations,
      stats: rebasedSeries.map((s) => ({
        id: s.id,
        name: s.name,
        start: s.baseVal,
        end: s.lastRaw,
        change: s.totalChange,
        isYieldOrSpread: s.isYieldOrSpread,
      })),
    };
  }, [selectedIds, seriesCache, horizon]);

  const compareChartOption = useMemo(() => {
    if (comparison.dates.length === 0) return {};
    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(10, 14, 20, 0.95)",
        borderColor: "rgba(255, 255, 255, 0.12)",
        textStyle: { color: "#e6edf3", fontSize: 11, fontFamily: "monospace" },
      },
      legend: {
        data: comparison.series.map((s) => s.name),
        textStyle: { color: "#8b949e", fontSize: 11, fontFamily: "monospace" },
        top: 0,
      },
      grid: { top: 45, right: 25, bottom: 30, left: 45, containLabel: true },
      xAxis: {
        type: "category",
        data: comparison.dates,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.15)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "monospace" },
      },
      yAxis: {
        type: "value",
        name: "Indexed (Start = 100)",
        nameTextStyle: { color: "#8b949e", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "monospace" },
      },
      series: comparison.series.map((s) => ({
        name: s.name,
        type: "line",
        data: s.rebased,
        showSymbol: false,
        lineStyle: { width: 2, color: s.color },
        itemStyle: { color: s.color },
      })),
    };
  }, [comparison]);

  // Portfolio Stress Calculations using real 250D HAC betas from drivers.json
  const stressResult = useMemo(() => {
    const totalWeight = Object.values(weights).reduce((a, b) => a + Math.max(0, b), 0);
    if (totalWeight <= 0 || driverRows.length === 0) {
      return { totalWeight: 0, totalShockPct: 0, weightedR2: 0, contributions: [] as { label: string; impactPct: number }[] };
    }

    const factorDefs = [
      { id: "row_equity", label: "Global Equity (ROW)", shockInput: shocks.row_equity / 100 },
      { id: "brent", label: "Brent Crude Oil", shockInput: shocks.brent / 100 },
      { id: "copper", label: "Copper", shockInput: shocks.copper / 100 },
      { id: "gold", label: "Gold", shockInput: shocks.gold / 100 },
      { id: "dxy", label: "US Dollar (DXY)", shockInput: shocks.dxy / 100 },
      // us10y beta is decimal return per 1 bp, so shockInput is raw bps
      { id: "us10y", label: "US 10Y Yield", shockInput: shocks.us10y },
    ];

    const contribMap: Record<string, number> = {};
    for (const f of factorDefs) contribMap[f.id] = 0;
    let weightedR2 = 0;

    for (const row of driverRows) {
      const w = Math.max(0, weights[row.target_id] ?? 0) / totalWeight;
      if (w <= 0) continue;
      weightedR2 += w * row.r2;
      for (const f of factorDefs) {
        const beta = row.drivers[f.id]?.beta ?? 0;
        // decimal return * 100 = % impact
        contribMap[f.id] += w * beta * f.shockInput * 100;
      }
    }

    const contributions = factorDefs.map((f) => ({
      label: f.label,
      impactPct: contribMap[f.id],
    }));
    const totalShockPct = contributions.reduce((a, b) => a + b.impactPct, 0);

    return { totalWeight, totalShockPct, weightedR2, contributions };
  }, [weights, shocks, driverRows]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-[var(--line)]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Interactive Workbench
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--tx-0)] mt-0.5">
            Asset Compare &amp; Portfolio Macro Stress
          </h1>
          <p className="text-xs text-[var(--tx-1)] mt-1">
            Compare any tracked global series rebased to 100 with live window correlations, or stress-test your portfolio against 250-day macro factor betas.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot status="ok" label="Client-Side Real Data" />
          <DataAgeBadge asOf={asOf} />
        </div>
      </div>

      {/* Mode Tabs */}
      <div className="flex items-center gap-2 border-b border-[var(--line)] pb-3">
        <button
          onClick={() => setActiveTab("compare")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono transition border ${
            activeTab === "compare"
              ? "bg-[var(--accent)]/15 text-[var(--accent)] border-[var(--accent)] font-semibold"
              : "bg-[var(--bg-1)] text-[var(--tx-2)] border-[var(--line)] hover:text-[var(--tx-0)]"
          }`}
        >
          <GitCompare className="w-3.5 h-3.5" />
          <span>Multi-Asset Compare</span>
        </button>
        <button
          onClick={() => setActiveTab("stress")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono transition border ${
            activeTab === "stress"
              ? "bg-[var(--accent)]/15 text-[var(--accent)] border-[var(--accent)] font-semibold"
              : "bg-[var(--bg-1)] text-[var(--tx-2)] border-[var(--line)] hover:text-[var(--tx-0)]"
          }`}
        >
          <ShieldAlert className="w-3.5 h-3.5" />
          <span>Portfolio Macro Stress Calculator</span>
        </button>
      </div>

      {activeTab === "compare" ? (
        <div className="space-y-6">
          {/* Controls */}
          <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] flex flex-col lg:flex-row lg:items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-mono text-[var(--tx-2)] mr-1">Assets (max 4):</span>
              {selectedIds.map((id, idx) => (
                <span
                  key={id}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-[var(--bg-0)] border border-[var(--line)] text-xs font-mono text-[var(--tx-0)]"
                >
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: LINE_COLORS[idx % LINE_COLORS.length] }}
                  />
                  <span>{id.toUpperCase()}</span>
                  {selectedIds.length > 1 && (
                    <button
                      onClick={() => handleRemoveAsset(id)}
                      className="text-[var(--tx-3)] hover:text-[var(--neg)]"
                      aria-label={`Remove ${id}`}
                    >
                      <X className="w-3 h-3" />
                    </button>
                  )}
                </span>
              ))}

              {selectedIds.length < 4 && availableIds.length > 0 && (
                <div className="inline-flex items-center gap-1">
                  <select
                    value={addCandidate}
                    onChange={(e) => setAddCandidate(e.target.value)}
                    aria-label="Add series to compare"
                    className="px-2 py-1 rounded bg-[var(--bg-0)] border border-[var(--line)] text-xs font-mono text-[var(--tx-0)]"
                  >
                    {availableIds
                      .filter((id) => !selectedIds.includes(id))
                      .map((id) => (
                        <option key={id} value={id}>
                          {id.toUpperCase()}
                        </option>
                      ))}
                  </select>
                  <button
                    onClick={handleAddAsset}
                    className="px-2 py-1 rounded bg-[var(--bg-2)] hover:bg-[var(--accent)] hover:text-slate-900 border border-[var(--line)] text-xs font-mono transition flex items-center gap-1"
                  >
                    <Plus className="w-3 h-3" /> Add
                  </button>
                </div>
              )}
            </div>

            <div className="flex items-center gap-1 bg-[var(--bg-0)] p-1 rounded border border-[var(--line)]">
              {Object.keys(HORIZON_DAYS).map((h) => (
                <button
                  key={h}
                  onClick={() => setHorizon(h)}
                  className={`px-2.5 py-1 rounded text-xs font-mono transition ${
                    horizon === h
                      ? "bg-[var(--accent)] text-slate-900 font-semibold"
                      : "text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                  }`}
                >
                  {h}
                </button>
              ))}
            </div>
          </div>

          <ChartCard
            eyebrow="Normalized Trajectory"
            title={`Relative Performance (${horizon}, Indexed = 100)`}
            subtitle="Aligned strictly on common trading dates across selected series"
            asOf={asOf}
            sourceText="Precomputed Daily Market Series"
            dataTable={
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                    <th className="py-1.5 px-2">Series</th>
                    <th className="py-1.5 px-2 text-right">Start Value</th>
                    <th className="py-1.5 px-2 text-right">Latest Value</th>
                    <th className="py-1.5 px-2 text-right">Window Change</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)]">
                  {comparison.stats.map((s) => (
                    <tr key={s.id}>
                      <td className="py-1.5 px-2 text-[var(--tx-0)]">{s.name} ({s.id.toUpperCase()})</td>
                      <td className="py-1.5 px-2 text-right">{s.start.toFixed(2)}</td>
                      <td className="py-1.5 px-2 text-right">{s.end.toFixed(2)}</td>
                      <td className={`py-1.5 px-2 text-right ${s.change >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]"}`}>
                        {s.change >= 0 ? "+" : ""}
                        {s.change.toFixed(2)}
                        {s.isYieldOrSpread ? " pts" : "%"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          >
            {loadingSeries && comparison.dates.length === 0 ? (
              <div className="h-[340px] flex items-center justify-center text-xs font-mono text-[var(--tx-2)] animate-pulse">
                Loading aligned market series...
              </div>
            ) : comparison.dates.length === 0 ? (
              <div className="h-[340px] flex items-center justify-center text-xs font-mono text-[var(--tx-2)]">
                Insufficient overlapping dates for selected assets.
              </div>
            ) : (
              <ReactECharts option={compareChartOption} style={{ height: "360px", width: "100%" }} notMerge={true} />
            )}
          </ChartCard>

          {/* Pairwise Correlations & Window Stats */}
          {comparison.correlations.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-2">
                <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--tx-2)]">
                  Window Daily Return Correlations ({horizon})
                </h3>
                <div className="space-y-1.5 font-mono text-xs">
                  {comparison.correlations.map((c) => (
                    <div key={`${c.a}-${c.b}`} className="flex items-center justify-between py-1 border-b border-[var(--line)] last:border-0">
                      <span className="text-[var(--tx-0)]">{c.a} vs {c.b}</span>
                      <span
                        className={
                          c.corr === null
                            ? "text-[var(--tx-3)]"
                            : c.corr >= 0.3
                            ? "text-[var(--pos)] font-semibold"
                            : c.corr <= -0.3
                            ? "text-[var(--neg)] font-semibold"
                            : "text-[var(--tx-1)]"
                        }
                      >
                        {c.corr !== null ? c.corr.toFixed(2) : "—"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-2">
                <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--tx-2)]">
                  Window Return Summary ({comparison.dates[0]} to {comparison.dates[comparison.dates.length - 1]})
                </h3>
                <div className="space-y-1.5 font-mono text-xs">
                  {comparison.stats.map((s) => (
                    <div key={s.id} className="flex items-center justify-between py-1 border-b border-[var(--line)] last:border-0 tabular-nums">
                      <span className="text-[var(--tx-0)]">{s.name}</span>
                      <span className={s.change >= 0 ? "text-[var(--pos)] font-semibold" : "text-[var(--neg)] font-semibold"}>
                        {s.change >= 0 ? "+" : ""}
                        {s.change.toFixed(2)}
                        {s.isYieldOrSpread ? " pts" : "%"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Tab 2: Portfolio Macro Stress Calculator */
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-6 p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-4">
            <div className="flex items-center justify-between border-b border-[var(--line)] pb-3">
              <div>
                <h3 className="text-sm font-semibold text-[var(--tx-0)]">1. Portfolio Weights (%)</h3>
                <p className="text-[11px] text-[var(--tx-2)]">Normalized automatically across active allocations</p>
              </div>
              <button
                onClick={() => setWeights({ spx: 40, nifty50: 40, dax: 20 })}
                className="text-xs font-mono text-[var(--tx-2)] hover:text-[var(--accent)] flex items-center gap-1"
              >
                <RotateCcw className="w-3 h-3" /> Reset
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {driverRows.map((row) => (
                <div key={row.target_id} className="flex items-center justify-between gap-2 p-2 rounded bg-[var(--bg-0)] border border-[var(--line)]">
                  <span className="text-xs text-[var(--tx-0)] truncate" title={row.target_name}>
                    {row.target_name}
                  </span>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={weights[row.target_id] ?? 0}
                    onChange={(e) =>
                      setWeights((prev) => ({
                        ...prev,
                        [row.target_id]: Math.max(0, Number(e.target.value) || 0),
                      }))
                    }
                    className="w-16 px-2 py-1 rounded bg-[var(--bg-1)] border border-[var(--line)] text-right font-mono text-xs text-[var(--tx-0)]"
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="lg:col-span-6 space-y-6">
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-4">
              <div className="flex items-center justify-between border-b border-[var(--line)] pb-3">
                <div>
                  <h3 className="text-sm font-semibold text-[var(--tx-0)]">2. Hypothetical Macro Shocks</h3>
                  <p className="text-[11px] text-[var(--tx-2)]">Uses trailing 250-day Newey-West HAC betas</p>
                </div>
                <div className="flex gap-1.5">
                  <button
                    onClick={() => setShocks({ row_equity: -3, brent: 15, copper: -5, gold: 4, dxy: 2, us10y: 50 })}
                    className="px-2 py-1 rounded bg-[var(--bg-0)] border border-[var(--line)] text-[10px] font-mono hover:border-[var(--accent)]"
                  >
                    Oil + Rates Spike
                  </button>
                  <button
                    onClick={() => setShocks({ row_equity: 3, brent: -5, copper: 5, gold: 0, dxy: -2, us10y: -40 })}
                    className="px-2 py-1 rounded bg-[var(--bg-0)] border border-[var(--line)] text-[10px] font-mono hover:border-[var(--accent)]"
                  >
                    Goldilocks Easing
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 font-mono text-xs">
                {[
                  { key: "brent", label: "Brent Crude (%)", min: -30, max: 30, step: 1 },
                  { key: "us10y", label: "US 10Y Yield (bps)", min: -100, max: 100, step: 5 },
                  { key: "dxy", label: "US Dollar DXY (%)", min: -10, max: 10, step: 0.5 },
                  { key: "copper", label: "Copper (%)", min: -20, max: 20, step: 1 },
                  { key: "gold", label: "Gold (%)", min: -15, max: 15, step: 1 },
                  { key: "row_equity", label: "Global Equity Ex-Target (%)", min: -10, max: 10, step: 0.5 },
                ].map((item) => {
                  const val = shocks[item.key as keyof typeof shocks];
                  return (
                    <div key={item.key} className="p-2.5 rounded bg-[var(--bg-0)] border border-[var(--line)] space-y-1">
                      <div className="flex justify-between">
                        <span className="text-[var(--tx-2)]">{item.label}</span>
                        <span className={val > 0 ? "text-[var(--pos)]" : val < 0 ? "text-[var(--neg)]" : "text-[var(--tx-0)]"}>
                          {val > 0 ? "+" : ""}
                          {val}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={item.min}
                        max={item.max}
                        step={item.step}
                        value={val}
                        onChange={(e) => setShocks((prev) => ({ ...prev, [item.key]: Number(e.target.value) }))}
                        className="w-full accent-[var(--accent)]"
                      />
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Estimated Impact Card */}
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[10px] font-mono uppercase tracking-wider text-[var(--tx-2)]">
                    Estimated Factor-Implied Portfolio Impact
                  </span>
                  <div
                    className={`text-3xl font-mono font-bold mt-1 tabular-nums ${
                      stressResult.totalShockPct >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]"
                    }`}
                  >
                    {stressResult.totalShockPct >= 0 ? "+" : ""}
                    {stressResult.totalShockPct.toFixed(2)}%
                  </div>
                </div>
                <div className="text-right font-mono text-xs text-[var(--tx-2)]">
                  <div>Weighted R²: {(stressResult.weightedR2 * 100).toFixed(1)}%</div>
                  <div>Active Weight: {stressResult.totalWeight}%</div>
                </div>
              </div>

              <div className="space-y-1.5 border-t border-[var(--line)] pt-3 font-mono text-xs">
                {stressResult.contributions.map((c) => (
                  <div key={c.label} className="flex justify-between items-center tabular-nums">
                    <span className="text-[var(--tx-2)]">{c.label}</span>
                    <span className={c.impactPct >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]"}>
                      {c.impactPct >= 0 ? "+" : ""}
                      {c.impactPct.toFixed(2)}%
                    </span>
                  </div>
                ))}
              </div>

              <p className="text-[11px] text-[var(--tx-3)] border-t border-[var(--line)] pt-2">
                Linear first-order sensitivity based on trailing 250D multi-factor regressions. Actual tail moves can exceed linear beta estimates during regime shifts.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
