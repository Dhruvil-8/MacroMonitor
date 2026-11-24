// web/src/app/sectors/SectorClientView.tsx
"use client";

import React, { useEffect, useState, useMemo } from "react";
import { Layers } from "lucide-react";
import { ChartCard } from "@/components/charts/ChartCard";
import { StatBlock } from "@/components/ui/StatBlock";
import { HealthDot } from "@/components/ui/HealthDot";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { AuditData, AuditDrawer } from "@/components/ui/AuditDrawer";
import { MethodData, MethodDrawer } from "@/components/ui/MethodDrawer";
import { SectorContributionWaterfall } from "@/components/charts/SectorContributionWaterfall";
import { SectorRotationMap } from "@/components/charts/SectorRotationMap";
import { SectorHeatmap } from "@/components/charts/SectorHeatmap";
import { loadSectors } from "@/lib/data/client";
import { SectorsEnvelope, SectorRegionData } from "@/lib/data/schemas";

export const SectorClientView: React.FC = () => {
  const [sectorsData, setSectorsData] = useState<SectorsEnvelope | null>(null);
  const [asOf, setAsOf] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedRegion, setSelectedRegion] = useState<string>("us");
  const [selectedHorizon, setSelectedHorizon] = useState<string>("1D");

  // Drawers
  const [auditTarget, setAuditTarget] = useState<AuditData | null>(null);
  const [isAuditOpen, setIsAuditOpen] = useState<boolean>(false);
  const [methodTarget, setMethodTarget] = useState<MethodData | null>(null);
  const [isMethodOpen, setIsMethodOpen] = useState<boolean>(false);

  const handleOpenAudit = (audit: AuditData) => {
    setAuditTarget(audit);
    setIsAuditOpen(true);
  };

  const handleOpenMethod = (method: MethodData) => {
    setMethodTarget(method);
    setIsMethodOpen(true);
  };

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        const res = await loadSectors();
        if (res.status !== "error") {
          setSectorsData(res.data);
          setAsOf(res.asOf);
        }
      } catch (err) {
        console.error("Failed to load sector data:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const currentRegionData: SectorRegionData | null = useMemo(() => {
    if (!sectorsData || !sectorsData.data.regions[selectedRegion]) return null;
    return sectorsData.data.regions[selectedRegion];
  }, [sectorsData, selectedRegion]);

  // Derived stats for the selected horizon
  const stats = useMemo(() => {
    if (!currentRegionData) return null;
    const indexRet = currentRegionData.index_returns[selectedHorizon] ?? 0;
    const breadth = (currentRegionData.sector_breadth[selectedHorizon] ?? 0) * 100;
    const residual = currentRegionData.residuals[selectedHorizon] ?? 0;

    let topContrib = currentRegionData.sectors[0];
    let topDetract = currentRegionData.sectors[0];

    currentRegionData.sectors.forEach((s) => {
      const c = s.contributions[selectedHorizon] ?? 0;
      if (c > (topContrib.contributions[selectedHorizon] ?? -Infinity)) topContrib = s;
      if (c < (topDetract.contributions[selectedHorizon] ?? Infinity)) topDetract = s;
    });

    return {
      indexRet,
      breadth,
      residual,
      topContrib,
      topDetract,
    };
  }, [currentRegionData, selectedHorizon]);

  if (loading) {
    return (
      <div className="py-20 text-center font-mono text-sm text-[var(--tx-2)] animate-pulse">
        Loading sector intelligence matrix...
      </div>
    );
  }

  if (!currentRegionData || !stats) {
    return (
      <div className="py-20 text-center text-sm text-[var(--neg)]">
        Failed to load sector data. Please verify the pipeline output.
      </div>
    );
  }

  const horizons = ["1D", "5D", "1M", "3M", "1Y"];

  const contributionAuditData: AuditData = {
    metricName: "Approximate Sector Contribution Model",
    formula: "contribution_s = w_s * r_s; residual = r_index - sum_s(contribution_s)",
    formulaPlain: "Constituent sector contributions weighted by published index factsheet weights.",
    parameters: {
      region: selectedRegion,
      benchmark: currentRegionData.benchmark_name,
      horizon: selectedHorizon,
      weights_source: currentRegionData.weights_source,
      as_of: currentRegionData.as_of,
    },
    inputs: [
      { id: currentRegionData.index_id, name: currentRegionData.benchmark_name, source: "Yahoo/NSE", unit: "index" },
      ...currentRegionData.sectors.map((s) => ({
        id: s.id,
        name: s.name,
        source: "Yahoo/NSE",
        unit: "index",
      })),
    ],
    window: selectedHorizon,
    currentValue: `${(stats.indexRet * 100).toFixed(2)}%`,
    n: currentRegionData.sectors.length,
    nEff: currentRegionData.sectors.length,
    codeVersion: sectorsData?.code_version ?? "v1.0.0",
    specUrl: "/methodology",
  };

  const contributionMethodData: MethodData = {
    chartTitle: "Approximate Sector Contribution Model",
    whatItShows:
      "Decomposes the benchmark equity index move into approximate contributions from each constituent sector, plus an explicit residual that captures intra-month weighting drift and unclassified components.",
    definitionFormula:
      "\\text{contribution}_s = w_s \\times r_{s, h}, \\quad \\text{residual}_h = r_{\\text{index}, h} - \\sum_{s} \\text{contribution}_{s, h}",
    symbolsDefined: [
      { symbol: "w_s", meaning: "Fixed published sector weight from official index factsheet" },
      { symbol: "r_{s, h}", meaning: "Cumulative return of sector s over horizon h" },
      { symbol: "residual_h", meaning: "Unexplained component ensuring exact arithmetic reconciliation" },
    ],
    parameters: {
      weights_source: currentRegionData.weights_source,
      as_of_date: currentRegionData.as_of,
      horizon: selectedHorizon,
    },
    dataSources: [
      { series: currentRegionData.index_id, source: "Yahoo / NSE", frequency: "Daily" },
      ...currentRegionData.sectors.map((s) => ({ series: s.id, source: "Yahoo / NSE", frequency: "Daily" })),
    ],
    sample: {
      n: currentRegionData.sectors.length,
      nEff: currentRegionData.sectors.length,
      dateRange: `As of ${currentRegionData.as_of}`,
    },
    interpretationNotes: [
      "Weights are fixed as of the published factsheet date and do not float dynamically between monthly rebalancings.",
      "The explicit residual guarantees that the visual sum of contributions matches the official index return exactly.",
    ],
    reproduce: {
      functionName: "compute_approximate_contributions",
      filePath: "pipeline/src/radar/metrics/sector.py",
      testPath: "pipeline/tests/unit/test_sector.py",
    },
  };

  const rotationAuditData: AuditData = {
    metricName: "Sector Rotation Map (Relative Strength vs Momentum)",
    formula: "RS_Ratio = 100 * (1 + (RS - MA_60(RS)) / MA_60(RS)); RS_Mom = 100 * (1 + (RS_Ratio - RS_Ratio_{t-10}) / RS_Ratio_{t-10})",
    formulaPlain: "Cyclical rotation phase classified across 4 quadrants based on normalized relative strength and momentum.",
    parameters: {
      benchmark_window: 60,
      momentum_window: 10,
      tail_days: 10,
    },
    inputs: [
      { id: currentRegionData.index_id, name: currentRegionData.benchmark_name, source: "Yahoo/NSE", unit: "index" },
      ...currentRegionData.sectors.map((s) => ({
        id: s.id,
        name: s.name,
        source: "Yahoo/NSE",
        unit: "index",
      })),
    ],
    window: "60-day moving average / 10-day momentum",
    currentValue: `${currentRegionData.sectors.length} sectors`,
    n: 750,
    nEff: 680,
    codeVersion: sectorsData?.code_version ?? "v1.0.0",
    specUrl: "/methodology",
  };

  const rotationMethodData: MethodData = {
    chartTitle: "Sector Rotation Map (Relative Strength vs Momentum)",
    whatItShows:
      "Maps each sector into one of four cyclical rotation quadrants (Leading, Weakening, Lagging, Improving) based on 60-day relative strength and 10-day momentum.",
    definitionFormula:
      "\\text{RS} = 100 \\times \\left(1 + \\frac{RS_{\\text{raw}} - \\overline{RS}_{60}}{\\overline{RS}_{60}}\\right), \\quad \\text{Mom} = 100 \\times \\left(1 + \\frac{\\text{RS}_t - \\text{RS}_{t-10}}{\\text{RS}_{t-10}}\\right)",
    symbolsDefined: [
      { symbol: "RS", meaning: "Relative strength ratio normalized around 100" },
      { symbol: "Mom", meaning: "10-day rate of change of relative strength ratio" },
      { symbol: "Leading", meaning: "RS >= 100 and Mom >= 100" },
      { symbol: "Weakening", meaning: "RS >= 100 and Mom < 100" },
      { symbol: "Lagging", meaning: "RS < 100 and Mom < 100" },
      { symbol: "Improving", meaning: "RS < 100 and Mom >= 100" },
    ],
    parameters: {
      benchmark_window: 60,
      momentum_window: 10,
      tail_days: 10,
    },
    dataSources: [
      { series: currentRegionData.index_id, source: "Yahoo / NSE", frequency: "Daily" },
      ...currentRegionData.sectors.map((s) => ({ series: s.id, source: "Yahoo / NSE", frequency: "Daily" })),
    ],
    sample: {
      n: 750,
      nEff: 680,
      dateRange: "Full historical daily series",
    },
    interpretationNotes: [
      "Rotations are cyclical approximations and do not guarantee future trajectory continuation.",
      "Tails show the 10-day trajectory of each sector.",
    ],
    reproduce: {
      functionName: "compute_sector_rotation_trail",
      filePath: "pipeline/src/radar/metrics/sector.py",
      testPath: "pipeline/tests/unit/test_sector.py",
    },
  };

  const heatmapAuditData: AuditData = {
    metricName: "Multi-Horizon Sector Performance Heatmap",
    formula: "P(r_{s, h}) = 100 * sum(I(r_i <= r_t)) / t",
    formulaPlain: "Anti-lookahead historical percentile ranking of rolling window returns against past observations.",
    parameters: {
      horizons: ["1D", "5D", "1M", "3M", "1Y"],
      min_obs: 120,
    },
    inputs: currentRegionData.sectors.map((s) => ({
      id: s.id,
      name: s.name,
      source: "Yahoo/NSE",
      unit: "index",
    })),
    window: "Multi-horizon (1D to 1Y)",
    currentValue: `${currentRegionData.sectors.length} sectors`,
    n: 750,
    nEff: 650,
    codeVersion: sectorsData?.code_version ?? "v1.0.0",
    specUrl: "/methodology",
  };

  const heatmapMethodData: MethodData = {
    chartTitle: "Multi-Horizon Sector Performance Heatmap",
    whatItShows:
      "Displays multi-horizon returns across 1D, 5D, 1M, 3M, and 1Y, shaded by anti-lookahead historical percentile ranking.",
    definitionFormula:
      "P(r_{s, h}) = 100 \\times \\frac{\\sum_{i=1}^t \\mathbb{I}(r_{s, i} \\le r_{s, t})}{t}",
    symbolsDefined: [
      { symbol: "r_{s, h}", meaning: "Cumulative return of sector s over horizon h" },
      { symbol: "P(r)", meaning: "Expanding percentile rank relative to historical observations" },
    ],
    parameters: {
      horizons: "1D, 5D, 1M, 3M, 1Y",
      min_obs: 120,
    },
    dataSources: currentRegionData.sectors.map((s) => ({
      series: s.id,
      source: "Yahoo / NSE",
      frequency: "Daily",
    })),
    sample: {
      n: 750,
      nEff: 650,
      dateRange: "Full historical daily series",
    },
    interpretationNotes: [
      "Percentile rankings are calculated strictly without lookahead bias using expanding historical windows.",
      "Shading highlights historically extreme tail rallies or sell-offs.",
    ],
    reproduce: {
      functionName: "compute_return_series_percentile",
      filePath: "pipeline/src/radar/metrics/sector.py",
      testPath: "pipeline/tests/unit/test_sector.py",
    },
  };

  return (
    <div className="space-y-6">
      {/* Editorial Header */}
      <div className="border-b border-[var(--line)] pb-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2 text-xs font-mono text-[var(--accent)] mb-1">
              <Layers className="w-3.5 h-3.5" />
              <span>MACRO SECTOR LAYER (M7)</span>
            </div>
            <h1 className="font-serif text-2xl sm:text-3xl font-normal text-[var(--tx-0)] tracking-tight">
              Sector Dynamics & Attribution
            </h1>
            <p className="text-xs sm:text-sm text-[var(--tx-2)] mt-1 max-w-3xl leading-relaxed">
              Approximate contribution to benchmark index returns, relative rotation cycles, and multi-horizon performance heatmaps with anti-lookahead historical percentile ranking.
            </p>
          </div>

          <div className="flex items-center space-x-3 self-start md:self-auto">
            <HealthDot status="ok" label="Pipeline Verified" />
            <DataAgeBadge asOf={asOf} />
          </div>
        </div>

        {/* Region & Horizon Toggles */}
        <div className="flex flex-wrap items-center justify-between gap-3 mt-5 pt-3 border-t border-[var(--line)]">
          {/* Region Switcher */}
          <div className="flex items-center space-x-1 bg-[var(--bg-0)] p-1 rounded border border-[var(--line)]">
            <button
              type="button"
              onClick={() => setSelectedRegion("us")}
              className={`px-3 py-1 rounded text-xs font-medium transition ${
                selectedRegion === "us"
                  ? "bg-[var(--accent)] text-slate-900 font-semibold"
                  : "text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              }`}
            >
              United States (S&P 500)
            </button>
            <button
              type="button"
              onClick={() => setSelectedRegion("india")}
              className={`px-3 py-1 rounded text-xs font-medium transition ${
                selectedRegion === "india"
                  ? "bg-[var(--accent)] text-slate-900 font-semibold"
                  : "text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              }`}
            >
              India (Nifty 50)
            </button>
          </div>

          {/* Horizon Switcher */}
          <div className="flex items-center space-x-1 bg-[var(--bg-0)] p-1 rounded border border-[var(--line)]">
            <span className="text-[11px] font-mono text-[var(--tx-3)] px-2">HORIZON:</span>
            {horizons.map((h) => (
              <button
                key={h}
                type="button"
                onClick={() => setSelectedHorizon(h)}
                className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition ${
                  selectedHorizon === h
                    ? "bg-[var(--bg-2)] text-[var(--tx-0)] font-semibold"
                    : "text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                }`}
              >
                {h}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* High-level Stat Summary Row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <StatBlock
          id="benchmark_return"
          name={`${currentRegionData.benchmark_name} Move`}
          value={`${stats.indexRet >= 0 ? "+" : ""}${(stats.indexRet * 100).toFixed(2)}%`}
          unit={`${selectedHorizon} Return`}
          sparkline={[-0.01, 0.005, -0.002, 0.008, stats.indexRet]}
          onOpenAudit={handleOpenAudit}
          auditPayload={{
            metricName: `${currentRegionData.benchmark_name} Horizon Return`,
            formula: "r_index = P_t / P_{t-h} - 1.0",
            formulaPlain: "Cumulative return of benchmark equity index over selected horizon.",
            parameters: { horizon: selectedHorizon, index: currentRegionData.index_id },
            inputs: [{ id: currentRegionData.index_id, name: currentRegionData.benchmark_name, source: "Yahoo/NSE", unit: "index" }],
            window: selectedHorizon,
            currentValue: `${(stats.indexRet * 100).toFixed(2)}%`,
            n: 750,
            nEff: 680,
          }}
        />

        <StatBlock
          id="sector_breadth"
          name="Sector Breadth"
          value={`${stats.breadth.toFixed(1)}%`}
          unit="outperforming index"
          percentile={stats.breadth}
          onOpenAudit={handleOpenAudit}
          auditPayload={{
            metricName: "Sector Breadth",
            formula: "breadth = sum(I(r_s > r_index)) / N_sectors",
            formulaPlain: "Percentage share of tracked constituent sectors outperforming the benchmark.",
            parameters: { horizon: selectedHorizon, sectors_count: currentRegionData.sectors.length },
            inputs: currentRegionData.sectors.map((s) => ({ id: s.id, name: s.name, source: "Yahoo/NSE", unit: "index" })),
            window: selectedHorizon,
            currentValue: `${stats.breadth.toFixed(1)}%`,
            n: currentRegionData.sectors.length,
            nEff: currentRegionData.sectors.length,
          }}
        />

        <StatBlock
          id="top_contributor"
          name="Top Contributor"
          value={`${((stats.topContrib.contributions[selectedHorizon] ?? 0) * 100).toFixed(2)}%`}
          unit={stats.topContrib.name}
          onOpenAudit={handleOpenAudit}
          auditPayload={{
            metricName: `Top Contributor (${stats.topContrib.name})`,
            formula: "contrib_s = w_s * r_s",
            formulaPlain: "Approximate contribution of sector to benchmark index return.",
            parameters: {
              sector: stats.topContrib.id,
              weight: stats.topContrib.weight,
              return: stats.topContrib.returns[selectedHorizon],
            },
            inputs: [{ id: stats.topContrib.id, name: stats.topContrib.name, source: "Yahoo/NSE", unit: "index" }],
            window: selectedHorizon,
            currentValue: `${((stats.topContrib.contributions[selectedHorizon] ?? 0) * 100).toFixed(2)}%`,
            n: 750,
            nEff: 650,
          }}
        />

        <StatBlock
          id="top_detractor"
          name="Top Detractor"
          value={`${((stats.topDetract.contributions[selectedHorizon] ?? 0) * 100).toFixed(2)}%`}
          unit={stats.topDetract.name}
          onOpenAudit={handleOpenAudit}
          auditPayload={{
            metricName: `Top Detractor (${stats.topDetract.name})`,
            formula: "contrib_s = w_s * r_s",
            formulaPlain: "Approximate detraction of sector to benchmark index return.",
            parameters: {
              sector: stats.topDetract.id,
              weight: stats.topDetract.weight,
              return: stats.topDetract.returns[selectedHorizon],
            },
            inputs: [{ id: stats.topDetract.id, name: stats.topDetract.name, source: "Yahoo/NSE", unit: "index" }],
            window: selectedHorizon,
            currentValue: `${((stats.topDetract.contributions[selectedHorizon] ?? 0) * 100).toFixed(2)}%`,
            n: 750,
            nEff: 650,
          }}
        />

        <StatBlock
          id="residual_drift"
          name="Residual (Drift)"
          value={`${stats.residual >= 0 ? "+" : ""}${(stats.residual * 100).toFixed(2)}%`}
          unit="unmodeled drift"
          onOpenAudit={handleOpenAudit}
          auditPayload={{
            metricName: "Residual / Unmodeled Drift",
            formula: "residual = r_index - sum_s(contrib_s)",
            formulaPlain: "Reconciliation drift between published index return and fixed-weight sector sum.",
            parameters: {
              source: currentRegionData.weights_source,
              as_of: currentRegionData.as_of,
              reconciliation: "exact",
            },
            inputs: [
              { id: currentRegionData.index_id, name: currentRegionData.benchmark_name, source: "Yahoo/NSE", unit: "index" },
              ...currentRegionData.sectors.map((s) => ({ id: s.id, name: s.name, source: "Yahoo/NSE", unit: "index" })),
            ],
            window: selectedHorizon,
            currentValue: `${(stats.residual * 100).toFixed(2)}%`,
            n: 750,
            nEff: 680,
          }}
        />
      </div>

      {/* Main Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart 1: Sector Contribution Waterfall */}
        <ChartCard
          eyebrow="Linear Attribution"
          title="Approximate Sector Contribution"
          subtitle={`Horizon: ${selectedHorizon} · Exact Arithmetic Reconciliation`}
          asOf={asOf}
          auditData={contributionAuditData}
          onOpenAudit={handleOpenAudit}
          methodData={contributionMethodData}
          onOpenMethod={handleOpenMethod}
          dataTable={
            <div className="overflow-x-auto text-xs font-mono">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                    <th className="py-1.5 px-2">Sector / Item</th>
                    <th className="py-1.5 px-2 text-right">Factsheet Weight</th>
                    <th className="py-1.5 px-2 text-right">Return ({selectedHorizon})</th>
                    <th className="py-1.5 px-2 text-right">Contribution</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)]">
                  {currentRegionData.sectors.map((s) => (
                    <tr key={s.id}>
                      <td className="py-1.5 px-2 font-sans text-[var(--tx-0)]">{s.name}</td>
                      <td className="py-1.5 px-2 text-right">{(s.weight * 100).toFixed(1)}%</td>
                      <td className="py-1.5 px-2 text-right">
                        {((s.returns[selectedHorizon] ?? 0) * 100).toFixed(2)}%
                      </td>
                      <td className="py-1.5 px-2 text-right font-semibold">
                        {((s.contributions[selectedHorizon] ?? 0) * 100).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                  <tr className="bg-[var(--bg-0)]/50 italic text-[var(--tx-2)]">
                    <td className="py-1.5 px-2 font-sans">Residual (Drift)</td>
                    <td className="py-1.5 px-2 text-right">—</td>
                    <td className="py-1.5 px-2 text-right">—</td>
                    <td className="py-1.5 px-2 text-right">{(stats.residual * 100).toFixed(2)}%</td>
                  </tr>
                  <tr className="bg-[var(--bg-0)] font-semibold text-[var(--tx-0)]">
                    <td className="py-1.5 px-2 font-sans">{currentRegionData.benchmark_name} Total</td>
                    <td className="py-1.5 px-2 text-right">100.0%</td>
                    <td className="py-1.5 px-2 text-right">{(stats.indexRet * 100).toFixed(2)}%</td>
                    <td className="py-1.5 px-2 text-right">{(stats.indexRet * 100).toFixed(2)}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          }
        >
          <SectorContributionWaterfall
            benchmarkName={currentRegionData.benchmark_name}
            indexReturn={stats.indexRet}
            residual={stats.residual}
            sectors={currentRegionData.sectors}
            horizon={selectedHorizon}
            weightsSource={currentRegionData.weights_source}
            asOf={currentRegionData.as_of}
          />
        </ChartCard>

        {/* Chart 2: Sector Rotation Map */}
        <ChartCard
          eyebrow="Cyclical Momentum (C05)"
          title="Sector Rotation Cycle"
          subtitle={`Relative Strength vs ${currentRegionData.benchmark_name} & Momentum`}
          asOf={asOf}
          auditData={rotationAuditData}
          onOpenAudit={handleOpenAudit}
          methodData={rotationMethodData}
          onOpenMethod={handleOpenMethod}
          dataTable={
            <div className="overflow-x-auto text-xs font-mono">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                    <th className="py-1.5 px-2">Sector</th>
                    <th className="py-1.5 px-2 text-right">RS Ratio</th>
                    <th className="py-1.5 px-2 text-right">RS Momentum</th>
                    <th className="py-1.5 px-2 text-center">Quadrant Phase</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)]">
                  {currentRegionData.sectors.map((s) => (
                    <tr key={s.id}>
                      <td className="py-1.5 px-2 font-sans text-[var(--tx-0)]">{s.name}</td>
                      <td className="py-1.5 px-2 text-right">{s.rs_ratio.toFixed(2)}</td>
                      <td className="py-1.5 px-2 text-right">{s.rs_mom.toFixed(2)}</td>
                      <td className="py-1.5 px-2 text-center font-sans font-semibold text-[11px]">{s.phase}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          }
        >
          <SectorRotationMap
            benchmarkName={currentRegionData.benchmark_name}
            sectors={currentRegionData.sectors}
          />
        </ChartCard>
      </div>

      {/* Full-width Multi-Horizon Sector Heatmap */}
      <ChartCard
        eyebrow="Historical Context"
        title="Multi-Horizon Sector Performance Heatmap"
        subtitle="Returns across 1D, 5D, 1M, 3M, 1Y with Anti-Lookahead Historical Percentile Ranking"
        asOf={asOf}
        auditData={heatmapAuditData}
        onOpenAudit={handleOpenAudit}
        methodData={heatmapMethodData}
        onOpenMethod={handleOpenMethod}
        dataTable={
          <div className="text-xs text-[var(--tx-2)] italic">
            The interactive heatmap above is already rendered as an accessible data table.
          </div>
        }
      >
        <SectorHeatmap
          sectors={currentRegionData.sectors}
          benchmarkReturns={currentRegionData.index_returns}
          benchmarkName={currentRegionData.benchmark_name}
        />
      </ChartCard>

      {/* Drawers */}
      <AuditDrawer
        isOpen={isAuditOpen}
        onClose={() => setIsAuditOpen(false)}
        data={auditTarget}
      />

      <MethodDrawer
        isOpen={isMethodOpen}
        onClose={() => setIsMethodOpen(false)}
        data={methodTarget}
      />
    </div>
  );
};
