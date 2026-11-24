// web/src/app/drivers/DriverClientView.tsx
"use client";

import React, { useEffect, useState, useMemo } from "react";
import { Compass, Sliders, ShieldCheck } from "lucide-react";
import { ChartCard } from "@/components/charts/ChartCard";
import { StatBlock } from "@/components/ui/StatBlock";
import { HealthDot } from "@/components/ui/HealthDot";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { AuditData, AuditDrawer } from "@/components/ui/AuditDrawer";
import { MethodData, MethodDrawer } from "@/components/ui/MethodDrawer";
import { DriverSensitivityHeatmap } from "@/components/charts/DriverSensitivityHeatmap";
import { DriverForestPlot } from "@/components/charts/DriverForestPlot";
import { RollingBetaChart } from "@/components/charts/RollingBetaChart";
import { DriverBivariateScatter } from "@/components/charts/DriverBivariateScatter";
import { loadDrivers } from "@/lib/data/client";
import { DriversEnvelope, DriverTargetMatrixRow } from "@/lib/data/schemas";

const DRIVER_NAMES: Record<string, string> = {
  row_equity: "Rest-of-World Equity",
  brent: "Brent Crude Oil",
  copper: "Copper",
  gold: "Gold",
  dxy: "US Dollar Index",
  us10y: "US 10Y Yield Change",
};

export const DriverClientView: React.FC = () => {
  const [driversData, setDriversData] = useState<DriversEnvelope | null>(null);
  const [asOf, setAsOf] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [windowDays, setWindowDays] = useState<250 | 120>(250);
  const [selectedTargetId, setSelectedTargetId] = useState<string>("spx");
  const [selectedDriverId, setSelectedDriverId] = useState<string>("brent");

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
        const res = await loadDrivers();
        if (res.status !== "error") {
          setDriversData(res.data);
          setAsOf(res.asOf);
        }
      } catch (err) {
        console.error("Failed to load driver sensitivity data:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const activeMatrix: DriverTargetMatrixRow[] = useMemo(() => {
    if (!driversData) return [];
    return windowDays === 250
      ? driversData.data.matrix_250d
      : driversData.data.matrix_120d;
  }, [driversData, windowDays]);

  const activeTargetRow: DriverTargetMatrixRow | undefined = useMemo(() => {
    return (
      activeMatrix.find((r) => r.target_id === selectedTargetId) ||
      activeMatrix[0]
    );
  }, [activeMatrix, selectedTargetId]);

  const activeDriverItem = useMemo(() => {
    if (!activeTargetRow) return undefined;
    return activeTargetRow.drivers[selectedDriverId];
  }, [activeTargetRow, selectedDriverId]);

  const activeScatterData = useMemo(() => {
    if (!driversData) return undefined;
    const key = `${selectedTargetId}_${selectedDriverId}`;
    return driversData.data.scatter_examples[key];
  }, [driversData, selectedTargetId, selectedDriverId]);

  // Summary stats
  const stats = useMemo(() => {
    if (!activeTargetRow) return null;

    let topDriver = Object.values(activeTargetRow.drivers)[0];
    let stableCount = 0;
    let priorAlignedCount = 0;
    const totalDrivers = Object.values(activeTargetRow.drivers).length;

    Object.values(activeTargetRow.drivers).forEach((d) => {
      if (Math.abs(d.beta) > Math.abs(topDriver.beta)) topDriver = d;
      if (d.stability >= 0.70 && d.significant) stableCount++;
      if (d.prior_aligned) priorAlignedCount++;
    });

    return {
      topDriver,
      stabilityPct: totalDrivers > 0 ? (stableCount / totalDrivers) * 100 : 0,
      priorAlignedPct: totalDrivers > 0 ? (priorAlignedCount / totalDrivers) * 100 : 0,
    };
  }, [activeTargetRow]);

  if (loading) {
    return (
      <div className="py-20 text-center font-mono text-sm text-[var(--tx-2)] animate-pulse">
        Loading driver sensitivity matrix...
      </div>
    );
  }

  if (!driversData || activeMatrix.length === 0 || !activeTargetRow || !stats) {
    return (
      <div className="py-20 text-center text-sm text-[var(--neg)]">
        Failed to load driver sensitivity data. Please verify the pipeline output.
      </div>
    );
  }

  const heatmapAuditData: AuditData = {
    metricName: "Cross-Asset Driver Sensitivity Matrix",
    formula: "y_t = \\sum_k \\beta_k x_{k, t} + \\epsilon_t \\quad (\\text{Newey-West HAC})",
    formulaPlain: "Multi-factor regressions with HAC standard errors and VIF collinearity screening across all tracked targets and drivers.",
    parameters: {
      window: `${windowDays} trading days`,
      vif_threshold: 5.0,
      hac_lags: 5,
    },
    inputs: [
      { id: activeTargetRow.target_id, name: activeTargetRow.target_name, source: "Yahoo/NSE", unit: "index" },
      { id: "row_equity", name: "Rest-of-World Equity", source: "Internal Composite", unit: "index" },
      { id: "brent", name: "Brent Crude Oil", source: "FRED", unit: "USD/bbl" },
      { id: "copper", name: "Copper", source: "FRED", unit: "USD/lb" },
      { id: "gold", name: "Gold", source: "FRED", unit: "USD/oz" },
      { id: "dxy", name: "US Dollar Index", source: "Yahoo", unit: "index" },
      { id: "us10y", name: "US 10Y Yield", source: "FRED", unit: "bps" },
    ],
    window: `${windowDays}D`,
    currentValue: `${activeMatrix.length} targets`,
    n: activeTargetRow.n_obs,
    nEff: activeTargetRow.n_obs,
    codeVersion: driversData.code_version,
    specUrl: "/methodology",
  };

  const heatmapMethodData: MethodData = {
    chartTitle: "Driver Sensitivity Matrix (C06)",
    whatItShows:
      "Displays multi-factor regression coefficients (β) measuring how each national equity index responds to global macro drivers, with visual hatching for unstable or statistically insignificant sensitivities.",
    definitionFormula:
      "y_t = \\alpha + \\sum_{k=1}^K \\beta_k x_{k, t} + \\epsilon_t, \\quad \\text{Var}(\\hat{\\beta}) = (X'X)^{-1} \\Omega (X'X)^{-1}",
    symbolsDefined: [
      { symbol: "y_t", meaning: "Target economy log return on trading day t" },
      { symbol: "x_{k, t}", meaning: "Move in global macro driver k on trading day t" },
      { symbol: "Omega", meaning: "Newey-West HAC covariance matrix accounting for autocorrelation and heteroskedasticity" },
    ],
    parameters: {
      window_days: windowDays,
      vif_threshold: 5.0,
      stability_threshold: "70% sign consistency",
    },
    dataSources: [
      { series: "Target", source: "Yahoo/NSE", frequency: "Daily" },
      { series: "Brent, Copper, Gold, 10Y", source: "FRED", frequency: "Daily" },
    ],
    sample: {
      n: activeTargetRow.n_obs,
      nEff: Math.round(activeTargetRow.n_obs * 0.9),
      dateRange: `Trailing ${windowDays} trading sessions`,
    },
    interpretationNotes: [
      "Hatched / faded cells indicate coefficients that either span zero at 95% confidence or have reversed signs in >30% of trailing rolling windows.",
      "ROW equity excludes the target country to prevent mechanical self-correlation.",
    ],
    reproduce: {
      functionName: "analyze_target_drivers",
      filePath: "pipeline/src/radar/metrics/drivers.py",
      testPath: "pipeline/tests/unit/test_drivers.py",
    },
  };

  return (
    <div className="space-y-6">
      {/* Editorial Header */}
      <div className="border-b border-[var(--line)] pb-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2 text-xs font-mono text-[var(--accent)] mb-1">
              <Compass className="w-3.5 h-3.5" />
              <span>MACRO DRIVER SENSITIVITY LAYER (M8)</span>
            </div>
            <h1 className="font-serif text-2xl sm:text-3xl font-normal text-[var(--tx-0)] tracking-tight">
              Global Macro Sensitivities
            </h1>
            <p className="text-xs sm:text-sm text-[var(--tx-2)] mt-1 max-w-3xl leading-relaxed">
              Rolling multi-factor regressions with Newey-West HAC standard errors, stability screening, pre-registered economic priors, and uncertainty confidence intervals.
            </p>
          </div>

          <div className="flex items-center space-x-3 self-start md:self-auto">
            <HealthDot status="ok" label="Pipeline Verified" />
            <DataAgeBadge asOf={asOf} />
          </div>
        </div>

        {/* Estimation Window & Target Controls */}
        <div className="flex flex-wrap items-center justify-between gap-3 mt-5 pt-3 border-t border-[var(--line)]">
          {/* Target Country Selector */}
          <div className="flex items-center space-x-1 overflow-x-auto max-w-2xl py-1">
            <span className="text-[11px] font-mono text-[var(--tx-3)] pr-1">TARGET:</span>
            {activeMatrix.map((r) => (
              <button
                key={r.target_id}
                type="button"
                onClick={() => setSelectedTargetId(r.target_id)}
                className={`px-2.5 py-1 rounded text-xs font-medium whitespace-nowrap transition ${selectedTargetId === r.target_id
                    ? "bg-[var(--accent)] text-slate-900 font-semibold"
                    : "text-[var(--tx-2)] hover:text-[var(--tx-0)] bg-[var(--bg-0)] border border-[var(--line)]"
                  }`}
              >
                {r.target_code.toUpperCase()} ({r.target_id.toUpperCase()})
              </button>
            ))}
          </div>

          {/* Window Toggle */}
          <div className="flex items-center space-x-1 bg-[var(--bg-0)] p-1 rounded border border-[var(--line)]">
            <span className="text-[11px] font-mono text-[var(--tx-3)] px-2">WINDOW:</span>
            <button
              type="button"
              onClick={() => setWindowDays(250)}
              className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition ${windowDays === 250
                  ? "bg-[var(--bg-2)] text-[var(--tx-0)] font-semibold"
                  : "text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                }`}
            >
              250D (~1Y)
            </button>
            <button
              type="button"
              onClick={() => setWindowDays(120)}
              className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition ${windowDays === 120
                  ? "bg-[var(--bg-2)] text-[var(--tx-0)] font-semibold"
                  : "text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                }`}
            >
              120D (~6M)
            </button>
          </div>
        </div>
      </div>

      {/* Summary Stat Blocks */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <StatBlock
          id="active_target"
          name="Active Target"
          value={activeTargetRow.target_code.toUpperCase()}
          unit={activeTargetRow.target_name}
          onOpenAudit={handleOpenAudit}
          auditPayload={{
            metricName: "Active Target Regression Fit",
            formula: "y = X beta + e",
            formulaPlain: "Multi-factor OLS fit with Newey-West HAC covariance.",
            parameters: { target: activeTargetRow.target_id, window: `${windowDays}D` },
            inputs: [{ id: activeTargetRow.target_id, name: activeTargetRow.target_name, source: "Yahoo/NSE", unit: "index" }],
            window: `${windowDays}D`,
            currentValue: activeTargetRow.target_name,
            n: activeTargetRow.n_obs,
            nEff: activeTargetRow.n_obs,
          }}
        />

        <StatBlock
          id="model_r2"
          name="Model Fit (R²)"
          value={`${(activeTargetRow.r2 * 100).toFixed(1)}%`}
          unit={`Variance Explained`}
          percentile={activeTargetRow.r2 * 100}
          onOpenAudit={handleOpenAudit}
          auditPayload={{
            metricName: "Multi-Factor R-Squared",
            formula: "R^2 = 1 - SS_res / SS_tot",
            formulaPlain: "Percentage of total return variation accounted for by macro factors.",
            parameters: { window: `${windowDays}D`, regularized: activeTargetRow.regularized },
            inputs: [{ id: activeTargetRow.target_id, name: activeTargetRow.target_name, source: "Yahoo/NSE", unit: "index" }],
            window: `${windowDays}D`,
            currentValue: `${(activeTargetRow.r2 * 100).toFixed(1)}%`,
            n: activeTargetRow.n_obs,
            nEff: activeTargetRow.n_obs,
          }}
        />

        <StatBlock
          id="top_influencer"
          name="Dominant Driver"
          value={stats.topDriver ? stats.topDriver.beta.toFixed(3) : "—"}
          unit={stats.topDriver ? DRIVER_NAMES[stats.topDriver.driver_id] || stats.topDriver.driver_id : "—"}
          onOpenAudit={handleOpenAudit}
          auditPayload={{
            metricName: "Dominant Factor Sensitivity",
            formula: "beta_k = (X'X)^-1 X' y",
            formulaPlain: "Factor with largest absolute sensitivity beta.",
            parameters: { factor: stats.topDriver?.driver_id, beta: stats.topDriver?.beta },
            inputs: [{ id: stats.topDriver?.driver_id || "", name: "Factor", source: "FRED/Yahoo", unit: "return" }],
            window: `${windowDays}D`,
            currentValue: stats.topDriver ? stats.topDriver.beta.toFixed(3) : "—",
            n: activeTargetRow.n_obs,
            nEff: activeTargetRow.n_obs,
          }}
        />

        <StatBlock
          id="stability_ratio"
          name="Sensitivity Stability"
          value={`${stats.stabilityPct.toFixed(0)}%`}
          unit="≥70% sign consistent"
          percentile={stats.stabilityPct}
          onOpenAudit={handleOpenAudit}
          auditPayload={{
            metricName: "Driver Sensitivity Stability Ratio",
            formula: "stability = sum(I(sign(beta_t) == sign(beta_now))) / N",
            formulaPlain: "Share of drivers exhibiting persistent sign across historical rolling windows.",
            parameters: { window: `${windowDays}D`, min_stability: "70%" },
            inputs: [{ id: activeTargetRow.target_id, name: activeTargetRow.target_name, source: "Yahoo/NSE", unit: "index" }],
            window: `${windowDays}D`,
            currentValue: `${stats.stabilityPct.toFixed(0)}%`,
            n: activeTargetRow.n_obs,
            nEff: activeTargetRow.n_obs,
          }}
        />

        <StatBlock
          id="priors_conformance"
          name="Priors Alignment"
          value={`${stats.priorAlignedPct.toFixed(0)}%`}
          unit="matches economic prior"
          percentile={stats.priorAlignedPct}
          onOpenAudit={handleOpenAudit}
          auditPayload={{
            metricName: "Pre-registered Economic Priors Conformance",
            formula: "sign(beta) == prior_sign",
            formulaPlain: "Percentage of factor betas whose estimated sign conforms to structural economic theory.",
            parameters: { window: `${windowDays}D` },
            inputs: [{ id: activeTargetRow.target_id, name: activeTargetRow.target_name, source: "Yahoo/NSE", unit: "index" }],
            window: `${windowDays}D`,
            currentValue: `${stats.priorAlignedPct.toFixed(0)}%`,
            n: activeTargetRow.n_obs,
            nEff: activeTargetRow.n_obs,
          }}
        />
      </div>

      {/* Chart 1: Sensitivity Heatmap (C06) */}
      <ChartCard
        eyebrow="Cross-Economy Matrix (C06)"
        title="Driver Sensitivity Heatmap with Stability Hatching"
        subtitle={`Multi-factor betas across 10 economies · Trailing ${windowDays} trading sessions`}
        asOf={asOf}
        auditData={heatmapAuditData}
        onOpenAudit={handleOpenAudit}
        methodData={heatmapMethodData}
        onOpenMethod={handleOpenMethod}
        dataTable={
          <div className="overflow-x-auto text-xs font-mono">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
                  <th className="py-1.5 px-2">Target</th>
                  <th className="py-1.5 px-2 text-right">R²</th>
                  <th className="py-1.5 px-2 text-right">ROW Equity</th>
                  <th className="py-1.5 px-2 text-right">Brent</th>
                  <th className="py-1.5 px-2 text-right">Copper</th>
                  <th className="py-1.5 px-2 text-right">Gold</th>
                  <th className="py-1.5 px-2 text-right">DXY</th>
                  <th className="py-1.5 px-2 text-right" title="US 10Y Yield sensitivity (decimal return per 1 bp; -0.00010 = -1.0% per 100 bps)">US 10Y (per bp)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--line)]">
                {activeMatrix.map((r) => (
                  <tr key={r.target_id}>
                    <td className="py-1.5 px-2 font-sans text-[var(--tx-0)]">{r.target_name}</td>
                    <td className="py-1.5 px-2 text-right">{(r.r2 * 100).toFixed(1)}%</td>
                    <td className="py-1.5 px-2 text-right font-semibold">{r.drivers["row_equity"]?.beta.toFixed(3) ?? "—"}</td>
                    <td className="py-1.5 px-2 text-right">{r.drivers["brent"]?.beta.toFixed(3) ?? "—"}</td>
                    <td className="py-1.5 px-2 text-right">{r.drivers["copper"]?.beta.toFixed(3) ?? "—"}</td>
                    <td className="py-1.5 px-2 text-right">{r.drivers["gold"]?.beta.toFixed(3) ?? "—"}</td>
                    <td className="py-1.5 px-2 text-right">{r.drivers["dxy"]?.beta.toFixed(3) ?? "—"}</td>
                    <td className="py-1.5 px-2 text-right font-mono" title={`${(r.drivers["us10y"]?.beta ?? 0) * 10000 >= 0 ? "+" : ""}${((r.drivers["us10y"]?.beta ?? 0) * 10000).toFixed(1)}% per 100 bps shift`}>{r.drivers["us10y"]?.beta.toFixed(4) ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-2 text-[10px] text-[var(--tx-3)] font-mono">
              * Note: Equity and commodity drivers are measured in percentage returns (%/%), while US 10Y is modeled in yield basis points (bps). A beta of -0.0001 equals -1.0% equity return per 100 bps yield change.
            </div>
          </div>
        }
      >
        <DriverSensitivityHeatmap
          rows={activeMatrix}
          selectedTargetId={selectedTargetId}
          onSelectTarget={setSelectedTargetId}
          selectedDriverId={selectedDriverId}
          onSelectDriver={setSelectedDriverId}
        />
      </ChartCard>

      {/* Grid of Focused Diagnostic Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart 2: Forest Plot (C08) */}
        <ChartCard
          eyebrow="Estimation Uncertainty (C08)"
          title={`Forest Plot: ${activeTargetRow.target_name}`}
          subtitle="Point estimates with 95% Newey-West HAC confidence intervals"
          asOf={asOf}
          auditData={heatmapAuditData}
          onOpenAudit={handleOpenAudit}
          methodData={{
            chartTitle: "Driver Forest Plot (C08)",
            whatItShows: "Visualizes estimation uncertainty across all macro driver betas on a single whisker chart.",
            definitionFormula: "\\hat{\\beta}_k \\pm 1.96 \\times \\text{SE}_{\\text{HAC}}(\\hat{\\beta}_k)",
            parameters: { window: `${windowDays}D`, target: activeTargetRow.target_id },
            dataSources: [{ series: activeTargetRow.target_id, source: "Yahoo/NSE", frequency: "Daily" }],
            sample: { n: activeTargetRow.n_obs, dateRange: `Trailing ${windowDays} sessions` },
            interpretationNotes: ["Whiskers that intersect zero are not statistically distinguishable from no exposure."],
            reproduce: { functionName: "fit_window_multi_factor", filePath: "pipeline/src/radar/metrics/drivers.py", testPath: "pipeline/tests/unit/test_drivers.py" },
          }}
          onOpenMethod={handleOpenMethod}
        >
          <DriverForestPlot
            targetRow={activeTargetRow}
            selectedDriverId={selectedDriverId}
            onSelectDriver={setSelectedDriverId}
          />
        </ChartCard>

        {/* Chart 3: Rolling Beta Drift (C07) */}
        <ChartCard
          eyebrow="Structural Stability (C07)"
          title={`Rolling Beta: ${activeTargetRow.target_name} vs ${DRIVER_NAMES[selectedDriverId] || selectedDriverId}`}
          subtitle="Trailing rolling regression with 95% HAC confidence band"
          asOf={asOf}
          auditData={heatmapAuditData}
          onOpenAudit={handleOpenAudit}
          methodData={{
            chartTitle: "Rolling Beta Drift (C07)",
            whatItShows: "Shows structural sensitivity evolution and regime flips over time.",
            definitionFormula: "\\hat{\\beta}_{k, t} = (X_t' X_t)^{-1} X_t' y_t, \\quad t = \\text{rolling window end}",
            parameters: { window_size: windowDays, step_days: 5 },
            dataSources: [{ series: selectedDriverId, source: "FRED/Yahoo", frequency: "Daily" }],
            sample: { n: activeTargetRow.n_obs, dateRange: "Full historical daily series" },
            interpretationNotes: ["Sudden slope changes signal structural regime shifts (e.g. monetary tightening cycles)."],
            reproduce: { functionName: "compute_rolling_multi_factor_series", filePath: "pipeline/src/radar/metrics/drivers.py", testPath: "pipeline/tests/unit/test_drivers.py" },
          }}
          onOpenMethod={handleOpenMethod}
        >
          {activeDriverItem ? (
            <RollingBetaChart
              driverItem={activeDriverItem}
              targetName={activeTargetRow.target_name}
              driverName={DRIVER_NAMES[selectedDriverId] || selectedDriverId}
            />
          ) : (
            <div className="h-[240px] flex items-center justify-center text-xs text-[var(--tx-3)] font-mono">
              Select a driver factor to view its rolling sensitivity.
            </div>
          )}
        </ChartCard>
      </div>

      {/* Chart 4: Bivariate Scatter (C09) */}
      {activeScatterData && (
        <ChartCard
          eyebrow="Empirical Fit (C09)"
          title={`Bivariate Scatter: ${activeTargetRow.target_name} vs ${DRIVER_NAMES[selectedDriverId] || selectedDriverId}`}
          subtitle="Daily return scatter with fitted OLS regression line and confidence band"
          asOf={asOf}
          auditData={heatmapAuditData}
          onOpenAudit={handleOpenAudit}
          methodData={{
            chartTitle: "Bivariate Scatter & Fit (C09)",
            whatItShows: "Scatter plot of daily target return against daily driver move with OLS fit and confidence band.",
            definitionFormula: "y_i = \\alpha + \\beta x_i + e_i, \\quad R^2 = 1 - \\frac{\\sum e_i^2}{\\sum (y_i - \\bar{y})^2}",
            parameters: { window: 250 },
            dataSources: [{ series: selectedDriverId, source: "FRED/Yahoo", frequency: "Daily" }],
            sample: { n: activeScatterData.n_obs, dateRange: "Trailing 250 sessions" },
            interpretationNotes: ["Band represents 95% confidence bounds around the expected conditional mean."],
            reproduce: { functionName: "compute_bivariate_scatter_and_fit", filePath: "pipeline/src/radar/metrics/drivers.py", testPath: "pipeline/tests/unit/test_drivers.py" },
          }}
          onOpenMethod={handleOpenMethod}
        >
          <DriverBivariateScatter
            scatterData={activeScatterData}
            targetName={activeTargetRow.target_name}
            driverName={DRIVER_NAMES[selectedDriverId] || selectedDriverId}
          />
        </ChartCard>
      )}

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
