// web/src/app/radar/page.tsx
"use client";

import React, { useEffect, useMemo, useState } from "react";
import { ObservationCard } from "@/components/insight/ObservationCard";
import { DivergencePanel } from "@/components/charts/DivergencePanel";
import { CusumDetectorWorkspace } from "@/components/charts/CusumDetectorWorkspace";
import { ZScoreBandChart } from "@/components/charts/ZScoreBandChart";
import { AuditData, AuditDrawer } from "@/components/ui/AuditDrawer";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { HealthDot } from "@/components/ui/HealthDot";
import { loadDivergences, loadObservations } from "@/lib/data/client";
import { DivergenceItem, ObservationItem } from "@/lib/data/schemas";
import { Filter, SlidersHorizontal } from "lucide-react";

export default function RadarPage() {
  const [observations, setObservations] = useState<ObservationItem[]>([]);
  const [divergences, setDivergences] = useState<DivergenceItem[]>([]);
  const [testedCount, setTestedCount] = useState<number>(10);
  const [asOf, setAsOf] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);

  // Filters
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");

  // Audit Drawer
  const [auditTarget, setAuditTarget] = useState<AuditData | null>(null);
  const [isAuditOpen, setIsAuditOpen] = useState<boolean>(false);

  const handleOpenAudit = (audit: AuditData) => {
    setAuditTarget(audit);
    setIsAuditOpen(true);
  };

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const [obsRes, divRes] = await Promise.all([
          loadObservations(),
          loadDivergences(),
        ]);

        if (obsRes.status !== "error") {
          setObservations(obsRes.data.data.observations);
          setAsOf(obsRes.asOf);
        }
        if (divRes.status !== "error") {
          setDivergences(divRes.data.data.pairs);
          setTestedCount(divRes.data.data.tested_pairs_count);
        }
      } catch (err) {
        console.error("Failed to load radar data:", err);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  const filteredObservations = useMemo(() => {
    const riskTypes = ["vol_shift", "vol_cone_breach", "downside_corr_jump", "pc1_sync_spike"];
    const crossTypes = ["country_divergence", "divergence", "breadth_deterioration", "fx_driven_move"];
    const macroTypes = ["commodity_shock", "dollar_regime", "driver_shift", "sector_leadership"];

    return observations.filter((obs) => {
      const matchSeverity =
        severityFilter === "all" || obs.severity === severityFilter;

      let matchCategory = true;
      if (categoryFilter === "risk") {
        matchCategory = riskTypes.includes(obs.type);
      } else if (categoryFilter === "cross_market") {
        matchCategory = crossTypes.includes(obs.type);
      } else if (categoryFilter === "macro") {
        matchCategory = macroTypes.includes(obs.type);
      }

      return matchSeverity && matchCategory;
    });
  }, [observations, severityFilter, categoryFilter]);

  const activeDivergencesCount = useMemo(() => {
    return divergences.filter((d) => d.active).length;
  }, [divergences]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[var(--line)]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Module / Radar
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--tx-0)] mt-0.5">
            Cross-Asset Radar &amp; Change Feed
          </h1>
          <p className="text-xs text-[var(--tx-1)] mt-1">
            Deterministic observation feed, pre-registered divergence detector, and Page&apos;s two-sided CUSUM change-point monitor.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot status="ok" label="Deterministic Rules" />
          <DataAgeBadge asOf={asOf} />
        </div>
      </div>

      {/* Main Grid: Change Feed (left 7 cols) & Divergence Panels (right 5 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Change Feed */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)]">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="w-4 h-4 text-[var(--accent)]" />
              <span className="text-xs font-semibold text-[var(--tx-0)]">
                Filter Observations ({filteredObservations.length} of {observations.length})
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2 text-xs">
              {/* Category Pills */}
              <div className="flex rounded border border-[var(--line)] overflow-hidden">
                {[
                  { id: "all", label: "All" },
                  { id: "risk", label: "Vol & Risk" },
                  { id: "cross_market", label: "Cross-Market" },
                  { id: "macro", label: "Macro" },
                ].map((cat) => (
                  <button
                    key={cat.id}
                    onClick={() => setCategoryFilter(cat.id)}
                    className={`px-2 py-0.5 font-mono text-[11px] transition-colors ${categoryFilter === cat.id
                      ? "bg-[var(--accent)] text-white font-medium"
                      : "bg-[var(--bg-0)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                      }`}
                  >
                    {cat.label}
                  </button>
                ))}
              </div>

              {/* Severity Buttons */}
              <div className="flex rounded border border-[var(--line)] overflow-hidden">
                {["all", "high", "medium", "low"].map((sev) => (
                  <button
                    key={sev}
                    onClick={() => setSeverityFilter(sev)}
                    className={`px-2 py-0.5 font-mono text-[11px] capitalize transition-colors ${severityFilter === sev
                      ? "bg-[var(--accent)] text-white font-medium"
                      : "bg-[var(--bg-0)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                      }`}
                  >
                    {sev}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Observations List */}
          <div className="space-y-3">
            {filteredObservations.length > 0 ? (
              filteredObservations.map((obs) => (
                <ObservationCard
                  key={obs.id}
                  observation={obs}
                  onOpenAudit={handleOpenAudit}
                />
              ))
            ) : (
              <div className="p-8 text-center rounded-lg bg-[var(--bg-1)] border border-[var(--line)] text-xs text-[var(--tx-2)] font-mono">
                No observations match the selected criteria.
              </div>
            )}
          </div>
        </div>

        {/* Pre-Registered Divergence Panels */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-3.5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] flex items-center justify-between">
            <div>
              <h3 className="text-xs font-semibold text-[var(--tx-0)] uppercase tracking-wider">
                Pre-Registered Divergence Pairs
              </h3>
              <p className="text-[11px] text-[var(--tx-2)] mt-0.5">
                Active alerts: <strong className="text-[var(--tx-0)]">{activeDivergencesCount}</strong> of {testedCount} tested
              </p>
            </div>
            <span className="text-[10px] px-2 py-0.5 rounded bg-[var(--bg-2)] border border-[var(--line)] font-mono text-[var(--tx-2)]">
              False Discovery Protected
            </span>
          </div>

          <div className="space-y-3">
            {divergences.map((div) => (
              <DivergencePanel
                key={div.id}
                divergence={div}
                testedPairsCount={testedCount}
                onOpenAudit={handleOpenAudit}
              />
            ))}
          </div>
        </div>
      </div>

      {/* C24: Cumulative Sum (CUSUM) Change-Point Detector Workspace */}
      <div id="c24-cusum">
        <CusumDetectorWorkspace
          divergences={divergences}
          onOpenAudit={handleOpenAudit}
        />
      </div>

      {/* C21: Standardized Z-Score Outlier Band Chart */}
      <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono uppercase tracking-widest text-[var(--accent)] font-semibold">
                C21 / Innovation Monitor
              </span>
              <h3 className="text-sm font-semibold text-[var(--tx-0)] font-mono">
                Standardized Z-Score Outlier Bands
              </h3>
            </div>
            <p className="text-xs text-[var(--tx-2)] mt-0.5">
              Normalized innovation trajectories plotted against &plusmn;2&sigma; and &plusmn;3&sigma; thresholds to isolate regime shifts.
            </p>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[var(--bg-2)] border border-[var(--line)] text-[var(--tx-2)]">
            Expanding 60D Horizon
          </span>
        </div>
        <ZScoreBandChart />
      </div>

      {/* Audit Drawer */}
      <AuditDrawer
        isOpen={isAuditOpen}
        onClose={() => setIsAuditOpen(false)}
        data={auditTarget}
      />
    </div>
  );
}
