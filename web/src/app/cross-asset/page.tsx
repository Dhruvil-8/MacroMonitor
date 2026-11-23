// web/src/app/cross-asset/page.tsx
"use client";

import React, { useEffect, useState } from "react";
import { ChartCard } from "@/components/charts/ChartCard";
import { CorrelationHeatmap } from "@/components/charts/CorrelationHeatmap";
import { RotationMap } from "@/components/charts/RotationMap";
import { PositioningChart } from "@/components/charts/PositioningChart";
import { AuditData, AuditDrawer } from "@/components/ui/AuditDrawer";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { HealthDot } from "@/components/ui/HealthDot";
import { loadCorrelations, loadPositioning, loadRotation } from "@/lib/data/client";
import { CorrelationsEnvelope, PositioningEnvelope, RotationEnvelope } from "@/lib/data/schemas";

export default function CrossAssetPage() {
  const [corrData, setCorrData] = useState<CorrelationsEnvelope | null>(null);
  const [rotData, setRotData] = useState<RotationEnvelope | null>(null);
  const [posData, setPosData] = useState<PositioningEnvelope | null>(null);
  const [asOf, setAsOf] = useState<string>("2026-11-20");
  const [loading, setLoading] = useState<boolean>(true);

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
        const [corrRes, rotRes, posRes] = await Promise.all([
          loadCorrelations(),
          loadRotation(),
          loadPositioning(),
        ]);

        if (corrRes.status !== "error") {
          setCorrData(corrRes.data);
          setAsOf(corrRes.asOf);
        }
        if (rotRes.status !== "error") {
          setRotData(rotRes.data);
        }
        if (posRes.status !== "error") {
          setPosData(posRes.data);
        }
      } catch (err) {
        console.error("Failed to load cross-asset data:", err);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[var(--line)]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Module / Cross-Asset
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--tx-0)] mt-0.5">
            Cross-Asset Correlations & Rotation
          </h1>
          <p className="text-xs text-[var(--tx-1)] mt-1">
            Correlation evolution, 3-period heatmap trio, and multi-horizon relative rotation cycles.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot status="ok" label="Stationary Returns" />
          <DataAgeBadge asOf={asOf} />
        </div>
      </div>

      {/* Main Grid: Heatmap Trio (left) & Rotation Map (right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Heatmap Trio */}
        <div className="lg:col-span-6">
          <ChartCard
            eyebrow="Pairwise Co-Movement"
            title="Correlation Heatmap Trio"
            subtitle="Pearson correlation of daily returns across 8 macro benchmarks"
            insightHeadline="Global equity correlation remains positive while US 10Y Treasury yield exhibits negative equity co-movement."
            asOf={asOf}
            auditData={{
              metricName: "Cross-Asset Pearson Correlation Matrix",
              formula: "\\rho = \\frac{\\text{Cov}(r_A, r_B)}{\\sigma_A \\sigma_B}",
              formulaPlain: "Sample Pearson correlation coefficient of daily stationary log returns over trailing 60 business days.",
              parameters: { window: 60, horizons: "now, 3M, 1Y" },
              inputs: (corrData?.data.benchmark_assets || []).map((id) => ({
                id,
                name: id.toUpperCase(),
                source: "Yahoo/FRED",
                unit: "returns",
              })),
              window: "60 business days",
              currentValue: "Diagonal: 1.00",
              n: 60,
              nEff: 48,
              codeVersion: "v1.0.0",
              specUrl: "/methodology",
            }}
            onOpenAudit={handleOpenAudit}
          >
            {corrData ? (
              <CorrelationHeatmap
                benchmarkAssets={corrData.data.benchmark_assets}
                heatmapNow={corrData.data.heatmap_now}
                heatmap3m={corrData.data.heatmap_3m}
                heatmap1y={corrData.data.heatmap_1y}
              />
            ) : (
              <div className="h-[320px] flex items-center justify-center text-xs text-[var(--tx-2)] font-mono">
                Loading Correlation Matrix...
              </div>
            )}
          </ChartCard>
        </div>

        {/* Relative Rotation Map */}
        <div className="lg:col-span-6">
          <ChartCard
            eyebrow="Momentum Cycle"
            title="Relative Rotation Map"
            subtitle="Relative strength ratio vs 10D momentum (Benchmark: S&P 500)"
            insightHeadline="Tech duration and industrial metals maintain positioning in the Leading quadrant relative to SPX."
            asOf={asOf}
            auditData={{
              metricName: "Relative Rotation Ratio and Momentum",
              formula: "\\text{RS} = \\frac{P_{\\text{asset}}}{P_{\\text{bench}}}, \\quad \\text{Ratio} = 100 \\times \\left(1 + \\frac{\\text{RS} - \\text{MA}_{60}(\\text{RS})}{\\text{MA}_{60}(\\text{RS})}\\right)",
              formulaPlain: "Relative strength ratio normalized against 60-day moving average vs 10-day rate of change.",
              parameters: { benchmark: "spx", window: 60, momentum_window: 10, tail_days: 10 },
              inputs: [{ id: "spx", name: "S&P 500 Benchmark", source: "Yahoo", unit: "index" }],
              window: "60D / 10D",
              currentValue: "Benchmark Origin: (100, 100)",
              n: 60,
              nEff: 42,
              codeVersion: "v1.0.0",
              specUrl: "/methodology",
            }}
            onOpenAudit={handleOpenAudit}
          >
            {rotData ? (
              <RotationMap
                benchmark={rotData.data.benchmark}
                assets={rotData.data.assets}
              />
            ) : (
              <div className="h-[320px] flex items-center justify-center text-xs text-[var(--tx-2)] font-mono">
                Loading Rotation Map...
              </div>
            )}
          </ChartCard>
        </div>
      </div>

      {/* Rolling Correlation Evolution for Pre-Registered Pairs */}
      {corrData && corrData.data.pairs && corrData.data.pairs.length > 0 && (
        <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-[var(--line)]">
            <div>
              <span className="text-[10px] uppercase font-mono tracking-widest text-[var(--tx-2)] block">
                Temporal Breakdown
              </span>
              <h3 className="text-base font-semibold text-[var(--tx-0)] tracking-tight">
                Rolling Correlation Evolution (60D vs 250D Shift)
              </h3>
            </div>
            <span className="text-xs font-mono text-[var(--tx-2)]">
              Shift = rho_60 - rho_250
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {corrData.data.pairs.map((p) => {
              const shiftPos = p.shift > 0;
              return (
                <div
                  key={p.id}
                  className="p-3.5 rounded bg-[var(--bg-0)] border border-[var(--line)] space-y-2 hover:border-[var(--line-strong)] transition-all"
                >
                  <div className="flex justify-between items-start">
                    <span className="text-xs font-mono font-medium text-[var(--tx-0)]">
                      {p.asset_a.toUpperCase()} / {p.asset_b.toUpperCase()}
                    </span>
                    <span
                      className={`text-[11px] font-mono font-medium px-1.5 py-0.5 rounded ${
                        Math.abs(p.shift) >= 0.4
                          ? "bg-[var(--warn)]/10 text-[var(--warn)] border border-[var(--warn)]/30"
                          : "text-[var(--tx-2)]"
                      }`}
                    >
                      Shift: {shiftPos ? "+" : ""}{p.shift.toFixed(2)}
                    </span>
                  </div>

                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-[var(--tx-2)]">60D Rho: <strong className="text-[var(--tx-0)]">{p.rho60 > 0 ? "+" : ""}{p.rho60.toFixed(2)}</strong></span>
                    <span className="text-[var(--tx-2)]">250D Rho: <strong className="text-[var(--tx-1)]">{p.rho250 > 0 ? "+" : ""}{p.rho250.toFixed(2)}</strong></span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* CFTC Commitments of Traders Positioning */}
      {posData && (
        <PositioningChart data={posData} />
      )}

      {/* Audit Drawer */}
      <AuditDrawer
        isOpen={isAuditOpen}
        onClose={() => setIsAuditOpen(false)}
        data={auditTarget}
      />
    </div>
  );
}
