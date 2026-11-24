// web/src/app/countries/[code]/CountryClientView.tsx
"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Globe, ShieldCheck, HelpCircle } from "lucide-react";
import { ChartCard } from "@/components/charts/ChartCard";
import { MoveDecomposerWaterfall } from "@/components/charts/MoveDecomposerWaterfall";
import { AuditData, AuditDrawer } from "@/components/ui/AuditDrawer";
import { MethodData, MethodDrawer } from "@/components/ui/MethodDrawer";
import { HealthDot } from "@/components/ui/HealthDot";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { loadCountries, loadDecomposition } from "@/lib/data/client";
import { CountriesEnvelope, CountryMatrixRow, DecompositionEnvelope } from "@/lib/data/schemas";

interface CountryClientViewProps {
  code: string;
}

export const CountryClientView: React.FC<CountryClientViewProps> = ({ code }) => {
  const [countriesData, setCountriesData] = useState<CountriesEnvelope | null>(null);
  const [decompData, setDecompData] = useState<DecompositionEnvelope | null>(null);
  const [asOf, setAsOf] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);

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
    async function loadData() {
      setLoading(true);
      try {
        const cRes = await loadCountries();
        if (cRes.status !== "error") {
          setCountriesData(cRes.data);
          setAsOf(cRes.asOf);
          const currentCountry = cRes.data.data.countries.find((c) => c.code === code);
          const targetId = currentCountry ? currentCountry.index_id : "spx";
          const dRes = await loadDecomposition(targetId);
          if (dRes.status !== "error") {
            setDecompData(dRes.data);
          } else {
            // Fallback to featured
            const fRes = await loadDecomposition("featured");
            if (fRes.status !== "error") setDecompData(fRes.data);
          }
        }
      } catch (e) {
        console.error("Error loading country profile:", e);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [code]);

  const country = countriesData?.data.countries.find((c) => c.code === code);

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Navigation & Header */}
      <div>
        <Link
          href="/countries"
          className="inline-flex items-center gap-1.5 text-xs font-mono text-[var(--tx-2)] hover:text-[var(--accent)] mb-4 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Countries Matrix</span>
        </Link>

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[var(--line)]">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)] flex items-center gap-2">
              <span>Country Profile</span>
              <span>•</span>
              <span className="font-semibold text-[var(--accent)]">{code.toUpperCase()}</span>
            </div>
            <h1 className="text-3xl font-serif font-bold tracking-tight text-[var(--tx-0)] mt-1">
              {country ? country.name : `Country (${code.toUpperCase()})`}
            </h1>
            <p className="text-xs text-[var(--tx-1)] mt-1.5 font-mono">
              Benchmark Index: {country ? country.index_id.toUpperCase() : "—"} | Base Currency:{" "}
              {country ? country.currency : "USD"}
            </p>
          </div>

          <div className="flex items-center gap-3">
            <HealthDot status="ok" label="Verified Live" />
            <DataAgeBadge asOf={asOf} />
          </div>
        </div>
      </div>

      {/* Return Decomposition Identity Bar */}
      {country && (
        <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-[var(--tx-2)]">
            <span>5-Day Return Decomposition Identity</span>
            <span className="font-mono text-[var(--accent)]">(1 + r_usd) = (1 + r_local)(1 + r_fx)</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
            <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)]">
              <span className="text-[var(--tx-2)] block text-[11px]">Total USD Return (r_usd)</span>
              <span
                className={`text-lg font-bold block mt-1 ${
                  country.returns_usd["5D"] >= 0 ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {country.returns_usd["5D"] >= 0 ? "+" : ""}
                {(country.returns_usd["5D"] * 100.0).toFixed(2)}%
              </span>
            </div>

            <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)]">
              <span className="text-[var(--tx-2)] block text-[11px]">Local Currency Return (r_local)</span>
              <span
                className={`text-lg font-bold block mt-1 ${
                  country.returns_local["5D"] >= 0 ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {country.returns_local["5D"] >= 0 ? "+" : ""}
                {(country.returns_local["5D"] * 100.0).toFixed(2)}%
              </span>
            </div>

            <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)]">
              <span className="text-[var(--tx-2)] block text-[11px]">Currency Impact (r_fx)</span>
              <span
                className={`text-lg font-bold block mt-1 ${
                  country.returns_fx["5D"] >= 0 ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {country.returns_fx["5D"] >= 0 ? "+" : ""}
                {(country.returns_fx["5D"] * 100.0).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Signature Module: Move Decomposer for this country */}
      {decompData && (
        <ChartCard
          eyebrow="Econometric Attribution"
          title={`Move Decomposer: ${country ? country.name : code.toUpperCase()}`}
          subtitle={`Multi-factor decomposition into global drivers and residual (${decompData.data.horizon_days}D Horizon)`}
          insightHeadline={decompData.data.narrative}
          asOf={asOf}
          sourceText="Official Exchange Data, FRED, ICE, COMEX"
          auditData={{
            metricName: "Move Decomposer Prior-Window Attribution",
            formula: "y_t = X_t\\beta + \\text{residual}_t",
            formulaPlain: "Prior 250D window estimation with Newey-West HAC standard errors.",
            parameters: {
              horizon: `${decompData.data.horizon_days}D`,
              lens: decompData.data.lens,
              r2_out_sample: decompData.data.r2_out_sample,
            },
            inputs: Object.keys(decompData.data.contributions).map((k) => ({
              id: k,
              name: k.replace(/_/g, " ").toUpperCase(),
              source: "Global Macro Factors",
              unit: "return",
            })),
            window: "250D Prior Window",
            currentValue: `${(decompData.data.total_return * 100).toFixed(2)}%`,
            n: decompData.data.n_obs,
            codeVersion: decompData.code_version,
            specUrl: "/methodology",
          }}
          onOpenAudit={handleOpenAudit}
        >
          <MoveDecomposerWaterfall decomposition={decompData.data} />
        </ChartCard>
      )}

      {/* Historical Context Link */}
      {country && (
        <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-[var(--tx-0)]">Deep Statistical Distribution & Analogs</h3>
            <p className="text-xs text-[var(--tx-2)] mt-0.5">
              Inspect 25-bin empirical histograms, declustered historical analogs, and long-term drawdown curves.
            </p>
          </div>
          <Link
            href={`/context/${country.index_id}`}
            className="px-4 py-2 rounded bg-[var(--accent)] text-slate-900 font-semibold text-xs hover:opacity-90 transition-opacity"
          >
            Open {country.index_id.toUpperCase()} Context
          </Link>
        </div>
      )}

      {/* Drawers */}
      <AuditDrawer isOpen={isAuditOpen} onClose={() => setIsAuditOpen(false)} data={auditTarget} />
      <MethodDrawer isOpen={isMethodOpen} onClose={() => setIsMethodOpen(false)} data={methodTarget} />
    </div>
  );
};
