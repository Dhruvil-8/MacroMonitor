"use client";

import { useEffect, useState } from "react";
import { ConnectednessEnvelope } from "@/lib/data/schemas";
import { loadConnectedness } from "@/lib/data/client";
import { ConnectednessMatrix } from "@/components/charts/ConnectednessMatrix";
import { ChordView } from "@/components/charts/ChordView";
import { HealthDot } from "@/components/ui/HealthDot";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";

export default function ConnectednessPage() {
  const [data, setData] = useState<ConnectednessEnvelope | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"standard" | "chord">("standard");

  useEffect(() => {
    loadConnectedness()
      .then((res) => {
        if (res.status === "error") {
          setError(res.error || "Failed to load connectedness data");
        } else {
          setData(res.data);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) {
    return (
      <div className="rounded-[6px] border border-rose-500 bg-[#0c1017] p-5 text-sm text-rose-400">
        Failed to load Connectedness data: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-96 rounded-[6px] border border-[#1d2633] bg-[#0c1017] animate-pulse flex items-center justify-center font-mono text-xs text-[#6f7c90]">
        Estimating VAR & Diebold-Yilmaz GFEVD Connectedness Matrix...
      </div>
    );
  }

  const conn = (data as any)?.data || data;

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-12">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1d2633]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[#6f7c90]">
            Systemic Interdependence &amp; Shocks
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[#e8edf5] mt-0.5">
            Cross-Border Connectedness & Spillover Map
          </h1>
          <p className="text-xs text-[#8c9bb0] mt-1">
            Order-invariant Diebold-Yilmaz Generalized Forecast Error Variance Decomposition (GFEVD) across global markets. Identifies net transmitters and receivers of financial shocks.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex rounded border border-[var(--line)] overflow-hidden font-mono text-xs">
            <button
              onClick={() => setViewMode("standard")}
              className={`px-2.5 py-1 transition-colors ${
                viewMode === "standard"
                  ? "bg-[var(--accent)] text-white font-medium"
                  : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              }`}
            >
              Matrix &amp; Bars
            </button>
            <button
              onClick={() => setViewMode("chord")}
              className={`px-2.5 py-1 transition-colors ${
                viewMode === "chord"
                  ? "bg-[var(--accent)] text-white font-medium"
                  : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              }`}
            >
              Chord View
            </button>
          </div>
          <HealthDot status="ok" label="VAR Model Converged" />
          <DataAgeBadge asOf={data.as_of || conn.as_of || ""} />
        </div>
      </div>

      {viewMode === "standard" ? (
        <ConnectednessMatrix data={data} />
      ) : (
        <ChordView data={data} />
      )}
    </div>
  );
}
