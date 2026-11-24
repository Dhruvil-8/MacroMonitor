"use client";

import { useEffect, useState } from "react";
import { CotContract, PositioningEnvelope } from "@/lib/data/schemas";
import { loadPositioning } from "@/lib/data/client";
import { PositioningChart } from "@/components/charts/PositioningChart";
import { PositioningLadder } from "@/components/charts/PositioningLadder";
import { HealthDot } from "@/components/ui/HealthDot";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";

export default function PositioningPage() {
  const [data, setData] = useState<PositioningEnvelope | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedContractId, setSelectedContractId] = useState<string>("sp500_emini");

  useEffect(() => {
    loadPositioning()
      .then((res) => {
        if (res.status === "error") {
          setError(res.error || "Failed to load positioning data");
        } else {
          setData(res.data);
          const first = (res.data as any)?.data?.contracts?.[0] || (res.data as any)?.contracts?.[0];
          if (first?.id) {
            setSelectedContractId(first.id);
          }
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) {
    return (
      <div className="rounded-[6px] border border-rose-500 bg-[#0c1017] p-5 text-sm text-rose-400">
        Failed to load CFTC Positioning data: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-96 rounded-[6px] border border-[#1d2633] bg-[#0c1017] animate-pulse flex items-center justify-center font-mono text-xs text-[#6f7c90]">
        Loading CFTC Commitments of Traders data...
      </div>
    );
  }

  const contracts: CotContract[] = (data as any)?.data?.contracts || (data as any)?.contracts || [];

  return (
    <div className="flex flex-col gap-8 max-w-7xl mx-auto pb-12">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1d2633]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[#6f7c90]">
            Macro Flows & Derivatives • CFTC Commitments of Traders
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[#e8edf5] mt-0.5">
            Futures Positioning & Crowding (COT)
          </h1>
          <p className="text-xs text-[#8c9bb0] mt-1">
            Weekly non-commercial speculative positioning as % of open interest. Quantifies crowding extremes using 156-week expanding percentiles and 2-week hysteresis.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot status="ok" label="CFTC Socrata Synchronized" />
          <DataAgeBadge asOf={data.as_of || (data as any).data?.as_of || ""} />
        </div>
      </div>

      {/* Module P2: Positioning Ladder Overview */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-[#e8edf5] tracking-wide font-mono uppercase flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[var(--accent)]" />
            Positioning Percentile Ladder (Cross-Asset)
          </h2>
          <span className="text-[11px] font-mono text-[#6f7c90]">
            Click any row to inspect historical series below
          </span>
        </div>
        <PositioningLadder
          contracts={contracts}
          selectedContractId={selectedContractId}
          onSelectContract={(c) => setSelectedContractId(c.id)}
        />
      </section>

      {/* Detailed Historical Chart for Selected Contract */}
      <section className="space-y-3 pt-4 border-t border-[#1d2633]">
        <PositioningChart
          data={data}
          selectedContractId={selectedContractId}
          onSelectContractId={(id) => setSelectedContractId(id)}
        />
      </section>
    </div>
  );
}
