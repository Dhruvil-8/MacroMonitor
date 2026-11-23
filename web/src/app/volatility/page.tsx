"use client";

import { useEffect, useState } from "react";
import { ImpliedVolEnvelope } from "@/lib/data/schemas";
import { loadImpliedVol } from "@/lib/data/client";
import { TermStructureChart } from "@/components/charts/TermStructureChart";
import { HealthDot } from "@/components/ui/HealthDot";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";

export default function VolatilityPage() {
  const [data, setData] = useState<ImpliedVolEnvelope | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadImpliedVol()
      .then((res) => {
        if (res.status === "error") {
          setError(res.error || "Failed to load options-implied volatility data");
        } else {
          setData(res.data);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) {
    return (
      <div className="rounded-[6px] border border-rose-500 bg-[#0c1017] p-5 text-sm text-rose-400">
        Failed to load Options-Implied Volatility data: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-96 rounded-[6px] border border-[#1d2633] bg-[#0c1017] animate-pulse flex items-center justify-center font-mono text-xs text-[#6f7c90]">
        Loading Cboe Volatility Term Structure & Sentiment data...
      </div>
    );
  }

  const volData = (data as any)?.data || data;

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-12">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1d2633]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[#6f7c90]">
            Market Risk & Derivatives • Options-Implied Layer
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[#e8edf5] mt-0.5">
            Cboe Volatility Term Structure & Tail Sentiment
          </h1>
          <p className="text-xs text-[#8c9bb0] mt-1">
            Term-structure slope (9D to 6M), contango/backwardation regimes, VVIX, SKEW tail-hedging demand, and Variance Risk Premium (VRP) proxy.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot status="ok" label="Cboe Data Feed Active" />
          <DataAgeBadge asOf={volData.as_of || "2026-11-20"} />
        </div>
      </div>

      <TermStructureChart data={data} />
    </div>
  );
}
