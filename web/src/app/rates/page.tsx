"use client";

import { useEffect, useState } from "react";
import { YieldCurveEnvelope } from "@/lib/data/schemas";
import { loadYieldCurve } from "@/lib/data/client";
import { YieldHeatmap } from "@/components/charts/YieldHeatmap";
import { HealthDot } from "@/components/ui/HealthDot";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";

export default function RatesPage() {
  const [data, setData] = useState<YieldCurveEnvelope | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadYieldCurve()
      .then((res) => {
        if (res.status === "error") {
          setError(res.error || "Failed to load yield curve data");
        } else {
          setData(res.data);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) {
    return (
      <div className="rounded-[6px] border border-rose-500 bg-[#0c1017] p-5 text-sm text-rose-400">
        Failed to load Yield Curve Lab data: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-96 rounded-[6px] border border-[#1d2633] bg-[#0c1017] animate-pulse flex items-center justify-center font-mono text-xs text-[#6f7c90]">
        Loading US Treasury Yield Curve Lab & Nelson-Siegel Factors...
      </div>
    );
  }

  const yc = (data as any)?.data || data;

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-12">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1d2633]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Sovereign Rates &amp; Inflation Expectations
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[#e8edf5] mt-0.5">
            US Treasury Yield Curve Lab
          </h1>
          <p className="text-xs text-[#8c9bb0] mt-1">
            Nelson-Siegel 3-factor decomposition (Level, Slope, Curvature), benchmark curve inversions, and 10-year TIPS real vs. breakeven inflation anatomy.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot status="ok" label="Treasury & FRED Active" />
          <DataAgeBadge asOf={data.as_of || yc.as_of || ""} />
        </div>
      </div>

      <YieldHeatmap data={data} />
    </div>
  );
}
