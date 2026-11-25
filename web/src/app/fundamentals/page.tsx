"use client";

import { useEffect, useState } from "react";
import { FundamentalsEnvelope } from "@/lib/data/schemas";
import { loadInventories } from "@/lib/data/client";
import { SeasonalBandChart } from "@/components/charts/SeasonalBandChart";
import { HealthDot } from "@/components/ui/HealthDot";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";

export default function FundamentalsPage() {
  const [data, setData] = useState<FundamentalsEnvelope | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadInventories()
      .then((res) => {
        if (res.status === "error") {
          setError(res.error || "Failed to load fundamentals inventory data");
        } else {
          setData(res.data);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) {
    return (
      <div className="rounded-[6px] border border-rose-500 bg-[#0c1017] p-5 text-sm text-rose-400">
        Failed to load Physical Fundamentals data: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-96 rounded-[6px] border border-[#1d2633] bg-[#0c1017] animate-pulse flex items-center justify-center font-mono text-xs text-[#6f7c90]">
        Loading EIA Weekly Petroleum & Natural Gas Storage Bands...
      </div>
    );
  }

  const fund = (data as any)?.data || data;

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-12">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1d2633]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[#6f7c90]">
            Physical Supply &amp; Storage
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[#e8edf5] mt-0.5">
            Energy Fundamentals: 5-Year Seasonal Inventory Bands
          </h1>
          <p className="text-xs text-[#8c9bb0] mt-1">
            US EIA commercial crude oil (ex-SPR), motor gasoline, distillates, and underground natural gas storage vs. 5-year seasonal envelopes.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot status="ok" label="EIA Socrata Synchronized" />
          <DataAgeBadge asOf={data.as_of || fund.as_of || ""} />
        </div>
      </div>

      <SeasonalBandChart data={data} />
    </div>
  );
}
