// web/src/app/data-health/page.tsx
"use client";

import React, { useEffect, useState } from "react";
import { SourceStatusEnvelope } from "@/lib/data/schemas";
import { loadSourceStatus } from "@/lib/data/client";
import { SourceStatusTable } from "@/components/ui/SourceStatusTable";
import { HealthDot } from "@/components/ui/HealthDot";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { ShieldCheck, ServerCrash, Cpu } from "lucide-react";

export default function DataHealthPage() {
  const [data, setData] = useState<SourceStatusEnvelope | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        const res = await loadSourceStatus();
        if (res.status === "error") {
          setError(res.error);
        } else {
          setData(res.data);
        }
      } catch (err: any) {
        setError(err.message || "Failed to load data health");
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const asOf = data?.as_of || "";

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[var(--line)]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Infrastructure &amp; Integrity (P0)
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--tx-0)] mt-0.5">
            Data Source Health &amp; Reliability Registry
          </h1>
          <p className="text-xs text-[var(--tx-1)] mt-1">
            Real-time status of external ingestion adapters, circuit breaker telemetry, rate limits, and public licensing compliance.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot
            status={data?.data.overall_status === "healthy" ? "ok" : "degraded"}
            label={data?.data.overall_status === "healthy" ? "All Systems Operational" : "Degraded Feeds"}
          />
          <DataAgeBadge asOf={asOf} />
        </div>
      </div>

      {loading ? (
        <div className="p-12 text-center rounded-lg bg-[var(--bg-1)] border border-[var(--line)] font-mono text-xs text-[var(--tx-2)] animate-pulse">
          Loading Data Source Reliability Registry...
        </div>
      ) : error ? (
        <div className="p-6 rounded-lg bg-rose-950/20 border border-rose-900/50 text-rose-300 font-mono text-xs">
          Failed to load source health status: {error}
        </div>
      ) : data ? (
        <SourceStatusTable
          sources={data.data.sources}
          overallStatus={data.data.overall_status}
          totalTracked={data.data.total_tracked_series}
          totalHealthy={data.data.total_healthy_series}
        />
      ) : null}
    </div>
  );
}
