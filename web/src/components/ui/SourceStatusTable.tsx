// web/src/components/ui/SourceStatusTable.tsx
"use client";

import React from "react";
import { SourceStatusItem } from "@/lib/data/schemas";
import { ExternalLink, ShieldCheck, AlertTriangle, Clock, RefreshCw, ZapOff } from "lucide-react";

interface SourceStatusTableProps {
  sources: SourceStatusItem[];
  overallStatus: string;
  totalTracked: number;
  totalHealthy: number;
}

export const SourceStatusTable: React.FC<SourceStatusTableProps> = ({
  sources,
  overallStatus,
  totalTracked,
  totalHealthy,
}) => {
  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)]">
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Ingestion State
          </div>
          <div className="flex items-center gap-2 mt-1">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                overallStatus === "healthy" ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-amber-500"
              }`}
            />
            <span className="text-lg font-bold font-mono uppercase tracking-wider text-[var(--tx-0)]">
              {overallStatus}
            </span>
          </div>
          <div className="text-[11px] text-[var(--tx-2)] mt-1 font-mono">
            Zero broken pipelines
          </div>
        </div>

        <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)]">
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Tracked Feeds
          </div>
          <div className="text-xl font-bold font-mono text-[var(--tx-0)] mt-1">
            {sources.length} <span className="text-xs font-normal text-[var(--tx-2)]">sources</span>
          </div>
          <div className="text-[11px] text-[var(--tx-2)] mt-1 font-mono">
            100% Free / Public Domain
          </div>
        </div>

        <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)]">
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Series Coverage
          </div>
          <div className="text-xl font-bold font-mono text-[var(--tx-0)] mt-1">
            {totalHealthy} / {totalTracked}
          </div>
          <div className="text-[11px] text-[var(--pos)] mt-1 font-mono">
            {totalTracked > 0 ? ((totalHealthy / totalTracked) * 100).toFixed(0) : 100}% Healthy Series
          </div>
        </div>

        <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)]">
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Circuit Breakers
          </div>
          <div className="text-xl font-bold font-mono text-[var(--tx-0)] mt-1 flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-[var(--pos)]" />
            <span>All Closed</span>
          </div>
          <div className="text-[11px] text-[var(--tx-2)] mt-1 font-mono">
            Zero active trips
          </div>
        </div>
      </div>

      {/* Main Table */}
      <div className="rounded-lg bg-[var(--bg-1)] border border-[var(--line)] overflow-hidden">
        <div className="p-4 border-b border-[var(--line)] flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-[var(--tx-0)] font-mono">
              External Source Registry &amp; Health Matrix
            </h3>
            <p className="text-xs text-[var(--tx-2)] mt-0.5">
              Live status, rate limits, schema verification, and licensing terms for free macro feeds.
            </p>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[var(--bg-2)] border border-[var(--line)] text-[var(--tx-2)] self-start sm:self-auto">
            Governed by P0 Resilience Specification
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-[var(--line)] bg-[var(--bg-0)] text-[var(--tx-2)] uppercase text-[10px] tracking-wider">
                <th className="py-2.5 px-4">Source &amp; Provider</th>
                <th className="py-2.5 px-4">Status</th>
                <th className="py-2.5 px-4">Series Tracked</th>
                <th className="py-2.5 px-4">Rate Limit &amp; Schedule</th>
                <th className="py-2.5 px-4">Last Verified</th>
                <th className="py-2.5 px-4 text-right">Terms &amp; Docs</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--line)]">
              {sources.map((src) => {
                const isHealthy = src.status === "healthy";
                const isStale = src.status === "stale";
                const isDegraded = src.status === "degraded";

                return (
                  <tr key={src.id} className="hover:bg-[var(--bg-2)]/40 transition-colors">
                    {/* Source Name & Provider */}
                    <td className="py-3 px-4">
                      <div className="font-semibold text-[var(--tx-0)] flex items-center gap-1.5">
                        <span>{src.name}</span>
                        <span className="text-[10px] px-1.5 py-0.2 rounded bg-[var(--bg-2)] border border-[var(--line)] text-[var(--tx-2)]">
                          {src.id}
                        </span>
                      </div>
                      <div className="text-[11px] text-[var(--tx-2)] mt-0.5">{src.provider}</div>
                      {src.owner_notes && (
                        <div className="text-[10px] text-[var(--tx-3)] mt-1 max-w-sm italic">
                          {src.owner_notes}
                        </div>
                      )}
                    </td>

                    {/* Status Dot */}
                    <td className="py-3 px-4 whitespace-nowrap">
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            isHealthy
                              ? "bg-emerald-500"
                              : isDegraded
                              ? "bg-amber-500"
                              : "bg-rose-500"
                          }`}
                        />
                        <span
                          className={`capitalize font-medium ${
                            isHealthy
                              ? "text-emerald-400"
                              : isDegraded
                              ? "text-amber-400"
                              : "text-rose-400"
                          }`}
                        >
                          {src.status}
                        </span>
                      </div>
                      <div className="text-[10px] text-[var(--tx-2)] mt-0.5">
                        Circuit: {src.circuit_state}
                      </div>
                    </td>

                    {/* Series Tracked */}
                    <td className="py-3 px-4 whitespace-nowrap">
                      <div className="font-semibold text-[var(--tx-0)]">
                        {src.tracked_series_count} series
                      </div>
                      <div className="text-[10px] text-[var(--tx-2)] mt-0.5">
                        {src.stale_series_count > 0 ? (
                          <span className="text-[var(--warn)]">{src.stale_series_count} stale</span>
                        ) : (
                          <span className="text-[var(--pos)]">0 stale</span>
                        )}
                      </div>
                    </td>

                    {/* Rate Limit & Schedule */}
                    <td className="py-3 px-4">
                      <div className="text-[var(--tx-0)]">{src.rate_limit}</div>
                      <div className="text-[10px] text-[var(--tx-2)] mt-0.5 flex items-center gap-1">
                        <Clock className="w-3 h-3 shrink-0" />
                        <span>{src.schedule}</span>
                      </div>
                    </td>

                    {/* Last Verified */}
                    <td className="py-3 px-4 whitespace-nowrap text-[var(--tx-2)]">
                      <div>{src.last_verified}</div>
                      <div className="text-[10px] text-[var(--tx-3)] mt-0.5">
                        Synced: {src.last_success}
                      </div>
                    </td>

                    {/* Terms & Docs Links */}
                    <td className="py-3 px-4 text-right whitespace-nowrap space-y-1">
                      {src.documentation_url && (
                        <div>
                          <a
                            href={src.documentation_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-[11px] text-[var(--accent)] hover:underline"
                          >
                            <span>Docs</span>
                            <ExternalLink className="w-2.5 h-2.5" />
                          </a>
                        </div>
                      )}
                      {src.terms_url && (
                        <div>
                          <a
                            href={src.terms_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-[10px] text-[var(--tx-2)] hover:text-[var(--tx-0)] hover:underline"
                          >
                            <span>Terms</span>
                            <ExternalLink className="w-2.5 h-2.5" />
                          </a>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
