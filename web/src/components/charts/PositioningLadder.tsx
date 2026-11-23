// web/src/components/charts/PositioningLadder.tsx
"use client";

import React, { useState } from "react";
import { CotContract } from "@/lib/data/schemas";
import { AlertTriangle, Clock, Filter, ArrowUpRight, ArrowDownRight } from "lucide-react";

interface PositioningLadderProps {
  contracts: CotContract[];
  onSelectContract?: (contract: CotContract) => void;
  selectedContractId?: string;
}

export const PositioningLadder: React.FC<PositioningLadderProps> = ({
  contracts,
  onSelectContract,
  selectedContractId,
}) => {
  const [assetClassFilter, setAssetClassFilter] = useState<string>("all");

  const filteredContracts = contracts.filter((c) => {
    if (assetClassFilter === "all") return true;
    return (c.asset_class || "general") === assetClassFilter;
  });

  return (
    <div className="space-y-4">
      {/* Filter and Legend */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 bg-[var(--bg-0)] rounded-md border border-[var(--line)] font-mono text-xs">
        <div className="flex items-center gap-2">
          <span className="text-[var(--tx-2)]">Asset Class:</span>
          <div className="flex rounded border border-[var(--line)] overflow-hidden">
            {[
              { id: "all", label: "All Contracts" },
              { id: "equities", label: "Equities" },
              { id: "rates", label: "Rates" },
              { id: "currencies", label: "Currencies" },
              { id: "commodities", label: "Commodities" },
            ].map((f) => (
              <button
                key={f.id}
                type="button"
                onClick={() => setAssetClassFilter(f.id)}
                className={`px-2.5 py-0.5 text-[11px] transition-colors ${
                  assetClassFilter === f.id
                    ? "bg-[var(--accent)] text-white font-medium"
                    : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-[10px] text-[var(--tx-2)]">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[var(--accent)] border border-white" />
            <span>Now (Latest)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-blue-400 opacity-60" />
            <span>-4 Weeks</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-slate-500 opacity-40" />
            <span>-13 Weeks</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="px-1 py-0.2 rounded bg-rose-950/40 text-rose-400 text-[9px] border border-rose-800/40 font-bold">
              &gt;95 / &lt;5
            </span>
            <span>Extremes</span>
          </div>
        </div>
      </div>

      {/* Ladder Table / Rows */}
      <div className="rounded-lg bg-[var(--bg-0)] border border-[var(--line)] divide-y divide-[var(--line)] overflow-hidden">
        {filteredContracts.map((c) => {
          const pctNow = c.percentile ?? 50.0;
          const pct4w = c.percentile_4w_ago ?? pctNow;
          const pct13w = c.percentile_13w_ago ?? pct4w;
          const isSelected = selectedContractId === c.id;

          const isExtreme = c.is_extreme || pctNow >= 95.0 || pctNow <= 5.0;

          return (
            <div
              key={c.id}
              onClick={() => onSelectContract && onSelectContract(c)}
              className={`p-3.5 transition-colors cursor-pointer ${
                isSelected
                  ? "bg-[var(--accent)]/10 border-l-2 border-l-[var(--accent)]"
                  : "hover:bg-[var(--bg-1)]/50"
              }`}
            >
              <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
                {/* Contract Info */}
                <div className="md:col-span-4">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-xs text-[var(--tx-0)] font-mono">
                      {c.name}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-[var(--bg-2)] border border-[var(--line)] text-[var(--tx-2)] font-mono uppercase">
                      {c.asset_class || "general"}
                    </span>
                    {isExtreme && (
                      <span className="text-[9px] px-1.5 py-0.2 rounded bg-rose-950/50 text-rose-400 border border-rose-800/50 font-mono font-bold uppercase">
                        {pctNow >= 95 ? "Crowded Long" : "Crowded Short"}
                      </span>
                    )}
                  </div>
                  <div className="text-[10px] text-[var(--tx-2)] mt-0.5 font-mono flex items-center gap-2">
                    <span>Net % OI: <b>{c.net_pct_oi ? `${(c.net_pct_oi * 100).toFixed(1)}%` : "N/A"}</b></span>
                    <span>•</span>
                    <span>Net: {c.net_speculative ? `${(c.net_speculative / 1000).toFixed(0)}k contracts` : "N/A"}</span>
                  </div>
                </div>

                {/* Percentile Strip Bar (Col 5) */}
                <div className="md:col-span-5 relative py-2">
                  <div className="h-2 w-full bg-[var(--bg-1)] rounded-full relative overflow-hidden border border-[var(--line)]">
                    {/* Shaded extreme bands <5% and >95% */}
                    <div className="absolute left-0 top-0 bottom-0 w-[5%] bg-rose-500/25" />
                    <div className="absolute right-0 top-0 bottom-0 w-[5%] bg-rose-500/25" />
                    {/* 50% centerline marker */}
                    <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-[var(--line)]" />
                  </div>

                  {/* Dot for 13 weeks ago */}
                  {c.percentile_13w_ago !== undefined && c.percentile_13w_ago !== null && (
                    <div
                      style={{ left: `${Math.min(99, Math.max(1, pct13w))}%` }}
                      className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-slate-500 opacity-40 pointer-events-none"
                      title={`13w ago: ${pct13w.toFixed(0)}th percentile`}
                    />
                  )}

                  {/* Dot for 4 weeks ago */}
                  {c.percentile_4w_ago !== undefined && c.percentile_4w_ago !== null && (
                    <div
                      style={{ left: `${Math.min(99, Math.max(1, pct4w))}%` }}
                      className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full bg-blue-400 opacity-70 pointer-events-none"
                      title={`4w ago: ${pct4w.toFixed(0)}th percentile`}
                    />
                  )}

                  {/* Dot for Now */}
                  <div
                    style={{ left: `${Math.min(99, Math.max(1, pctNow))}%` }}
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5 rounded-full bg-[var(--accent)] border-2 border-white shadow-[0_0_8px_rgba(232,118,90,0.6)] pointer-events-none"
                    title={`Current: ${pctNow.toFixed(1)}th percentile`}
                  />
                </div>

                {/* Percentile Stats & Changes (Col 3) */}
                <div className="md:col-span-3 flex items-center justify-end gap-3 text-right font-mono text-xs">
                  <div>
                    <div className="text-sm font-bold text-[var(--tx-0)]">
                      {pctNow.toFixed(0)}
                      <span className="text-[10px] text-[var(--tx-2)] font-normal">th pct</span>
                    </div>
                    <div className="text-[10px] text-[var(--tx-2)]">
                      Z: {c.z_score ? `${c.z_score > 0 ? "+" : ""}${c.z_score.toFixed(2)}σ` : "N/A"}
                    </div>
                  </div>

                  {/* 4W Change */}
                  <div className="text-[10px] pl-2 border-l border-[var(--line)]">
                    <div className="text-[var(--tx-2)]">4W Δ</div>
                    <div
                      className={`font-semibold ${
                        (c.chg_4w_pct_oi ?? 0) >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]"
                      }`}
                    >
                      {c.chg_4w_pct_oi !== undefined && c.chg_4w_pct_oi !== null
                        ? `${c.chg_4w_pct_oi > 0 ? "+" : ""}${(c.chg_4w_pct_oi * 100).toFixed(1)}%`
                        : "N/A"}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Lag Footer */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between text-[11px] font-mono text-[var(--tx-2)] pt-2 border-t border-[var(--line)]">
        <div className="flex items-center gap-1.5">
          <Clock className="w-3.5 h-3.5 text-[var(--tx-3)]" />
          <span>Point-in-Time Lag: <b>Positions as of Tuesday, published Friday</b> (CFTC Socrata feed)</span>
        </div>
        <span>Lookback: 156-week expanding percentile • Minimum 104 observations</span>
      </div>
    </div>
  );
};
