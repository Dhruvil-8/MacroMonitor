// web/src/components/charts/OiBuildupMap.tsx
"use client";

import React from "react";
import { OiBuildupItem } from "@/lib/data/schemas";

interface OiBuildupMapProps {
  items: OiBuildupItem[];
  pcr: number;
}

export const OiBuildupMap: React.FC<OiBuildupMapProps> = ({ items, pcr }) => {
  const getBadgeColor = (buildup: string) => {
    switch (buildup) {
      case "Long Buildup":
        return "bg-emerald-950/60 text-emerald-400 border-emerald-500/30";
      case "Short Buildup":
        return "bg-rose-950/60 text-rose-400 border-rose-500/30";
      case "Short Covering":
        return "bg-amber-950/60 text-amber-400 border-amber-500/30";
      case "Long Unwinding":
        return "bg-purple-950/60 text-purple-400 border-purple-500/30";
      default:
        return "bg-[#131b26] text-[#6f7c90] border-[#232f42]";
    }
  };

  return (
    <div className="space-y-4">
      {/* PCR Banner */}
      <div className="flex items-center justify-between p-3 rounded-lg bg-[var(--bg-2)] border border-[var(--line)]">
        <div>
          <span className="text-[10px] font-mono uppercase tracking-wider text-[var(--tx-2)] block">
            Nifty Put-Call Ratio (PCR)
          </span>
          <div className="text-lg font-mono font-bold text-[var(--tx-0)] mt-0.5">
            {(pcr ?? 1.08).toFixed(2)}{" "}
            <span className="text-xs font-normal text-[var(--tx-2)]">
              {(pcr ?? 1.08) > 1.3 ? "(High / Resistance)" : (pcr ?? 1.08) < 0.7 ? "(Low / Oversold)" : "(Neutral Band)"}
            </span>
          </div>
        </div>
        <div className="text-right text-[11px] font-mono text-[var(--tx-1)]">
          Total Put OI / Total Call OI
        </div>
      </div>

      {/* 4-Quadrant Description Pills */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
        <div className="p-2 rounded bg-emerald-950/20 border border-emerald-500/20 text-emerald-400">
          <div className="font-semibold">Long Buildup</div>
          <div className="text-[10px] text-emerald-500/80">ΔP &gt; 0, ΔOI &gt; 0</div>
        </div>
        <div className="p-2 rounded bg-rose-950/20 border border-rose-500/20 text-rose-400">
          <div className="font-semibold">Short Buildup</div>
          <div className="text-[10px] text-rose-500/80">ΔP &lt; 0, ΔOI &gt; 0</div>
        </div>
        <div className="p-2 rounded bg-amber-950/20 border border-amber-500/20 text-amber-400">
          <div className="font-semibold">Short Covering</div>
          <div className="text-[10px] text-amber-500/80">ΔP &gt; 0, ΔOI &lt; 0</div>
        </div>
        <div className="p-2 rounded bg-purple-950/20 border border-purple-500/20 text-purple-400">
          <div className="font-semibold">Long Unwinding</div>
          <div className="text-[10px] text-purple-500/80">ΔP &lt; 0, ΔOI &lt; 0</div>
        </div>
      </div>

      {/* Instruments Table */}
      <div className="overflow-x-auto rounded-lg border border-[var(--line)]">
        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-[var(--bg-0)] text-[var(--tx-2)] uppercase text-[10px] border-b border-[var(--line)]">
            <tr>
              <th className="py-2.5 px-3">Contract</th>
              <th className="py-2.5 px-3 text-right">Price Δ%</th>
              <th className="py-2.5 px-3 text-right">OI Δ%</th>
              <th className="py-2.5 px-3 text-center">Buildup Classification</th>
              <th className="py-2.5 px-3 text-right">PCR</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--line)] bg-[var(--bg-1)]">
            {items.map((item) => {
              const pChg = item.price_change_pct ?? (item as any).price_chg_pct ?? 0;
              const oiChg = item.oi_change_pct ?? (item as any).oi_chg_pct ?? 0;
              const pcrVal = item.pcr ?? 1.0;
              return (
                <tr key={item.symbol} className="hover:bg-[var(--bg-2)] transition-colors">
                  <td className="py-2.5 px-3 font-semibold text-[var(--tx-0)]">{item.symbol}</td>
                  <td className={`py-2.5 px-3 text-right font-semibold ${pChg >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    {pChg >= 0 ? "+" : ""}{pChg.toFixed(2)}%
                  </td>
                  <td className={`py-2.5 px-3 text-right font-semibold ${oiChg >= 0 ? "text-[var(--accent)]" : "text-[var(--tx-2)]"}`}>
                    {oiChg >= 0 ? "+" : ""}{oiChg.toFixed(2)}%
                  </td>
                  <td className="py-2.5 px-3 text-center">
                    <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold border ${getBadgeColor(item.buildup)}`}>
                      {item.buildup}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-right text-[var(--tx-1)]">{pcrVal.toFixed(2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
