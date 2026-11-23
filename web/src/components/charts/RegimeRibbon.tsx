// web/src/components/charts/RegimeRibbon.tsx
"use client";

import React from "react";
import { RegimeRibbonPoint } from "@/lib/data/schemas";

interface Props {
  ribbon: RegimeRibbonPoint[];
  currentRegime: string;
  currentProbStress: number;
}

export function RegimeRibbon({ ribbon, currentRegime, currentProbStress }: Props) {
  const isStress = currentRegime === "stress";

  return (
    <div className="p-3.5 rounded-lg bg-[var(--bg-0)] border border-[var(--line)] space-y-2 font-mono">
      <div className="flex items-center justify-between text-xs">
        <span className="font-semibold text-[var(--tx-0)]">
          Historical Regime Probability Ribbon (Last {ribbon.length} Trading Days)
        </span>
        <div className="flex items-center gap-2">
          <span
            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
              isStress
                ? "bg-rose-950/70 border border-rose-800 text-rose-400"
                : "bg-emerald-950/70 border border-emerald-800 text-emerald-400"
            }`}
          >
            Current: {currentRegime.toUpperCase()} ({(currentProbStress * 100).toFixed(0)}%)
          </span>
          <span className="text-[11px] text-[var(--tx-2)]">Filtered Point-in-Time</span>
        </div>
      </div>

      <div className="flex h-5 w-full rounded overflow-hidden border border-[var(--line)]">
        {ribbon.map((pt, idx) => {
          const prob = pt.filtered_prob_stress;
          const bg =
            prob > 0.6
              ? "#ef4444"
              : prob > 0.4
              ? "#f59e0b"
              : prob > 0.2
              ? "#38bdf8"
              : "#10b981";
          return (
            <div
              key={idx}
              style={{
                backgroundColor: bg,
                opacity: 0.3 + prob * 0.7,
                width: `${100 / ribbon.length}%`,
              }}
              title={`${pt.date}: P(Stress)=${(prob * 100).toFixed(1)}%`}
              className="h-full transition-opacity hover:opacity-100 cursor-pointer"
            />
          );
        })}
      </div>

      <div className="flex justify-between text-[10px] text-[var(--tx-2)] pt-1">
        <span>{ribbon[0]?.date ?? "—"}</span>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded bg-[#10b981]" /> Calm (&lt;20%)
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded bg-[#38bdf8]" /> Neutral
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded bg-[#ef4444]" /> Stress (&gt;60%)
          </span>
        </div>
        <span>{ribbon[ribbon.length - 1]?.date ?? "—"}</span>
      </div>
    </div>
  );
}
