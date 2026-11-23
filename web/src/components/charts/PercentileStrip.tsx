// web/src/components/charts/PercentileStrip.tsx
// C20: Percentile strip with "you are here" marker and historical distribution markings (Section 5.1)
"use client";

import React from "react";

interface PercentileStripProps {
  label: string;
  percentile: number; // 0 to 100
  currentValue: string;
  p10?: string;
  p50?: string;
  p90?: string;
}

export const PercentileStrip: React.FC<PercentileStripProps> = ({
  label,
  percentile,
  currentValue,
  p10 = "10th Pct",
  p50 = "Median (50th)",
  p90 = "90th Pct",
}) => {
  const clamped = Math.max(0, Math.min(100, percentile));

  return (
    <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3 font-mono">
      <div className="flex items-center justify-between text-xs">
        <span className="text-[var(--tx-0)] font-semibold">{label}</span>
        <div className="flex items-center gap-2">
          <span className="text-[var(--tx-2)]">Level:</span>
          <span className="text-[var(--tx-0)] font-bold">{currentValue}</span>
          <span className="px-1.5 py-0.5 rounded bg-[var(--bg-2)] border border-[var(--line)] text-[var(--accent)] text-[10px]">
            {clamped.toFixed(0)}th Pct
          </span>
        </div>
      </div>

      {/* Horizontal Gradient Strip */}
      <div className="relative pt-4 pb-2">
        <div className="h-3 rounded-full bg-gradient-to-r from-emerald-500/80 via-amber-500/70 to-rose-500/90 w-full relative">
          {/* Tick 10% */}
          <div className="absolute top-0 bottom-0 left-[10%] w-[1px] bg-black/60" />
          {/* Tick 50% */}
          <div className="absolute top-0 bottom-0 left-[50%] w-[1px] bg-black/60" />
          {/* Tick 90% */}
          <div className="absolute top-0 bottom-0 left-[90%] w-[1px] bg-black/60" />
        </div>

        {/* "You Are Here" Marker */}
        <div
          className="absolute top-0 -translate-x-1/2 flex flex-col items-center pointer-events-none transition-all duration-300"
          style={{ left: `${clamped}%` }}
        >
          <div className="px-1.5 py-0.5 rounded bg-white text-slate-900 font-bold text-[9px] shadow uppercase tracking-wider">
            Today
          </div>
          <div className="w-0 h-0 border-l-[3px] border-l-transparent border-r-[3px] border-r-transparent border-t-[4px] border-t-white" />
        </div>
      </div>

      {/* Reference Ticks */}
      <div className="flex justify-between text-[10px] text-[var(--tx-2)]">
        <span>0% (Min)</span>
        <span className="hidden sm:inline">{p10}</span>
        <span>{p50}</span>
        <span className="hidden sm:inline">{p90}</span>
        <span>100% (Max)</span>
      </div>
    </div>
  );
};
