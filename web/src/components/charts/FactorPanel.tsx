// web/src/components/charts/FactorPanel.tsx
"use client";

import React from "react";
import { YieldCurveEnvelope } from "@/lib/data/schemas";

interface Props {
  data: YieldCurveEnvelope;
}

export function FactorPanel({ data }: Props) {
  const yc = (data as any).data || data;
  const factors = yc.factors || {
    level: 0,
    slope: 0,
    curvature: 0,
    rmse: 0,
    level_percentile: 50,
    slope_percentile: 50,
    curvature_percentile: 50,
  };

  return (
    <div className="space-y-3 font-mono">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-bold uppercase tracking-wider text-[var(--tx-0)]">
          Nelson-Siegel 3-Factor Decomposition
        </h4>
        <span className="text-[10px] text-[var(--tx-2)]">Diebold-Li formulation (λ = 0.0609)</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Factor L: Level */}
        <div className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] space-y-1">
          <div className="text-[10px] uppercase text-[var(--tx-2)] flex items-center justify-between">
            <span>Factor L (Level)</span>
            <span className="text-[9px] px-1 py-0.2 rounded bg-[var(--bg-1)]">Long-term anchor</span>
          </div>
          <div className="text-2xl font-bold text-[var(--tx-0)]">{factors.level.toFixed(2)}%</div>
          <div className="text-xs text-[var(--tx-2)] flex justify-between pt-1 border-t border-[var(--line)]">
            <span>Historical Percentile:</span>
            <strong className="text-[var(--accent)]">
              {(factors.level_percentile ?? 50).toFixed(0)}th
            </strong>
          </div>
        </div>

        {/* Factor S: Slope */}
        <div className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] space-y-1">
          <div className="text-[10px] uppercase text-[var(--tx-2)] flex items-center justify-between">
            <span>Factor S (Slope)</span>
            <span className="text-[9px] px-1 py-0.2 rounded bg-[var(--bg-1)]">Short vs Long</span>
          </div>
          <div
            className={`text-2xl font-bold ${
              factors.slope < 0 ? "text-rose-400" : "text-emerald-400"
            }`}
          >
            {factors.slope > 0 ? "+" : ""}
            {factors.slope.toFixed(2)}%
          </div>
          <div className="text-xs text-[var(--tx-2)] flex justify-between pt-1 border-t border-[var(--line)]">
            <span>Historical Percentile:</span>
            <strong className="text-[var(--accent)]">
              {(factors.slope_percentile ?? 50).toFixed(0)}th
            </strong>
          </div>
        </div>

        {/* Factor C: Curvature */}
        <div className="p-3.5 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] space-y-1">
          <div className="text-[10px] uppercase text-[var(--tx-2)] flex items-center justify-between">
            <span>Factor C (Curvature)</span>
            <span className="text-[9px] px-1 py-0.2 rounded bg-[var(--bg-1)]">Belly vs Wings</span>
          </div>
          <div className="text-2xl font-bold text-[var(--tx-0)]">{factors.curvature.toFixed(2)}%</div>
          <div className="text-xs text-[var(--tx-2)] flex justify-between pt-1 border-t border-[var(--line)]">
            <span>Fit RMSE:</span>
            <strong className="text-sky-400">{factors.rmse.toFixed(3)}%</strong>
          </div>
        </div>
      </div>
    </div>
  );
}
