// web/src/components/charts/DriverSensitivityHeatmap.tsx
"use client";

import React from "react";
import { DriverTargetMatrixRow } from "@/lib/data/schemas";
import { AlertTriangle, CheckCircle2 } from "lucide-react";

interface DriverSensitivityHeatmapProps {
  rows: DriverTargetMatrixRow[];
  selectedTargetId: string;
  onSelectTarget: (targetId: string) => void;
  selectedDriverId: string;
  onSelectDriver: (driverId: string) => void;
}

const FACTOR_ORDER = [
  { id: "row_equity", name: "ROW Equity", sub: "% / %" },
  { id: "brent", name: "Brent Oil", sub: "% / %" },
  { id: "copper", name: "Copper", sub: "% / %" },
  { id: "gold", name: "Gold", sub: "% / %" },
  { id: "dxy", name: "US Dollar (DXY)", sub: "% / %" },
  { id: "us10y", name: "US 10Y Yield", sub: "per 1 bp" },
];

export const DriverSensitivityHeatmap: React.FC<DriverSensitivityHeatmapProps> = ({
  rows,
  selectedTargetId,
  onSelectTarget,
  selectedDriverId,
  onSelectDriver,
}) => {
  const getCellBg = (driverId: string, beta: number, stability: number, significant: boolean) => {
    const isUnstable = stability < 0.70 || !significant;
    const opacity = isUnstable ? 0.35 : 0.85;

    // US 10Y is in basis points, so its beta is in decimal-return-per-bp (e.g. 0.0001 = 1% per 100 bps)
    const scaleFactor = driverId === "us10y" ? 2500 : 25;
    const mag = Math.abs(beta) * scaleFactor;

    if (mag < 0.005) {
      return "rgba(29, 38, 51, 0.4)";
    }

    if (beta > 0) {
      // Teal / positive
      const intensity = Math.min(1.0, Math.max(0.15, mag));
      return `rgba(63, 182, 168, ${intensity * opacity})`;
    } else {
      // Coral / negative
      const intensity = Math.min(1.0, Math.max(0.15, mag));
      return `rgba(232, 118, 90, ${intensity * opacity})`;
    }
  };

  return (
    <div className="space-y-3">
      {/* Legend & Guidance */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-[var(--tx-2)] px-1">
        <div className="flex flex-wrap items-center gap-3 text-[11px]">
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-3 rounded bg-[var(--pos)]/80 inline-block"></span>
            <span>Positive Beta</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-3 rounded bg-[var(--neg)]/80 inline-block"></span>
            <span>Negative Beta</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-3 h-3 rounded border border-dashed border-[var(--warn)] inline-block opacity-60"></span>
            <span>Unstable / Insignificant (&lt;70% or 0 in CI)</span>
          </div>
        </div>
        <div className="text-[11px] font-mono text-[var(--tx-3)]">
          Click any row or cell to inspect detail
        </div>
      </div>

      <div className="overflow-x-auto rounded border border-[var(--line)]">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="bg-[var(--bg-0)] border-b border-[var(--line)] text-[var(--tx-2)]">
              <th className="py-2.5 px-3 font-medium">Target Economy</th>
              <th className="py-2.5 px-2 text-right font-mono font-medium">R²</th>
              {FACTOR_ORDER.map((f) => (
                <th
                  key={f.id}
                  onClick={() => onSelectDriver(f.id)}
                  className={`py-2 px-3 font-medium text-center cursor-pointer transition ${
                    selectedDriverId === f.id
                      ? "text-[var(--accent)] font-semibold bg-[var(--bg-2)]"
                      : "hover:text-[var(--tx-0)]"
                  }`}
                >
                  <div>{f.name}</div>
                  <div className="text-[10px] font-normal text-[var(--tx-3)] font-mono">{f.sub}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--line)] font-mono">
            {rows.map((r) => {
              const isSelectedTarget = r.target_id === selectedTargetId;
              return (
                <tr
                  key={r.target_id}
                  onClick={() => onSelectTarget(r.target_id)}
                  className={`cursor-pointer transition ${
                    isSelectedTarget
                      ? "bg-[var(--bg-2)]/80"
                      : "hover:bg-[var(--bg-0)]/50"
                  }`}
                >
                  <td className="py-2 px-3 font-sans font-medium text-[var(--tx-0)] flex items-center justify-between">
                    <span>{r.target_name}</span>
                    {isSelectedTarget && (
                      <span className="text-[10px] font-mono text-[var(--accent)] uppercase font-semibold">
                        active
                      </span>
                    )}
                  </td>
                  <td className="py-2 px-2 text-right text-[var(--tx-2)]">
                    {(r.r2 * 100).toFixed(1)}%
                  </td>
                  {FACTOR_ORDER.map((f) => {
                    const item = r.drivers[f.id];
                    if (!item) {
                      return (
                        <td key={f.id} className="py-2 px-3 text-center text-[var(--tx-3)]">
                          —
                        </td>
                      );
                    }
                    const isUnstable = item.stability < 0.70 || !item.significant;
                    const bg = getCellBg(f.id, item.beta, item.stability, item.significant);
                    const isCellActive = isSelectedTarget && selectedDriverId === f.id;

                    return (
                      <td
                        key={f.id}
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectTarget(r.target_id);
                          onSelectDriver(f.id);
                        }}
                        style={{ backgroundColor: bg }}
                        className={`py-2 px-3 text-center transition relative ${
                          isUnstable ? "border-dashed" : ""
                        } ${isCellActive ? "ring-2 ring-[var(--accent)] font-bold z-10" : ""}`}
                        title={`${r.target_name} vs ${f.name}\nBeta: ${item.beta.toFixed(4)}${f.id === "us10y" ? ` (${(item.beta * 10000 >= 0 ? "+" : "")}${(item.beta * 10000).toFixed(1)}% per 100 bps yield change)` : ""}\n95% CI: [${item.ci_lower.toFixed(4)}, ${item.ci_upper.toFixed(4)}]\nStability: ${(item.stability * 100).toFixed(0)}%\nPrior: ${item.prior} (${item.prior_aligned ? "Aligned" : "Divergent"})`}
                      >
                        <div className="flex items-center justify-center space-x-1">
                          <span
                            className={
                              item.beta >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]"
                            }
                          >
                            {item.beta >= 0 ? "+" : ""}
                            {f.id === "us10y" ? item.beta.toFixed(4) : item.beta.toFixed(3)}
                          </span>
                          {isUnstable && (
                            <span title="Unstable or statistically insignificant coefficient">
                              <AlertTriangle className="w-2.5 h-2.5 text-[var(--warn)] opacity-80" />
                            </span>
                          )}
                          {!item.prior_aligned && !isUnstable && (
                            <span
                              className="text-[9px] text-[var(--warn)] font-bold"
                              title="Diverges from pre-registered economic prior"
                            >
                              !
                            </span>
                          )}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
