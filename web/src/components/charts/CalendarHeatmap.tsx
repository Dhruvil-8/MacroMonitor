// web/src/components/charts/CalendarHeatmap.tsx
"use client";

import React, { useMemo } from "react";
import { AlertTriangle } from "lucide-react";

interface CalendarHeatmapItem {
  year: number;
  month: number;
  return: number;
}

interface CalendarHeatmapProps {
  data: CalendarHeatmapItem[];
  assetName: string;
}

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export const CalendarHeatmap: React.FC<CalendarHeatmapProps> = ({ data, assetName }) => {
  const { years, matrix } = useMemo(() => {
    if (!data || data.length === 0) return { years: [], matrix: {} };

    const yearSet = Array.from(new Set(data.map((d) => d.year))).sort((a, b) => b - a);
    const m: Record<number, Record<number, number>> = {};

    data.forEach((d) => {
      if (!m[d.year]) m[d.year] = {};
      m[d.year][d.month] = d.return;
    });

    return { years: yearSet, matrix: m };
  }, [data]);

  const getColor = (val: number | undefined) => {
    if (val === undefined || isNaN(val)) return "rgba(255, 255, 255, 0.03)";
    if (val > 0) {
      const alpha = Math.min(Math.abs(val) * 10, 0.75);
      return `rgba(63, 185, 80, ${alpha.toFixed(2)})`;
    } else if (val < 0) {
      const alpha = Math.min(Math.abs(val) * 10, 0.75);
      return `rgba(248, 81, 73, ${alpha.toFixed(2)})`;
    }
    return "rgba(255, 255, 255, 0.05)";
  };

  return (
    <div className="flex flex-col gap-3">
      {/* Non-negotiable Statistical Warning Banner */}
      <div className="flex items-center gap-2 p-2.5 rounded bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs font-mono">
        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
        <span>
          <strong>Methodological Notice:</strong> Calendar month patterns may reflect chance; small annual sample sizes preclude statistical significance. Displayed strictly for descriptive context.
        </span>
      </div>

      <div className="overflow-x-auto text-xs font-mono">
        <table className="w-full text-center border-collapse">
          <thead>
            <tr className="border-b border-[var(--line)] text-[var(--tx-2)]">
              <th className="py-2 px-2 text-left font-semibold">Year</th>
              {MONTH_NAMES.map((m) => (
                <th key={m} className="py-2 px-1 text-center font-normal">{m}</th>
              ))}
              <th className="py-2 px-2 text-right font-semibold">Full Year</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--line)]">
            {years.map((y) => {
              // Annual cumulative return: product of (1 + r) - 1
              let annualProd = 1.0;
              let hasAny = false;
              for (let month = 1; month <= 12; month++) {
                const r = matrix[y]?.[month];
                if (r !== undefined) {
                  annualProd *= (1.0 + r);
                  hasAny = true;
                }
              }
              const annualReturn = hasAny ? annualProd - 1.0 : undefined;

              return (
                <tr key={y} className="hover:bg-white/5 transition-colors">
                  <td className="py-1.5 px-2 text-left font-bold text-[var(--tx-0)]">{y}</td>
                  {Array.from({ length: 12 }, (_, i) => i + 1).map((month) => {
                    const ret = matrix[y]?.[month];
                    return (
                      <td
                        key={month}
                        className="py-1.5 px-1 font-mono text-[11px]"
                        style={{ backgroundColor: getColor(ret) }}
                        title={`${MONTH_NAMES[month - 1]} ${y}: ${ret !== undefined ? (ret * 100).toFixed(2) + "%" : "—"}`}
                      >
                        {ret !== undefined ? `${ret > 0 ? "+" : ""}${(ret * 100).toFixed(1)}%` : "—"}
                      </td>
                    );
                  })}
                  <td className="py-1.5 px-2 text-right font-bold" style={{ color: (annualReturn ?? 0) >= 0 ? "#3fb950" : "#f85149" }}>
                    {annualReturn !== undefined ? `${annualReturn >= 0 ? "+" : ""}${(annualReturn * 100).toFixed(1)}%` : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
