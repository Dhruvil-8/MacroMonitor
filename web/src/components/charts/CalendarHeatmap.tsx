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
  unit?: string;
}

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export const CalendarHeatmap: React.FC<CalendarHeatmapProps> = ({ data, assetName, unit }) => {
  const isDiff = unit === "bps" || unit === "pct" || unit === "usd_bn" || unit === "usd_mn";

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

  const formatValue = (val: number | undefined) => {
    if (val === undefined || isNaN(val)) return "—";
    if (unit === "bps") {
      return `${val > 0 ? "+" : ""}${val.toFixed(0)} bps`;
    }
    if (unit === "pct") {
      return `${val > 0 ? "+" : ""}${val.toFixed(2)}%`;
    }
    if (unit === "usd_bn" || unit === "usd_mn") {
      return `${val > 0 ? "+" : ""}${val.toFixed(1)}${unit === "usd_bn" ? "B" : "M"}`;
    }
    return `${val > 0 ? "+" : ""}${(val * 100).toFixed(1)}%`;
  };

  const getColor = (val: number | undefined) => {
    if (val === undefined || isNaN(val)) return "rgba(255, 255, 255, 0.03)";
    const scale = unit === "bps" ? 0.02 : unit === "pct" ? 2.0 : unit === "usd_bn" ? 0.05 : 10;
    const alpha = Math.min(Math.abs(val) * scale, 0.75);
    if (val > 0) {
      return `rgba(63, 185, 80, ${alpha.toFixed(2)})`;
    } else if (val < 0) {
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
              <th className="py-2 px-2 text-right font-semibold">{isDiff ? "Full Year Net" : "Full Year"}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--line)]">
            {years.map((y) => {
              // If price series, compound geometrically: product of (1 + r) - 1
              // If difference series (rates/spreads), accumulate arithmetically: sum(diff)
              let annualOutcome: number | undefined;
              if (isDiff) {
                let sum = 0.0;
                let hasAny = false;
                for (let month = 1; month <= 12; month++) {
                  const r = matrix[y]?.[month];
                  if (r !== undefined) {
                    sum += r;
                    hasAny = true;
                  }
                }
                annualOutcome = hasAny ? sum : undefined;
              } else {
                let annualProd = 1.0;
                let hasAny = false;
                for (let month = 1; month <= 12; month++) {
                  const r = matrix[y]?.[month];
                  if (r !== undefined) {
                    annualProd *= (1.0 + r);
                    hasAny = true;
                  }
                }
                annualOutcome = hasAny ? annualProd - 1.0 : undefined;
              }

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
                        title={`${MONTH_NAMES[month - 1]} ${y}: ${formatValue(ret)}`}
                      >
                        {formatValue(ret)}
                      </td>
                    );
                  })}
                  <td className="py-1.5 px-2 text-right font-bold" style={{ color: (annualOutcome ?? 0) >= 0 ? "#3fb950" : "#f85149" }}>
                    {formatValue(annualOutcome)}
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
