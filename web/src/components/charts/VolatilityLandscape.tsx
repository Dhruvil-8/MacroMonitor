// web/src/components/charts/VolatilityLandscape.tsx
"use client";

import React, { useState } from "react";
import { VolLandscapeItem } from "@/lib/data/schemas";
import { ArrowUpDown, ShieldCheck } from "lucide-react";
import { AuditData } from "../ui/AuditDrawer";

interface VolatilityLandscapeProps {
  landscape: VolLandscapeItem[];
  onOpenAudit?: (data: AuditData) => void;
}

type SortField = "vol20_percentile" | "vol_trend" | "vol10" | "vol20" | "vol60" | "asset";

export const VolatilityLandscape: React.FC<VolatilityLandscapeProps> = ({
  landscape,
  onOpenAudit,
}) => {
  const [sortField, setSortField] = useState<SortField>("vol20_percentile");
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const sortedData = [...landscape].sort((a, b) => {
    const valA = a[sortField];
    const valB = b[sortField];
    if (typeof valA === "string" && typeof valB === "string") {
      return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    }
    return sortAsc ? Number(valA) - Number(valB) : Number(valB) - Number(valA);
  });

  // Diverging color for percentile
  const getPercentileBg = (pct: number) => {
    if (pct >= 90) return "rgba(232, 118, 90, 0.25)"; // extreme high vol (coral)
    if (pct >= 75) return "rgba(230, 185, 77, 0.20)"; // high vol (amber)
    if (pct <= 25) return "rgba(63, 182, 168, 0.20)"; // low vol (teal)
    return "transparent";
  };

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto rounded border border-[var(--line)]">
        <table className="w-full text-xs text-left">
          <thead className="bg-[var(--bg-0)] text-[var(--tx-2)] uppercase font-mono text-[11px] border-b border-[var(--line)]">
            <tr>
              <th
                onClick={() => handleSort("asset")}
                className="py-2.5 px-3 cursor-pointer hover:text-[var(--tx-0)] select-none"
              >
                <div className="flex items-center gap-1">
                  <span>Asset</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-2.5 px-3">Class</th>
              <th
                onClick={() => handleSort("vol10")}
                className="py-2.5 px-3 text-right cursor-pointer hover:text-[var(--tx-0)] select-none"
              >
                10D Vol
              </th>
              <th
                onClick={() => handleSort("vol20")}
                className="py-2.5 px-3 text-right cursor-pointer hover:text-[var(--tx-0)] select-none"
              >
                20D Vol
              </th>
              <th
                onClick={() => handleSort("vol60")}
                className="py-2.5 px-3 text-right cursor-pointer hover:text-[var(--tx-0)] select-none"
              >
                60D Vol
              </th>
              <th
                onClick={() => handleSort("vol_trend")}
                className="py-2.5 px-3 text-right cursor-pointer hover:text-[var(--tx-0)] select-none"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>Trend</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th
                onClick={() => handleSort("vol20_percentile")}
                className="py-2.5 px-3 text-right cursor-pointer hover:text-[var(--tx-0)] select-none"
              >
                <div className="flex items-center justify-end gap-1">
                  <span>Percentile</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-2.5 px-3 text-right">Audit</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--line)] font-mono text-[11px]">
            {sortedData.slice(0, 15).map((row) => {
              const trendPos = row.vol_trend > 0;
              return (
                <tr
                  key={row.asset}
                  className="hover:bg-[var(--bg-2)]/50 transition-colors"
                >
                  <td className="py-2.5 px-3 font-sans font-medium text-[var(--tx-0)]">
                    <div>{row.name}</div>
                    <div className="text-[10px] text-[var(--tx-2)] font-mono">
                      {row.asset.toUpperCase()}
                    </div>
                  </td>
                  <td className="py-2.5 px-3 text-[var(--tx-2)] uppercase text-[10px]">
                    {row.group.replace("_", " ")}
                  </td>
                  <td className="py-2.5 px-3 text-right text-[var(--tx-1)]">
                    {(row.vol10 * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3 text-right font-medium text-[var(--tx-0)]">
                    {(row.vol20 * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3 text-right text-[var(--tx-1)]">
                    {(row.vol60 * 100).toFixed(1)}%
                  </td>
                  <td
                    className={`py-2.5 px-3 text-right font-medium ${
                      trendPos ? "text-[var(--neg)]" : "text-[var(--pos)]"
                    }`}
                  >
                    {trendPos ? "+" : ""}
                    {(row.vol_trend * 100).toFixed(1)}%
                  </td>
                  <td
                    className="py-2.5 px-3 text-right font-medium"
                    style={{ backgroundColor: getPercentileBg(row.vol20_percentile) }}
                  >
                    {row.vol20_percentile.toFixed(0)}th
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    {onOpenAudit && (
                      <button
                        onClick={() =>
                          onOpenAudit({
                            metricName: `${row.name} Realized Volatility`,
                            formula: "\\sigma_{w, t} = \\text{std}(r_{t-w+1 \\dots t}) \\times \\sqrt{252}",
                            formulaPlain: "Annualized sample standard deviation of daily log returns over trailing 20 business days.",
                            parameters: { window: 20, annualization_factor: "sqrt(252)", ddof: 1 },
                            inputs: [{ id: row.asset, name: row.name, source: "Yahoo/FRED", unit: "index" }],
                            window: "20 business days",
                            currentValue: `${(row.vol20 * 100).toFixed(1)}%`,
                            percentile: row.vol20_percentile,
                            n: row.n,
                            nEff: row.n_eff,
                            specUrl: "/methodology",
                          })
                        }
                        className="p-1 rounded hover:bg-[var(--bg-0)] text-[var(--tx-2)] hover:text-[var(--accent)]"
                        title="Audit volatility metric"
                      >
                        <ShieldCheck className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="flex justify-between items-center text-[11px] text-[var(--tx-2)] font-mono px-1">
        <span>Showing top 15 sorted by {sortField}</span>
        <span>All volatility figures annualized via sqrt(252)</span>
      </div>
    </div>
  );
};
