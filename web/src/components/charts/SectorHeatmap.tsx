// web/src/components/charts/SectorHeatmap.tsx
"use client";

import React, { useState, useMemo } from "react";
import { SectorItem } from "@/lib/data/schemas";
import { ArrowUpDown } from "lucide-react";

interface SectorHeatmapProps {
  sectors: SectorItem[];
  benchmarkReturns: Record<string, number>;
  benchmarkName: string;
}

type SortField = "name" | "weight" | "1D" | "5D" | "1M" | "3M" | "1Y" | "phase";

export const SectorHeatmap: React.FC<SectorHeatmapProps> = ({
  sectors,
  benchmarkReturns,
  benchmarkName,
}) => {
  const [sortField, setSortField] = useState<SortField>("weight");
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const sortedSectors = useMemo(() => {
    return [...sectors].sort((a, b) => {
      let valA: any;
      let valB: any;

      if (sortField === "name") {
        valA = a.name;
        valB = b.name;
        return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      } else if (sortField === "weight") {
        valA = a.weight;
        valB = b.weight;
      } else if (sortField === "phase") {
        valA = a.phase;
        valB = b.phase;
        return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      } else {
        // Horizon return
        valA = a.returns[sortField] ?? 0;
        valB = b.returns[sortField] ?? 0;
      }

      return sortAsc ? valA - valB : valB - valA;
    });
  }, [sectors, sortField, sortAsc]);

  // Color interpolation for historical percentile
  const getPercentileStyle = (pct: number) => {
    if (pct >= 85) {
      return { backgroundColor: "rgba(63, 182, 168, 0.28)", color: "#5eead4" };
    } else if (pct >= 65) {
      return { backgroundColor: "rgba(63, 182, 168, 0.12)", color: "#3fb6a8" };
    } else if (pct <= 15) {
      return { backgroundColor: "rgba(232, 118, 90, 0.28)", color: "#fca5a5" };
    } else if (pct <= 35) {
      return { backgroundColor: "rgba(232, 118, 90, 0.12)", color: "#e8765a" };
    }
    return { backgroundColor: "transparent", color: "#a9b4c4" };
  };

  const getPhaseBadge = (phase: string) => {
    switch (phase) {
      case "Leading":
        return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-[#3fb6a8]/15 text-[#3fb6a8]">Leading</span>;
      case "Weakening":
      case "Fading":
        return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-[#e6b94d]/15 text-[#e6b94d]">Weakening</span>;
      case "Lagging":
        return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-[#e8765a]/15 text-[#e8765a]">Lagging</span>;
      case "Improving":
      case "Recovering":
      default:
        return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-[#6ea8ff]/15 text-[#6ea8ff]">Improving</span>;
    }
  };

  const horizons = ["1D", "5D", "1M", "3M", "1Y"];

  return (
    <div className="space-y-2">
      {/* Legend */}
      <div className="flex flex-wrap items-center justify-between text-[11px] text-[#8b98a5] px-1">
        <div className="flex items-center space-x-2">
          <span>Heatmap shading indicates <strong className="text-[#e8edf5]">historical percentile rank</strong>:</span>
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#e8765a]/25 text-[#fca5a5]">Bottom 15%</span>
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#e8765a]/10 text-[#e8765a]">15%–35%</span>
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#1d2633] text-[#a9b4c4]">Neutral</span>
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#3fb6a8]/10 text-[#3fb6a8]">65%–85%</span>
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#3fb6a8]/25 text-[#5eead4]">Top 15%</span>
        </div>
      </div>

      <div className="overflow-x-auto rounded border border-[#1d2633]">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="bg-[#0c1017] border-b border-[#1d2633] text-[#8b98a5]">
              <th
                onClick={() => handleSort("name")}
                className="py-2.5 px-3 font-medium cursor-pointer hover:text-[#e8edf5]"
              >
                <div className="flex items-center space-x-1">
                  <span>Sector</span>
                  <ArrowUpDown className="w-3 h-3 opacity-60" />
                </div>
              </th>
              <th
                onClick={() => handleSort("weight")}
                className="py-2.5 px-3 font-medium text-right cursor-pointer hover:text-[#e8edf5]"
              >
                <div className="flex items-center justify-end space-x-1">
                  <span>Weight</span>
                  <ArrowUpDown className="w-3 h-3 opacity-60" />
                </div>
              </th>
              {horizons.map((h) => (
                <th
                  key={h}
                  onClick={() => handleSort(h as SortField)}
                  className="py-2.5 px-3 font-medium text-right cursor-pointer hover:text-[#e8edf5]"
                >
                  <div className="flex items-center justify-end space-x-1">
                    <span>{h}</span>
                    <ArrowUpDown className="w-3 h-3 opacity-60" />
                  </div>
                </th>
              ))}
              <th
                onClick={() => handleSort("phase")}
                className="py-2.5 px-3 font-medium text-center cursor-pointer hover:text-[#e8edf5]"
              >
                <div className="flex items-center justify-center space-x-1">
                  <span>Rotation Phase</span>
                  <ArrowUpDown className="w-3 h-3 opacity-60" />
                </div>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#161d28] font-mono">
            {/* Benchmark row */}
            <tr className="bg-[#121823]/60 font-semibold text-[#e8edf5]">
              <td className="py-2 px-3 font-sans">{benchmarkName} (Index)</td>
              <td className="py-2 px-3 text-right">100.0%</td>
              {horizons.map((h) => {
                const ret = (benchmarkReturns[h] ?? 0) * 100;
                return (
                  <td
                    key={h}
                    className={`py-2 px-3 text-right ${ret >= 0 ? "text-[#3fb6a8]" : "text-[#e8765a]"}`}
                  >
                    {ret >= 0 ? "+" : ""}
                    {ret.toFixed(2)}%
                  </td>
                );
              })}
              <td className="py-2 px-3 text-center text-[#8b98a5] font-sans text-[11px]">—</td>
            </tr>

            {/* Sector rows */}
            {sortedSectors.map((s) => {
              const weightPct = (s.weight * 100).toFixed(1);
              return (
                <tr key={s.id} className="hover:bg-[#121823]/40 transition">
                  <td className="py-2 px-3 font-sans text-[#e8edf5] font-medium">{s.name}</td>
                  <td className="py-2 px-3 text-right text-[#8b98a5]">{weightPct}%</td>
                  {horizons.map((h) => {
                    const ret = (s.returns?.[h] ?? 0) * 100;
                    const pct = s.percentiles?.[h] ?? 50;
                    const style = getPercentileStyle(pct);
                    return (
                      <td
                        key={h}
                        style={style}
                        className="py-2 px-3 text-right transition"
                        title={`${s.name} ${h}: ${ret >= 0 ? "+" : ""}${ret.toFixed(2)}% (Historical Percentile: ${pct.toFixed(0)}th)`}
                      >
                        <div>
                          <span>{ret >= 0 ? "+" : ""}{ret.toFixed(2)}%</span>
                          <span className="block text-[9px] opacity-75 font-sans">P{pct.toFixed(0)}</span>
                        </div>
                      </td>
                    );
                  })}
                  <td className="py-2 px-3 text-center">{getPhaseBadge(s.phase)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
