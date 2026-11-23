"use client";

import { AnaloguePresetDetailData } from "@/lib/data/schemas";

interface AnalogueMatchTableProps {
  detail: AnaloguePresetDetailData;
  selectedRank: number;
  onSelectRank: (rank: number) => void;
}

export function AnalogueMatchTable({ detail, selectedRank, onSelectRank }: AnalogueMatchTableProps) {
  const activeMatch = detail.matches.find((m) => m.rank === selectedRank) ?? detail.matches[0];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
      {/* Today vs Selected Historical Match Feature Vector Comparison */}
      <div className="lg:col-span-5 rounded-[6px] border border-[#1d2633] bg-[#07090d] p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[#6f7c90]">
            Today vs Match #{activeMatch?.rank ?? 1} ({activeMatch?.date ?? "—"})
          </h3>
          <span className="text-[11px] font-mono tabular-nums text-[#38bdf8]">
            D = {activeMatch?.distance.toFixed(2) ?? "—"}
          </span>
        </div>

        <div className="flex flex-col gap-2.5">
          {detail.feature_labels.map((label, idx) => {
            const todayVal = detail.today_vector[idx] ?? 0;
            const matchVal = activeMatch?.features[idx] ?? 0;
            const maxAbs = Math.max(2.5, Math.abs(todayVal), Math.abs(matchVal));
            return (
              <div key={label} className="flex flex-col gap-1">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-[#e8edf5] font-medium truncate">{label}</span>
                  <span className="font-mono tabular-nums text-[#8c9bb0]">
                    Now {todayVal >= 0 ? "+" : ""}{todayVal.toFixed(2)}σ · Match {matchVal >= 0 ? "+" : ""}{matchVal.toFixed(2)}σ
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-1.5 h-2">
                  <div className="bg-[#1d2633] rounded-[2px] overflow-hidden flex items-center">
                    <div
                      className="h-full bg-[#38bdf8]"
                      style={{ width: `${Math.min(100, (Math.abs(todayVal) / maxAbs) * 100)}%` }}
                    />
                  </div>
                  <div className="bg-[#1d2633] rounded-[2px] overflow-hidden flex items-center">
                    <div
                      className="h-full bg-[#2dd4bf]"
                      style={{ width: `${Math.min(100, (Math.abs(matchVal) / maxAbs) * 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="text-[11px] font-mono text-[#6f7c90] pt-2 border-t border-[#1d2633]">
          Effective Independent Features: N_indep = {detail.n_indep.toFixed(2)} / {detail.n_features} · Shrinkage δ = {detail.shrinkage_delta.toFixed(3)} · Gap ≥ {detail.decluster_gap_days}D
        </div>
      </div>

      {/* Top K Declustered Matches Table */}
      <div className="lg:col-span-7 overflow-x-auto rounded-[6px] border border-[#1d2633]">
        <table className="w-full text-xs border-collapse">
          <caption className="sr-only">Top historical matches separated by at least 20 trading days</caption>
          <thead className="bg-[#07090d] border-b border-[#1d2633] text-[#6f7c90] text-left">
            <tr>
              <th className="py-2 px-3 font-medium">Rank</th>
              <th className="py-2 px-3 font-medium">Episode Date</th>
              <th className="py-2 px-3 font-medium text-right">Distance D</th>
              <th className="py-2 px-3 font-medium text-right">Similarity</th>
              <th className="py-2 px-3 font-medium text-right">SPX +3M</th>
              <th className="py-2 px-3 font-medium text-right">SPX 3M MDD</th>
              <th className="py-2 px-3 font-medium text-right">US10Y +3M</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#1d2633] font-mono tabular-nums">
            {detail.matches.map((m) => {
              const active = m.rank === (activeMatch?.rank ?? 1);
              return (
                <tr
                  key={m.rank}
                  onClick={() => onSelectRank(m.rank)}
                  className={`cursor-pointer transition-colors ${
                    active
                      ? "bg-[#38bdf8]/10"
                      : "hover:bg-[#121823]/60"
                  }`}
                >
                  <td className="py-1.5 px-3 text-[#6f7c90]">#{m.rank}</td>
                  <td className="py-1.5 px-3 text-[#e8edf5] font-medium">{m.date}</td>
                  <td className="py-1.5 px-3 text-right text-[#8c9bb0]">{m.distance.toFixed(2)}</td>
                  <td className="py-1.5 px-3 text-right text-[#38bdf8]">{m.similarity_score.toFixed(1)}%</td>
                  <td className={`py-1.5 px-3 text-right ${(m.spx_fwd_3m ?? 0) >= 0 ? "text-[#2dd4bf]" : "text-[#f87171]"}`}>
                    {m.spx_fwd_3m != null ? `${m.spx_fwd_3m >= 0 ? "+" : ""}${m.spx_fwd_3m.toFixed(1)}%` : "—"}
                  </td>
                  <td className="py-1.5 px-3 text-right text-[#f59e0b]">
                    {m.spx_mdd_3m != null ? `${m.spx_mdd_3m.toFixed(1)}%` : "—"}
                  </td>
                  <td className="py-1.5 px-3 text-right text-[#8c9bb0]">
                    {m.us10y_fwd_3m != null ? `${m.us10y_fwd_3m >= 0 ? "+" : ""}${m.us10y_fwd_3m.toFixed(2)}pp` : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
