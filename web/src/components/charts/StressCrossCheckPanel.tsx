// web/src/components/charts/StressCrossCheckPanel.tsx
"use client";

import React, { useState } from "react";
import { BenchmarkData } from "@/lib/data/schemas";
import { ShieldCheck, Calendar, Activity, AlertCircle, Compass } from "lucide-react";

interface StressCrossCheckPanelProps {
  data: BenchmarkData;
}

export const StressCrossCheckPanel: React.FC<StressCrossCheckPanelProps> = ({ data }) => {
  const [selectedTab, setSelectedTab] = useState<"agreement" | "episodes">("agreement");

  return (
    <div className="space-y-4">
      {/* Tab Switcher & Context Strip */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 bg-[var(--bg-0)] rounded-md border border-[var(--line)]">
        <div className="flex rounded border border-[var(--line)] overflow-hidden font-mono text-xs">
          <button
            type="button"
            onClick={() => setSelectedTab("agreement")}
            className={`px-3 py-1 transition-colors ${
              selectedTab === "agreement"
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
            }`}
          >
            External Benchmarks
          </button>
          <button
            type="button"
            onClick={() => setSelectedTab("episodes")}
            className={`px-3 py-1 transition-colors ${
              selectedTab === "episodes"
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-1)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
            }`}
          >
            Crisis Episodes Shading ({data.episodes?.length ?? 0})
          </button>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono">
          <div className="flex items-center gap-1.5 text-[var(--tx-2)]">
            <Compass className="w-3.5 h-3.5 text-[var(--accent)]" />
            <span>Economic Policy Uncertainty:</span>
            <strong className="text-[var(--tx-0)]">{(data.policy_uncertainty_level ?? 125.4).toFixed(1)}</strong>
            <span className="text-[10px] text-[var(--tx-2)]">({(data.policy_uncertainty_percentile ?? 68.0).toFixed(0)}th pct)</span>
          </div>
        </div>
      </div>

      {selectedTab === "agreement" ? (
        <div className="space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {data.stress_cross_checks.map((cc) => (
              <div
                key={cc.benchmark_name}
                className="p-3.5 rounded-lg bg-[var(--bg-0)] border border-[var(--line)] space-y-2.5 font-mono text-xs"
              >
                <div className="flex items-center justify-between border-b border-[var(--line)] pb-2">
                  <div className="font-semibold text-[var(--tx-0)] text-sm">
                    {cc.benchmark_name}
                  </div>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-2)] text-[var(--tx-2)]">
                    N_eff = {cc.effective_n}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center pt-1">
                  <div className="p-2 rounded bg-[var(--bg-1)] border border-[var(--line)]">
                    <div className="text-[10px] text-[var(--tx-2)] uppercase">Pearson r</div>
                    <div className="text-base font-bold text-[var(--tx-0)] mt-0.5">
                      {cc.pearson_corr > 0 ? "+" : ""}{cc.pearson_corr.toFixed(2)}
                    </div>
                  </div>
                  <div className="p-2 rounded bg-[var(--bg-1)] border border-[var(--line)]">
                    <div className="text-[10px] text-[var(--tx-2)] uppercase">Spearman ρ</div>
                    <div className="text-base font-bold text-[var(--tx-0)] mt-0.5">
                      {cc.spearman_rank_corr > 0 ? "+" : ""}{cc.spearman_rank_corr.toFixed(2)}
                    </div>
                  </div>
                  <div className="p-2 rounded bg-[var(--bg-1)] border border-[var(--line)]">
                    <div className="text-[10px] text-[var(--tx-2)] uppercase">Lead / Lag</div>
                    <div className="text-base font-bold text-[var(--accent)] mt-0.5">
                      {cc.best_lead_lag === 0
                        ? "Sync"
                        : cc.best_lead_lag > 0
                        ? `+${cc.best_lead_lag}w Lead`
                        : `${cc.best_lead_lag}w Lag`}
                    </div>
                  </div>
                </div>

                {/* Cross-correlation spectrum spark */}
                {cc.cross_correlation && cc.cross_correlation.length > 0 && (
                  <div className="pt-2">
                    <div className="text-[10px] text-[var(--tx-2)] mb-1 flex items-center justify-between">
                      <span>Cross-Correlation Spectrum (±4 Weeks):</span>
                      <span className="text-[9px] text-[var(--tx-3)]">Peak: {cc.best_lead_lag}w</span>
                    </div>
                    <div className="grid grid-cols-9 gap-1 text-center text-[10px]">
                      {cc.cross_correlation.map((pt) => (
                        <div
                          key={pt.lag_weeks}
                          className={`p-1 rounded border ${
                            pt.lag_weeks === cc.best_lead_lag
                              ? "bg-[var(--accent)]/15 border-[var(--accent)] text-[var(--accent)] font-bold"
                              : "bg-[var(--bg-1)] border-[var(--line)] text-[var(--tx-2)]"
                          }`}
                        >
                          <div className="text-[8px] text-[var(--tx-3)]">{pt.lag_weeks > 0 ? `+${pt.lag_weeks}` : pt.lag_weeks}w</div>
                          <div>{pt.correlation.toFixed(2)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="text-[11px] font-mono text-[var(--tx-2)] p-2.5 rounded bg-[var(--bg-0)]/50 border border-[var(--line)]">
            <b>Methodological Note:</b> Independent verification demonstrates whether our synthetic daily stress composite faithfully leads or mirrors the weekly official St. Louis Fed &amp; Chicago Fed financial condition indices without forward bias.
          </div>
        </div>
      ) : (
        /* Episodes Table */
        <div className="rounded-lg bg-[var(--bg-0)] border border-[var(--line)] overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-[var(--line)] bg-[var(--bg-1)] text-[var(--tx-2)] uppercase text-[10px] tracking-wider">
                  <th className="py-2 px-3">Episode Name &amp; Period</th>
                  <th className="py-2 px-3">Category</th>
                  <th className="py-2 px-3">Peak Stress Date</th>
                  <th className="py-2 px-3">Peak Score</th>
                  <th className="py-2 px-3">Elevation Ratio</th>
                  <th className="py-2 px-3 text-right">Lead / Lag to Official Peak</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--line)]">
                {data.episodes.map((ep) => (
                  <tr key={ep.id} className="hover:bg-[var(--bg-2)]/30 transition-colors">
                    <td className="py-2.5 px-3">
                      <div className="font-semibold text-[var(--tx-0)]">{ep.name}</div>
                      <div className="text-[10px] text-[var(--tx-2)] mt-0.5">
                        {ep.start_date} to {ep.end_date}
                      </div>
                    </td>
                    <td className="py-2.5 px-3">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-2)] border border-[var(--line)] text-[var(--tx-2)] capitalize">
                        {ep.category.replace("_", " ")}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-[var(--tx-1)]">{ep.peak_stress_date}</td>
                    <td className="py-2.5 px-3 font-bold text-[var(--neg)]">{ep.peak_stress_score.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-[var(--tx-0)] font-medium">
                      {ep.elevated_ratio.toFixed(1)}x normal
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span
                        className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          ep.lead_lag_to_official_peak_days <= 0
                            ? "bg-emerald-950/40 text-emerald-400 border border-emerald-800/40"
                            : "bg-amber-950/40 text-amber-400 border border-amber-800/40"
                        }`}
                      >
                        {ep.lead_lag_to_official_peak_days === 0
                          ? "Synchronous (0d)"
                          : ep.lead_lag_to_official_peak_days < 0
                          ? `${Math.abs(ep.lead_lag_to_official_peak_days)}d Lead`
                          : `+${ep.lead_lag_to_official_peak_days}d Lag`}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
