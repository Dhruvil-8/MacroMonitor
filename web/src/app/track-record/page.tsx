'use client';

import React, { useEffect, useState } from 'react';
import { loadTrackRecord } from '@/lib/data/client';
import type { TrackRecordData } from '@/lib/data/schemas';
import { TrackRecordBars } from '@/components/charts/TrackRecordBars';

export default function TrackRecordPage() {
  const [data, setData] = useState<TrackRecordData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterFamily, setFilterFamily] = useState<string>('all');

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await loadTrackRecord();
        if (res.status !== 'error') {
          setData(res.data.data);
        }
      } catch (err) {
        console.error('Failed to load track record:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading || !data) {
    return (
      <div className="space-y-6">
        <div className="p-8 rounded-lg bg-[#0c1017] border border-[#1d2633] text-center font-mono text-xs text-[#6f7c90]">
          Loading Pre-Registered Track Record & Empirical Validation...
        </div>
      </div>
    );
  }

  const netEdge = (data.overall_hit_rate - data.overall_base_rate).toFixed(1);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#1d2633] pb-4">
        <div>
          <div className="flex items-center space-x-2.5">
            <h1 className="text-xl font-bold text-[#e8edf5] tracking-tight">
              Track Record & Validation (P11)
            </h1>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#16202c] border border-[#26354a] text-[#38bdf8]">
              Pre-Registered & Declustered
            </span>
          </div>
          <p className="text-xs text-[#8993a4] mt-1">
            Empirical out-of-sample forward evaluation of observation rules vs unconditioned base rates
          </p>
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono">
          <div className="px-3 py-1.5 rounded bg-[#111620] border border-[#1d2633] text-[#a9b4c4]">
            As of: <span className="text-[#e8edf5] font-semibold">{data.as_of}</span>
          </div>
        </div>
      </div>

      {/* Credibility & Anti-Spurious Mandate Callout */}
      <div className="p-4 rounded-lg bg-[#0d141f] border border-[#1d2d42] text-xs space-y-2">
        <div className="flex items-center space-x-2 text-[#38bdf8] font-semibold">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <span>Pre-Registration & Anti-Hindsight Protocol (docs/spec/track_record.md)</span>
        </div>
        <p className="text-[#94a3b8] leading-relaxed">
          {data.statement}
        </p>
      </div>

      {/* Top Stat Blocks */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-lg bg-[#0c1017] border border-[#1d2633]">
          <div className="text-[11px] font-mono text-[#6f7c90]">Declustered Episodes</div>
          <div className="text-2xl font-mono font-bold text-[#e8edf5] mt-1">
            {data.total_episodes}
          </div>
          <div className="text-[10px] text-[#8993a4] mt-0.5">10-day separation filter</div>
        </div>

        <div className="p-4 rounded-lg bg-[#0c1017] border border-[#1d2633]">
          <div className="text-[11px] font-mono text-[#6f7c90]">Overall Hit Rate</div>
          <div className="text-2xl font-mono font-bold text-[#38bdf8] mt-1">
            {data.overall_hit_rate}%
          </div>
          <div className="text-[10px] text-[#8993a4] mt-0.5">20-day forward horizon</div>
        </div>

        <div className="p-4 rounded-lg bg-[#0c1017] border border-[#1d2633]">
          <div className="text-[11px] font-mono text-[#6f7c90]">Unconditional Base Rate</div>
          <div className="text-2xl font-mono font-bold text-[#94a3b8] mt-1">
            {data.overall_base_rate}%
          </div>
          <div className="text-[10px] text-[#8993a4] mt-0.5">Random unalerted days</div>
        </div>

        <div className="p-4 rounded-lg bg-[#0c1017] border border-[#1d2633]">
          <div className="text-[11px] font-mono text-[#6f7c90]">Net Statistical Edge</div>
          <div className="text-2xl font-mono font-bold text-[#4ade80] mt-1">
            +{netEdge}%
          </div>
          <div className="text-[10px] text-[#8993a4] mt-0.5">Empirical - Base rate</div>
        </div>
      </div>

      {/* Visual Bars & Calibration */}
      <TrackRecordBars rules={data.rules} calibration={data.calibration} />

      {/* Detailed Rule Catalog Table */}
      <div className="rounded-lg bg-[#0c1017] border border-[#1d2633] p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#1d2633] pb-3">
          <div>
            <h3 className="text-sm font-semibold text-[#e8edf5]">
              Rule-by-Rule Validation Ledger
            </h3>
            <p className="text-xs text-[#8993a4] mt-0.5">
              Strictly segregated sample periods with negative findings honestly disclosed
            </p>
          </div>
        </div>

        <div className="overflow-x-auto rounded border border-[#1d2633]">
          <table className="w-full text-left font-mono text-xs">
            <thead className="bg-[#121824] text-[#8993a4] border-b border-[#1d2633]">
              <tr>
                <th className="py-2 px-3">Rule Name & ID</th>
                <th className="py-2 px-3">Pre-Registered Hypothesis</th>
                <th className="py-2 px-3 text-right">Episodes</th>
                <th className="py-2 px-3 text-right">Hit Rate</th>
                <th className="py-2 px-3 text-right">Base Rate</th>
                <th className="py-2 px-3 text-right">Edge (95% CI)</th>
                <th className="py-2 px-3 text-right">Lead</th>
                <th className="py-2 px-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1d2633] text-[#e8edf5]">
              {data.rules.map((r) => {
                const isValidated = r.status.includes('Validated');
                const isContext = r.status.includes('Context Only');
                return (
                  <tr key={r.rule_id} className="hover:bg-[#161f2e] transition">
                    <td className="py-2 px-3 font-semibold text-[#89a4c7]">
                      <div>{r.rule_name}</div>
                      <div className="text-[10px] text-[#6f7c90] font-normal">{r.rule_id}</div>
                    </td>
                    <td className="py-2 px-3 text-[11px] text-[#cbd5e1] max-w-xs">
                      {r.hypothesis}
                    </td>
                    <td className="py-2 px-3 text-right text-[#a9b4c4]">{r.n_episodes}</td>
                    <td className="py-2 px-3 text-right font-bold text-[#38bdf8]">{r.hit_rate}%</td>
                    <td className="py-2 px-3 text-right text-[#94a3b8]">{r.base_rate}%</td>
                    <td className="py-2 px-3 text-right">
                      <span className={r.edge_pct > 0 ? 'text-[#4ade80] font-bold' : 'text-[#f87171]'}>
                        {r.edge_pct > 0 ? `+${r.edge_pct}` : r.edge_pct}%
                      </span>
                      <div className="text-[10px] text-[#64748b]">
                        [{r.ci_lower}%, {r.ci_upper}%]
                      </div>
                    </td>
                    <td className="py-2 px-3 text-right text-[#cbd5e1]">{r.median_lead_days}d</td>
                    <td className="py-2 px-3 text-center">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-[10px] border ${
                          isValidated
                            ? 'bg-[#102a20] border-[#2d8a64] text-[#4ade80]'
                            : isContext
                            ? 'bg-[#1e2430] border-[#334155] text-[#94a3b8]'
                            : 'bg-[#1c2438] border-[#3b82f6] text-[#93c5fd]'
                        }`}
                      >
                        {r.status}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Rule Version Changelog */}
      <div className="rounded-lg bg-[#0c1017] border border-[#1d2633] p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-[#1d2633] pb-3">
          <div>
            <h3 className="text-sm font-semibold text-[#e8edf5]">
              Observation Rule Version Changelog
            </h3>
            <p className="text-xs text-[#8993a4] mt-0.5">
              Guaranteed parameter immutability: rules are never adjusted retrospectively without version increments
            </p>
          </div>
          <span className="text-xs font-mono text-[#6f7c90]">Git-Verified</span>
        </div>

        <div className="divide-y divide-[#1d2633]">
          {data.changelog.map((entry, idx) => (
            <div key={idx} className="py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <span className="font-mono text-[#38bdf8] font-bold">{entry.version}</span>
                  <span className="text-[#64748b]">•</span>
                  <span className="font-mono text-[#cbd5e1]">{entry.rule_id}</span>
                  <span className="text-[#64748b]">•</span>
                  <span className="font-mono text-[11px] text-[#6f7c90]">{entry.date}</span>
                </div>
                <div className="text-[#94a3b8]">{entry.change_summary}</div>
              </div>
              <div className="font-mono text-[11px] text-[#64748b] bg-[#111620] px-2 py-1 rounded border border-[#1d2633] shrink-0">
                commit: {entry.commit_hash}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
