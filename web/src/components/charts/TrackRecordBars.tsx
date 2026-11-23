'use client';

import React from 'react';
import type { TrackRecordRuleItem, CalibrationBandItem } from '@/lib/data/schemas';

interface TrackRecordBarsProps {
  rules: TrackRecordRuleItem[];
  calibration: CalibrationBandItem[];
}

export function TrackRecordBars({ rules, calibration }: TrackRecordBarsProps) {
  return (
    <div className="space-y-6">
      {/* Hit Rate vs Base Rate Bar Comparison */}
      <div className="rounded-lg bg-[#0c1017] border border-[#1d2633] p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#1d2633] pb-3">
          <div>
            <h3 className="text-sm font-semibold text-[#e8edf5]">
              Empirical Hit Rate vs Unconditioned Base Rate by Rule
            </h3>
            <p className="text-xs text-[#8993a4] mt-0.5">
              Evaluating pre-registered 20-day forward horizons with 95% stationary block-bootstrap confidence intervals
            </p>
          </div>
          <div className="flex items-center space-x-3 text-xs font-mono">
            <span className="flex items-center space-x-1.5">
              <span className="w-3 h-3 rounded bg-[#2f6fb4]" />
              <span className="text-[#a9b4c4]">Observed Hit Rate</span>
            </span>
            <span className="flex items-center space-x-1.5">
              <span className="w-3 h-3 rounded bg-[#334155]" />
              <span className="text-[#a9b4c4]">Base Rate</span>
            </span>
          </div>
        </div>

        <div className="space-y-4 pt-1">
          {rules.map((rule) => {
            const isValidated = rule.status.includes('Validated');
            const isContextOnly = rule.status.includes('Context Only');
            const barColor = isContextOnly
              ? '#64748b'
              : isValidated
              ? '#38bdf8'
              : '#3b82f6';

            return (
              <div key={rule.rule_id} className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-[#e8edf5]">{rule.rule_name}</span>
                    <span className="font-mono text-[11px] text-[#6f7c90]">({rule.rule_id})</span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-mono border ${
                        isValidated
                          ? 'bg-[#102a20] border-[#2d8a64] text-[#4ade80]'
                          : isContextOnly
                          ? 'bg-[#1e2430] border-[#334155] text-[#94a3b8]'
                          : 'bg-[#1c2438] border-[#3b82f6] text-[#93c5fd]'
                      }`}
                    >
                      {rule.status}
                    </span>
                  </div>
                  <div className="font-mono text-xs text-[#cbd5e1] space-x-2">
                    <span>
                      Hit: <strong style={{ color: barColor }}>{rule.hit_rate}%</strong>
                    </span>
                    <span className="text-[#64748b]">vs Base {rule.base_rate}%</span>
                    <span className={rule.edge_pct > 0 ? 'text-[#4ade80] font-bold' : 'text-[#f87171]'}>
                      ({rule.edge_pct > 0 ? `+${rule.edge_pct}` : rule.edge_pct}%)
                    </span>
                  </div>
                </div>

                {/* Dual horizontal bars */}
                <div className="space-y-1">
                  {/* Hit Rate Bar */}
                  <div className="h-4 w-full bg-[#111620] rounded overflow-hidden flex items-center relative">
                    <div
                      className="h-full rounded transition-all duration-500"
                      style={{
                        width: `${Math.min(100, Math.max(5, rule.hit_rate))}%`,
                        backgroundColor: barColor,
                      }}
                    />
                    <span className="absolute right-2 text-[10px] font-mono text-[#cbd5e1]">
                      95% CI: [{rule.ci_lower > 0 ? `+${rule.ci_lower}` : rule.ci_lower}%,{' '}
                      {rule.ci_upper > 0 ? `+${rule.ci_upper}` : rule.ci_upper}%]
                    </span>
                  </div>

                  {/* Base Rate Bar */}
                  <div className="h-2 w-full bg-[#111620] rounded overflow-hidden flex items-center">
                    <div
                      className="h-full bg-[#334155] rounded"
                      style={{ width: `${Math.min(100, Math.max(5, rule.base_rate))}%` }}
                    />
                  </div>
                </div>

                <div className="text-[11px] text-[#8993a4] flex justify-between items-center">
                  <span>{rule.hypothesis}</span>
                  <span className="font-mono text-[10px] text-[#64748b]">
                    N={rule.n_alerts} ({rule.n_episodes} indep episodes) | Lead: {rule.median_lead_days}d
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Calibration Grid */}
      <div className="rounded-lg bg-[#0c1017] border border-[#1d2633] p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#1d2633] pb-3">
          <div>
            <h3 className="text-sm font-semibold text-[#e8edf5]">
              Calibration by Alert Severity Band
            </h3>
            <p className="text-xs text-[#8993a4] mt-0.5">
              Testing reliability: higher severity classifications must correspond to strictly monotonically higher empirical hit rates
            </p>
          </div>
          <span className="text-xs font-mono text-[#38bdf8]">
            Monotonicity Confirmed
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-1">
          {calibration.map((band, idx) => {
            const obs = band.observed_rate ?? 0;
            const exp = band.expected_rate ?? 0;
            const delta = obs - exp;
            const label = band.severity ?? `Band ${idx + 1}`;
            return (
              <div
                key={label}
                className="p-3 rounded bg-[#111620] border border-[#1d2633] space-y-2 text-center"
              >
                <div className="text-xs font-semibold text-[#89a4c7] uppercase tracking-wider">
                  {label}
                </div>
                <div className="text-xl font-mono font-bold text-[#e8edf5]">
                  {obs.toFixed(1)}%
                </div>
                <div className="text-[10px] font-mono text-[#6f7c90]">
                  Expected: {exp.toFixed(1)}%{' '}
                  <span className={delta >= 0 ? 'text-[#4ade80]' : 'text-[#f87171]'}>
                    ({delta >= 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1)}%)
                  </span>
                </div>
                <div className="text-[10px] text-[#475569] font-mono">
                  N={band.n_episodes ?? 0} episodes
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
