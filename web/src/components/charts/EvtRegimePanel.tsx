'use client';

import React, { useState } from 'react';
import type { TailsRegimesData } from '@/lib/data/schemas';

interface EvtRegimePanelProps {
  data: TailsRegimesData | null;
}

export function EvtRegimePanel({ data }: EvtRegimePanelProps) {
  const [activeTab, setActiveTab] = useState<'tails' | 'regimes'>('tails');

  if (!data) {
    return (
      <div className="p-6 rounded-lg bg-[#0c1017] border border-[#1d2633] text-center text-[#6f7c90] font-mono text-xs">
        Loading EVT Tails and Markov-Switching Regimes...
      </div>
    );
  }

  const { evt, regimes, target_name, as_of, interpretation } = data;
  const isStress = regimes.current_regime === 'stress';

  return (
    <div className="rounded-lg bg-[#0c1017] border border-[#1d2633] p-5 space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#1d2633] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <h3 className="text-sm font-semibold text-[#e8edf5] tracking-tight">
              Advanced Tail Risk & Latent Regimes (P10)
            </h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#16202c] border border-[#26354a] text-[#89a4c7]">
              {target_name}
            </span>
          </div>
          <p className="text-xs text-[#8993a4] mt-0.5">
            Generalized Pareto (POT) tail parameterization & Hamilton 2-regime Markov switching
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {/* Regime Badge */}
          <div
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs font-mono font-medium border ${
              isStress
                ? 'bg-[#3d1418] border-[#c74e5b] text-[#f87171]'
                : 'bg-[#102a20] border-[#2d8a64] text-[#4ade80]'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isStress ? 'bg-[#f87171] animate-pulse' : 'bg-[#4ade80]'
              }`}
            />
            <span>
              Regime: {regimes.current_regime.toUpperCase()} ({(regimes.current_prob_stress * 100).toFixed(0)}%)
            </span>
          </div>

          {/* Tail Fatness Badge */}
          <div className="px-2.5 py-1 rounded text-xs font-mono bg-[#181e29] border border-[#273244] text-[#a0aec0]">
            Tail: <span className="text-[#e2e8f0] font-medium">{evt.tail_fatness}</span> (ξ={evt.shape_xi})
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex space-x-2 border-b border-[#1d2633]">
        <button
          onClick={() => setActiveTab('tails')}
          className={`pb-2 px-3 text-xs font-medium transition-colors border-b-2 ${
            activeTab === 'tails'
              ? 'border-[#2f6fb4] text-[#e8edf5]'
              : 'border-transparent text-[#6f7c90] hover:text-[#a9b4c4]'
          }`}
        >
          EVT Generalized Pareto Tails
        </button>
        <button
          onClick={() => setActiveTab('regimes')}
          className={`pb-2 px-3 text-xs font-medium transition-colors border-b-2 ${
            activeTab === 'regimes'
              ? 'border-[#2f6fb4] text-[#e8edf5]'
              : 'border-transparent text-[#6f7c90] hover:text-[#a9b4c4]'
          }`}
        >
          Markov-Switching Volatility
        </button>
      </div>

      {/* Tab 1: EVT Tails */}
      {activeTab === 'tails' && (
        <div className="space-y-4">
          {/* Summary Stat Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 rounded bg-[#111620] border border-[#1d2633]">
              <div className="text-[11px] font-mono text-[#6f7c90]">Shape Parameter (ξ)</div>
              <div className="text-base font-mono font-bold text-[#e8edf5] mt-0.5">
                {evt.shape_xi > 0 ? `+${evt.shape_xi.toFixed(3)}` : evt.shape_xi.toFixed(3)}
              </div>
              <div className="text-[10px] text-[#8993a4] mt-0.5">
                {evt.shape_xi > 0.1 ? 'Heavy Fréchet tail' : 'Near-exponential'}
              </div>
            </div>

            <div className="p-3 rounded bg-[#111620] border border-[#1d2633]">
              <div className="text-[11px] font-mono text-[#6f7c90]">Scale (β) & Threshold (u)</div>
              <div className="text-base font-mono font-bold text-[#e8edf5] mt-0.5">
                {evt.scale_beta.toFixed(2)} <span className="text-xs text-[#6f7c90]">/ {evt.threshold_u.toFixed(2)}σ</span>
              </div>
              <div className="text-[10px] text-[#8993a4] mt-0.5">95% expanding POT cutoff</div>
            </div>

            <div className="p-3 rounded bg-[#111620] border border-[#1d2633]">
              <div className="text-[11px] font-mono text-[#6f7c90]">99.0% EVT VaR</div>
              <div className="text-base font-mono font-bold text-[#f59e0b] mt-0.5">
                {evt.var_99_pct.toFixed(2)}%
              </div>
              <div className="text-[10px] text-[#8993a4] mt-0.5">
                vs {evt.gaussian_var_99.toFixed(2)}% Gauss ({evt.excess_ratio_over_gaussian.toFixed(1)}x)
              </div>
            </div>

            <div className="p-3 rounded bg-[#111620] border border-[#1d2633]">
              <div className="text-[11px] font-mono text-[#6f7c90]">99.0% Expected Shortfall</div>
              <div className="text-base font-mono font-bold text-[#ef4444] mt-0.5">
                {evt.es_99_pct.toFixed(2)}%
              </div>
              <div className="text-[10px] text-[#8993a4] mt-0.5">Tail conditional loss E[L|L&gt;VaR]</div>
            </div>
          </div>

          {/* Quantile Comparison Table */}
          <div className="overflow-x-auto rounded border border-[#1d2633]">
            <table className="w-full text-left font-mono text-xs">
              <thead className="bg-[#121824] text-[#8993a4] border-b border-[#1d2633]">
                <tr>
                  <th className="py-2 px-3">Quantile / Horizon</th>
                  <th className="py-2 px-3">Gaussian VaR</th>
                  <th className="py-2 px-3">Empirical VaR</th>
                  <th className="py-2 px-3 text-[#f59e0b]">EVT GPD VaR</th>
                  <th className="py-2 px-3 text-[#ef4444]">EVT Expected Shortfall</th>
                  <th className="py-2 px-3">Tail Multiplier</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1d2633] text-[#e8edf5]">
                <tr className="hover:bg-[#161f2e] transition">
                  <td className="py-2 px-3 font-semibold text-[#89a4c7]">99.0% (1-in-100 days)</td>
                  <td className="py-2 px-3">{evt.gaussian_var_99.toFixed(2)}%</td>
                  <td className="py-2 px-3">{evt.empirical_var_99.toFixed(2)}%</td>
                  <td className="py-2 px-3 text-[#f59e0b] font-semibold">{evt.var_99_pct.toFixed(2)}%</td>
                  <td className="py-2 px-3 text-[#ef4444] font-semibold">{evt.es_99_pct.toFixed(2)}%</td>
                  <td className="py-2 px-3 text-[#38bdf8] font-bold">
                    {(evt.var_99_pct / Math.max(0.01, evt.gaussian_var_99)).toFixed(2)}x
                  </td>
                </tr>
                <tr className="hover:bg-[#161f2e] transition">
                  <td className="py-2 px-3 font-semibold text-[#89a4c7]">99.5% (1-in-200 days)</td>
                  <td className="py-2 px-3">{evt.gaussian_var_995.toFixed(2)}%</td>
                  <td className="py-2 px-3">{evt.empirical_var_995.toFixed(2)}%</td>
                  <td className="py-2 px-3 text-[#f59e0b] font-semibold">{evt.var_995_pct.toFixed(2)}%</td>
                  <td className="py-2 px-3 text-[#ef4444] font-semibold">{evt.es_995_pct.toFixed(2)}%</td>
                  <td className="py-2 px-3 text-[#38bdf8] font-bold">
                    {(evt.var_995_pct / Math.max(0.01, evt.gaussian_var_995)).toFixed(2)}x
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Mean-Excess Curve */}
          {evt.mean_excess_curve && evt.mean_excess_curve.length > 0 && (
            <div className="p-3.5 rounded bg-[#111620] border border-[#1d2633] space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-[#e8edf5]">
                  Mean Excess Function e(u) = E[X - u | X &gt; u]
                </span>
                <span className="text-[11px] font-mono text-[#6f7c90]">
                  Linearly upward slope confirms heavy-tailed Pareto decay
                </span>
              </div>
              <div className="grid grid-cols-5 sm:grid-cols-10 gap-1.5 pt-1">
                {evt.mean_excess_curve.map((pt) => (
                  <div
                    key={pt.quantile}
                    className="p-1.5 rounded bg-[#0b0e14] border border-[#1d2633] text-center font-mono"
                  >
                    <div className="text-[10px] text-[#6f7c90]">{pt.quantile}%</div>
                    <div className="text-xs font-bold text-[#e2e8f0]">{pt.mean_excess.toFixed(2)}</div>
                    <div className="text-[9px] text-[#475569]">N={pt.count}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="text-[11px] text-[#6f7c90] italic">
            * Note: EVT tail estimates are fitted to historical loss excesses above the 95th percentile.
            Model-based estimates are subject to parameter uncertainty beyond the observed sample.
          </div>
        </div>
      )}

      {/* Tab 2: Markov-Switching Regimes */}
      {activeTab === 'regimes' && (
        <div className="space-y-4">
          {/* Key Parameters */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="p-3 rounded bg-[#111620] border border-[#1d2633]">
              <div className="text-[11px] font-mono text-[#6f7c90]">Point-in-Time Stress Probability</div>
              <div className="text-lg font-mono font-bold mt-0.5 text-[#e8edf5]">
                {(regimes.current_prob_stress * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] text-[#8993a4] mt-0.5">Filtered P(S_t = Stress | Y_1..t)</div>
            </div>

            <div className="p-3 rounded bg-[#111620] border border-[#1d2633]">
              <div className="text-[11px] font-mono text-[#6f7c90]">Expected Calm Duration</div>
              <div className="text-lg font-mono font-bold text-[#4ade80] mt-0.5">
                {regimes.expected_duration_days.calm_regime} days
              </div>
              <div className="text-[10px] text-[#8993a4] mt-0.5">
                p00 = {regimes.transition_matrix.p00_calm_to_calm.toFixed(2)}
              </div>
            </div>

            <div className="p-3 rounded bg-[#111620] border border-[#1d2633]">
              <div className="text-[11px] font-mono text-[#6f7c90]">Expected Stress Duration</div>
              <div className="text-lg font-mono font-bold text-[#f87171] mt-0.5">
                {regimes.expected_duration_days.stress_regime} days
              </div>
              <div className="text-[10px] text-[#8993a4] mt-0.5">
                p11 = {regimes.transition_matrix.p11_stress_to_stress.toFixed(2)}
              </div>
            </div>
          </div>

          {/* Regime Timeline Ribbon */}
          {regimes.ribbon && regimes.ribbon.length > 0 && (
            <div className="p-3.5 rounded bg-[#111620] border border-[#1d2633] space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-[#e8edf5]">
                  Historical Regime Probability Ribbon (Last {regimes.ribbon.length} Trading Days)
                </span>
                <span className="text-[11px] font-mono text-[#6f7c90]">
                  Filtered (point-in-time) vs Smoothed
                </span>
              </div>

              {/* Strip visualization */}
              <div className="h-10 w-full flex rounded overflow-hidden border border-[#1d2633] mt-2">
                {regimes.ribbon.map((r, idx) => {
                  const prob = r.smoothed_prob_stress;
                  // Color interpolation from calm (green/slate) to stress (red)
                  const bgColor =
                    prob >= 0.70
                      ? '#ef4444'
                      : prob >= 0.40
                      ? '#f59e0b'
                      : prob >= 0.20
                      ? '#2f6fb4'
                      : '#102a20';
                  return (
                    <div
                      key={idx}
                      title={`${r.date}: P(Stress)=${(prob * 100).toFixed(0)}% (${r.regime})`}
                      style={{ backgroundColor: bgColor }}
                      className="flex-1 h-full cursor-pointer hover:opacity-80 transition-opacity"
                    />
                  );
                })}
              </div>

              <div className="flex justify-between items-center text-[10px] font-mono text-[#6f7c90] pt-1">
                <span>{regimes.ribbon[0]?.date}</span>
                <div className="flex items-center space-x-3">
                  <span className="flex items-center space-x-1">
                    <span className="w-2.5 h-2.5 rounded-sm bg-[#102a20] border border-[#2d8a64]" />
                    <span>Calm (&lt;20%)</span>
                  </span>
                  <span className="flex items-center space-x-1">
                    <span className="w-2.5 h-2.5 rounded-sm bg-[#2f6fb4]" />
                    <span>Transitional (20-40%)</span>
                  </span>
                  <span className="flex items-center space-x-1">
                    <span className="w-2.5 h-2.5 rounded-sm bg-[#f59e0b]" />
                    <span>Elevated (40-70%)</span>
                  </span>
                  <span className="flex items-center space-x-1">
                    <span className="w-2.5 h-2.5 rounded-sm bg-[#ef4444]" />
                    <span>High Stress (&ge;70%)</span>
                  </span>
                </div>
                <span>{regimes.ribbon[regimes.ribbon.length - 1]?.date}</span>
              </div>
            </div>
          )}

          {/* Transition Matrix */}
          <div className="p-3 rounded bg-[#111620] border border-[#1d2633] space-y-2">
            <div className="text-xs font-semibold text-[#e8edf5]">Regime Transition Matrix</div>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2 rounded bg-[#0b0e14] border border-[#1d2633]">
                <div className="text-[#6f7c90] text-[10px]">P(Calm → Calm)</div>
                <div className="text-sm font-bold text-[#4ade80]">
                  {(regimes.transition_matrix.p00_calm_to_calm * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-2 rounded bg-[#0b0e14] border border-[#1d2633]">
                <div className="text-[#6f7c90] text-[10px]">P(Calm → Stress)</div>
                <div className="text-sm font-bold text-[#f59e0b]">
                  {(regimes.transition_matrix.p01_calm_to_stress * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-2 rounded bg-[#0b0e14] border border-[#1d2633]">
                <div className="text-[#6f7c90] text-[10px]">P(Stress → Calm)</div>
                <div className="text-sm font-bold text-[#38bdf8]">
                  {(regimes.transition_matrix.p10_stress_to_calm * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-2 rounded bg-[#0b0e14] border border-[#1d2633]">
                <div className="text-[#6f7c90] text-[10px]">P(Stress → Stress)</div>
                <div className="text-sm font-bold text-[#f87171]">
                  {(regimes.transition_matrix.p11_stress_to_stress * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Narrative Interpretation */}
      <div className="p-3 rounded bg-[#111620] border border-[#1d2633] text-xs text-[#a9b4c4]">
        <span className="font-semibold text-[#e8edf5] mr-1">Statistical Summary:</span>
        {interpretation}
      </div>
    </div>
  );
}
