// web/src/app/methodology/page.tsx
"use client";

import React, { useState } from "react";
import { ShieldCheck, BookOpen, Database, Sliders, ExternalLink, Scale, CheckCircle2, AlertTriangle, EyeOff, Lock, Code2 } from "lucide-react";

export default function MethodologyPage() {
  const [activeTab, setActiveTab] = useState<"math" | "sources" | "limitations" | "privacy" | "params" | "compliance">("math");

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Page Header */}
      <div className="pb-4 border-b border-[var(--line)]">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">Governance & Standards</span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[var(--bg-2)] border border-[var(--line)] text-[var(--accent)]">
            Single Source of Truth
          </span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-[var(--tx-0)] mt-1">
          Methodology, Specifications & Traceability
        </h1>
        <p className="text-sm text-[var(--tx-1)] mt-1.5 max-w-3xl leading-relaxed">
          MacroMonitor is built upon a single brand promise: <strong className="text-[var(--tx-0)] font-semibold">&ldquo;Every number is traceable.&rdquo;</strong> Every metric, chart, and alert links to an exact mathematical formula, unadulterated source inputs, autocorrelation-adjusted sample size (N_eff), and hyperparameter lineage.
        </p>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-[var(--line)] pb-3 text-xs font-mono overflow-x-auto">
        <button
          onClick={() => setActiveTab("math")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded transition-colors whitespace-nowrap ${
            activeTab === "math"
              ? "bg-[var(--bg-2)] text-[var(--tx-0)] border border-[var(--accent)]"
              : "text-[var(--tx-2)] hover:text-[var(--tx-1)]"
          }`}
        >
          <BookOpen className="w-3.5 h-3.5 text-[var(--accent)]" />
          Mathematical Specifications
        </button>
        <button
          onClick={() => setActiveTab("sources")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded transition-colors whitespace-nowrap ${
            activeTab === "sources"
              ? "bg-[var(--bg-2)] text-[var(--tx-0)] border border-[var(--accent)]"
              : "text-[var(--tx-2)] hover:text-[var(--tx-1)]"
          }`}
        >
          <Database className="w-3.5 h-3.5 text-[var(--pos)]" />
          Data Sources & Lineage
        </button>
        <button
          onClick={() => setActiveTab("limitations")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded transition-colors whitespace-nowrap ${
            activeTab === "limitations"
              ? "bg-[var(--bg-2)] text-[var(--tx-0)] border border-[var(--accent)]"
              : "text-[var(--tx-2)] hover:text-[var(--tx-1)]"
          }`}
        >
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
          Known Limitations & Boundaries
        </button>
        <button
          onClick={() => setActiveTab("privacy")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded transition-colors whitespace-nowrap ${
            activeTab === "privacy"
              ? "bg-[var(--bg-2)] text-[var(--tx-0)] border border-[var(--accent)]"
              : "text-[var(--tx-2)] hover:text-[var(--tx-1)]"
          }`}
        >
          <Lock className="w-3.5 h-3.5 text-sky-400" />
          Privacy & Architecture
        </button>
        <button
          onClick={() => setActiveTab("params")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded transition-colors whitespace-nowrap ${
            activeTab === "params"
              ? "bg-[var(--bg-2)] text-[var(--tx-0)] border border-[var(--accent)]"
              : "text-[var(--tx-2)] hover:text-[var(--tx-1)]"
          }`}
        >
          <Sliders className="w-3.5 h-3.5 text-[var(--warn)]" />
          Hyperparameters (params.yaml)
        </button>
        <button
          onClick={() => setActiveTab("compliance")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded transition-colors whitespace-nowrap ${
            activeTab === "compliance"
              ? "bg-[var(--bg-2)] text-[var(--tx-0)] border border-[var(--accent)]"
              : "text-[var(--tx-2)] hover:text-[var(--tx-1)]"
          }`}
        >
          <Scale className="w-3.5 h-3.5 text-[var(--tx-1)]" />
          Regulatory Non-Advisory Mandate
        </button>
      </div>

      {/* Tab 1: Mathematical Specifications */}
      {activeTab === "math" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Spec 1: Effective Sample Size */}
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-[var(--line)]">
                <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)]">
                  1. Effective Sample Size (N_eff)
                </h3>
                <span className="text-[10px] font-mono text-[var(--accent)]">s03-percentile-zscore-neff</span>
              </div>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Financial time series exhibit serial autocorrelation, which inflates naive degrees of freedom and creates spurious statistical significance. We compute the lag-1 autocorrelation <code className="text-[var(--accent)]">&rho;₁</code> and discount raw sample size N:
              </p>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] font-mono text-xs text-[var(--tx-0)] text-center">
                N_eff = N &times; (1 &minus; &rho;₁) / (1 + &rho;₁)
              </div>
              <p className="text-[11px] text-[var(--tx-2)]">
                If &rho;₁ &le; 0, N_eff = N. When &rho;₁ approaches 1.0, effective independent observations decrease substantially.
              </p>
            </div>

            {/* Spec 2: Expanding Percentiles & Anti-Lookahead */}
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-[var(--line)]">
                <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)]">
                  2. Expanding Empirical Percentile
                </h3>
                <span className="text-[10px] font-mono text-[var(--accent)]">s03-percentile-zscore-neff</span>
              </div>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                To eliminate lookahead bias, all percentiles and z-scores are strictly computed against expanding empirical history up to time t:
              </p>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] font-mono text-xs text-[var(--tx-0)] text-center">
                {"P_t = (1 / N_t) ∑ 1_{x_s ≤ x_t} × 100"}
              </div>
              <p className="text-[11px] text-[var(--tx-2)]">
                {"For expanding z-scores, μ and σ are computed strictly over indices 0 to t−1, completely excluding current point t."}
              </p>
            </div>

            {/* Spec 3: Move Decomposer */}
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-[var(--line)]">
                <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)]">
                  3. Move Decomposer (OLS + HAC + Ridge)
                </h3>
                <span className="text-[10px] font-mono text-[var(--accent)]">s06-move-decomposer</span>
              </div>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Country index returns are decomposed into macro factors estimated strictly over a prior 250-day window ending before the move horizon:
              </p>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] font-mono text-xs text-[var(--tx-0)] text-center">
                {"r_{i,t} = α + β_{row} r_{row,t} + β_{fx} r_{fx,t} + β_{rates} r_{rates,t} + β_{comm} r_{comm,t} + ε_t"}
              </div>
              <p className="text-[11px] text-[var(--tx-2)]">
                Standard errors use Newey-West HAC. If Variance Inflation Factor (VIF) &ge; 5.0, Ridge regression (&alpha;=1.0) is triggered automatically with an explicit UI badge.
              </p>
            </div>

            {/* Spec 4: Sector Contribution Waterfall */}
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-[var(--line)]">
                <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)]">
                  4. Sector Contribution & Arithmetic Residual
                </h3>
                <span className="text-[10px] font-mono text-[var(--accent)]">s07-sector-contribution</span>
              </div>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Approximate sector contributions are computed using static index weights and reconciled with an explicit residual:
              </p>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] font-mono text-xs text-[var(--tx-0)] text-center">
                {"contrib_s = w_s × r_s,    residual = r_{index} − ∑ contrib_s"}
              </div>
              <p className="text-[11px] text-[var(--tx-2)]">
                Guarantees exact arithmetic reconciliation. Weights are updated annually or quarterly with documented rebalance dates.
              </p>
            </div>

            {/* Spec 5: Macro Driver Sensitivities */}
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-[var(--line)]">
                <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)]">
                  5. Macro Driver Rolling Sensitivities
                </h3>
                <span className="text-[10px] font-mono text-[var(--accent)]">s08-macro-drivers</span>
              </div>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Rolling 250D and 120D multi-factor regressions with Newey-West HAC covariance. Sensitivities include stability scoring:
              </p>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] font-mono text-xs text-[var(--tx-0)] text-center">
                {"Stability = 1 − (IQR_{120D} / Range_{250D})"}
              </div>
              <p className="text-[11px] text-[var(--tx-2)]">
                Empirical beta is benchmarked against pre-registered economic priors to detect structural regime breaks.
              </p>
            </div>

            {/* Spec 6: Risk Lab Cones & Tail Risk */}
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-[var(--line)]">
                <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)]">
                  6. Volatility Cones, VaR & Expected Shortfall
                </h3>
                <span className="text-[10px] font-mono text-[var(--accent)]">s09-risk-lab</span>
              </div>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Annualized realized volatility plotted across 10D, 20D, 60D, 120D, and 250D horizons against expanding percentiles (10th, 25th, 50th, 75th, 90th).
              </p>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] font-mono text-xs text-[var(--tx-0)] text-center">
                {"VaR_{99} = Q_{0.01}(r),    ES_{99} = E[r | r ≤ VaR_{99}]"}
              </div>
              <p className="text-[11px] text-[var(--tx-2)]">
                {"First principal component (PCA) synchronization enforces US sign normalization (v_{1,US} > 0) to guarantee directional coherence."}
              </p>
            </div>

            {/* Spec 7: CUSUM Innovation Control Chart */}
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-[var(--line)]">
                <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)]">
                  7. CUSUM Change Detector (Chart C24)
                </h3>
                <span className="text-[10px] font-mono text-[var(--accent)]">s11-radar-divergence</span>
              </div>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Page&apos;s two-sided Cumulative Sum control chart on standardized momentum innovations with slack k=0.5 and threshold h=4.0:
              </p>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] font-mono text-xs text-[var(--tx-0)] text-center">
                {"S_t^+ = max(0, S_{t-1}^+ + z_t − 0.5),    S_t^- = max(0, S_{t-1}^- − z_t − 0.5)"}
              </div>
              <p className="text-[11px] text-[var(--tx-2)]">
                Detects persistent regime drift while rejecting transient white noise spikes, eliminating false alarm clustering.
              </p>
            </div>

            {/* Spec 8: Market State Map */}
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-[var(--line)]">
                <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)]">
                  8. Market State Map Regimes
                </h3>
                <span className="text-[10px] font-mono text-[var(--accent)]">s05-pulse-page</span>
              </div>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Assets are mapped into a 4-quadrant state space based on 60-day regression trend t-statistic and 20-day volatility percentile:
              </p>
              <ul className="text-xs font-mono space-y-1 text-[var(--tx-1)]">
                <li>• <strong className="text-emerald-400">Orderly Trend:</strong> {"|t| ≥ 2.0 and Vol %tile ≤ 50%"}</li>
                <li>• <strong className="text-rose-400">Violent Trend:</strong> {"|t| ≥ 2.0 and Vol %tile > 50%"}</li>
                <li>• <strong className="text-sky-400">Quiet Range:</strong> {"|t| < 2.0 and Vol %tile ≤ 50%"}</li>
                <li>• <strong className="text-amber-400">Choppy:</strong> {"|t| < 2.0 and Vol %tile > 50%"}</li>
              </ul>
            </div>

            {/* Spec 9: US Net Liquidity */}
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-[var(--line)]">
                <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)]">
                  9. US Net Liquidity Formula
                </h3>
                <span className="text-[10px] font-mono text-[var(--accent)]">s09-liquidity-module</span>
              </div>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Federal Reserve balance sheet decomposition aligned to Billions of USD ($B):
              </p>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] font-mono text-xs text-[var(--tx-0)] text-center">
                {"Net Liquidity = (WALCL / 1000) − (WTREGEN / 1000) − RRPONTSYD"}
              </div>
              <p className="text-[11px] text-[var(--tx-2)]">
                Overlaid with S&P 500 alongside rolling 60-day correlation, accompanied by explicit &ldquo;Association, not mechanical causation&rdquo; caveat.
              </p>
            </div>

            {/* Spec 10: F&O Open Interest Buildup */}
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-[var(--line)]">
                <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)]">
                  10. F&O OI Buildup Matrix & PCR
                </h3>
                <span className="text-[10px] font-mono text-[var(--accent)]">s10-india-flows</span>
              </div>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Derivatives positioning classified by price delta (&Delta;P) and open interest delta (&Delta;OI):
              </p>
              <ul className="text-xs font-mono space-y-1 text-[var(--tx-1)]">
                <li>• <strong className="text-emerald-400">Long Buildup:</strong> {"ΔP > 0, ΔOI > 0"}</li>
                <li>• <strong className="text-rose-400">Short Buildup:</strong> {"ΔP < 0, ΔOI > 0"}</li>
                <li>• <strong className="text-amber-400">Short Covering:</strong> {"ΔP > 0, ΔOI < 0"}</li>
                <li>• <strong className="text-purple-400">Long Unwinding:</strong> {"ΔP < 0, ΔOI < 0"}</li>
              </ul>
              <p className="text-[11px] text-[var(--tx-2)]">
                Put-Call Ratio (PCR = &sum; Put OI / &sum; Call OI) with zero-division guard.
              </p>
            </div>

            {/* Spec 11: Financial Stress Composite */}
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3 md:col-span-2">
              <div className="flex items-center justify-between pb-2 border-b border-[var(--line)]">
                <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)]">
                  11. Cross-Asset Stress Composite (Zero Synthetic Imputation)
                </h3>
                <span className="text-[10px] font-mono text-[var(--accent)]">s11-stress-composite-briefing</span>
              </div>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Cross-asset friction index computed as the mean of available standardized indicators without synthetic imputation:
              </p>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] font-mono text-xs text-[var(--tx-0)] text-center">
                {"S_t = (1 / |K_t|) ∑ c_{k, t}   where c_{k, t} are available standardized friction metrics"}
              </div>
              <p className="text-[11px] text-[var(--tx-2)]">
                Missing components are dropped transparently from the denominator |K_t| and reported in the Audit drawer. No forward-filling across series boundaries.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Data Sources & Lineage */}
      {activeTab === "sources" && (
        <div className="space-y-4">
          <div className="overflow-x-auto rounded-lg border border-[var(--line)]">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-[var(--bg-0)] text-[var(--tx-2)] uppercase text-[10px] border-b border-[var(--line)]">
                <tr>
                  <th className="py-3 px-4">Provider</th>
                  <th className="py-3 px-4">Coverage</th>
                  <th className="py-3 px-4">Series Identifiers</th>
                  <th className="py-3 px-4">Ingestion Cadence</th>
                  <th className="py-3 px-4">License / Attribution</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--line)] bg-[var(--bg-1)]">
                <tr>
                  <td className="py-3 px-4 font-semibold text-[var(--tx-0)]">FRED (St. Louis Fed)</td>
                  <td className="py-3 px-4 text-[var(--tx-1)]">US Treasury yields, Fed Balance sheet, TGA, RRP, SOFR</td>
                  <td className="py-3 px-4 text-[var(--accent)]">WALCL, WTREGEN, RRPONTSYD, DGS10, SOFR</td>
                  <td className="py-3 px-4 text-[var(--tx-2)]">Daily / Weekly Wed</td>
                  <td className="py-3 px-4 text-[var(--tx-2)]">Public domain (Federal Reserve Board)</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-semibold text-[var(--tx-0)]">Yahoo Finance</td>
                  <td className="py-3 px-4 text-[var(--tx-1)]">Global equity indices, US sectors, FX spot pairs, Commodities, VIX</td>
                  <td className="py-3 px-4 text-[var(--accent)]">^GSPC, ^IXIC, ^NSEI, ^VIX, CL=F, GC=F</td>
                  <td className="py-3 px-4 text-[var(--tx-2)]">Daily market close</td>
                  <td className="py-3 px-4 text-[var(--tx-2)]">Open web market data</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-semibold text-[var(--tx-0)]">NSE India</td>
                  <td className="py-3 px-4 text-[var(--tx-1)]">Nifty 50, Nifty Bank, sectoral indices, India VIX, FII/DII net flows</td>
                  <td className="py-3 px-4 text-[var(--accent)]">NIFTY 50, INDIA VIX, FII/DII Cash, F&O Bhavcopy</td>
                  <td className="py-3 px-4 text-[var(--tx-2)]">Daily post-market</td>
                  <td className="py-3 px-4 text-[var(--tx-2)]">NSE Daily Reports & SEBI Disclosures</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-semibold text-[var(--tx-0)]">New York Fed Markets</td>
                  <td className="py-3 px-4 text-[var(--tx-1)]">Overnight Reverse Repo Operations (ON RRP)</td>
                  <td className="py-3 px-4 text-[var(--accent)]">RRPONTSYD</td>
                  <td className="py-3 px-4 text-[var(--tx-2)]">Daily 13:15 ET</td>
                  <td className="py-3 px-4 text-[var(--tx-2)]">Public domain (Federal Reserve Bank of NY)</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-semibold text-[var(--tx-0)]">DefiLlama / On-Chain</td>
                  <td className="py-3 px-4 text-[var(--tx-1)]">Total circulating USD stablecoins aggregate</td>
                  <td className="py-3 px-4 text-[var(--accent)]">peggedUSD</td>
                  <td className="py-3 px-4 text-[var(--tx-2)]">Daily</td>
                  <td className="py-3 px-4 text-[var(--tx-2)]">Open DefiLlama public API</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 3: Known Limitations & Boundaries */}
      {activeTab === "limitations" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)] flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Strict Scope Boundary: Macro Only
              </h3>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                MacroMonitor tracks <strong>macro and multi-asset series exclusively</strong> (country index level, sovereign yields, exchange rates, broad commodities, and sector indices).
              </p>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                There is a <strong>permanent structural prohibition</strong> on individual company stocks, ticker screeners, watchlist picking, price targets, or micro-cap instruments.
              </p>
            </div>

            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)] flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Daily Frequency & Close Prices
              </h3>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                All models evaluate end-of-day official closing prices. The system does not ingest tick-level data, order book depth, or real-time streaming quotes.
              </p>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Time-zone alignment is handled via our published exchange calendar matrix (<code className="text-[var(--accent)] font-mono">calendars.yaml</code>), with 5-day horizon defaults for cross-regional models.
              </p>
            </div>

            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)] flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Association vs. Mechanical Causation
              </h3>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Statistical regression coefficients, correlations, and liquidity overlays denote empirical co-movement and descriptive covariance, <strong>never mechanical causation</strong>.
              </p>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Factor decomposition models use prior-window estimation windows to prevent lookahead, but cannot foresee regime shifts or exogenous geopolitical shocks.
              </p>
            </div>

            <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
              <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)] flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Historical Analogues & Non-Prediction
              </h3>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Declustered historical analogues identify past episodes with similar multivariate signatures. Past return distributions following an analogue provide context, not forward guarantees.
              </p>
              <p className="text-xs text-[var(--tx-1)] leading-relaxed">
                Every analogue card displays empirical dispersion, highlighting that historical variance is wide and past outcomes do not repeat deterministically.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Privacy & Architecture */}
      {activeTab === "privacy" && (
        <div className="space-y-6">
          <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
            <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)] flex items-center gap-2">
              <Lock className="w-4 h-4 text-emerald-400" />
              100% Static Zero-Tracking Architecture
            </h3>
            <p className="text-xs text-[var(--tx-1)] leading-relaxed">
              MacroMonitor is compiled to <strong>100% static JSON bundles and pre-rendered HTML</strong> via Next.js static export. There is no user database, no authentication server, and no server-side telemetry.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] text-center">
                <span className="text-xs font-mono font-semibold text-[var(--pos)]">Zero Cookies</span>
                <p className="text-[11px] text-[var(--tx-2)] mt-1">No session cookies, tracking cookies, or advertising identifiers.</p>
              </div>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] text-center">
                <span className="text-xs font-mono font-semibold text-[var(--pos)]">Zero Analytics Trackers</span>
                <p className="text-[11px] text-[var(--tx-2)] mt-1">No Google Analytics, Meta Pixel, Hotjar, or third-party behavioral scripts.</p>
              </div>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] text-center">
                <span className="text-xs font-mono font-semibold text-[var(--pos)]">Zero PII Collected</span>
                <p className="text-[11px] text-[var(--tx-2)] mt-1">No personally identifiable information is requested, ingested, or stored.</p>
              </div>
            </div>
          </div>

          <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-3">
            <h3 className="text-sm font-mono font-semibold text-[var(--tx-0)] flex items-center gap-2">
              <Code2 className="w-4 h-4 text-[var(--accent)]" />
              Deterministic Pipeline Lineage
            </h3>
            <p className="text-xs text-[var(--tx-1)] leading-relaxed">
              Every data artifact exported to <code className="text-[var(--accent)] font-mono">/data/v1/</code> contains an embedded cryptographic parameter hash (<code className="text-[var(--accent)] font-mono">params_hash</code>), code version tag (<code className="text-[var(--accent)] font-mono">code_version</code>), and precise timestamp (<code className="text-[var(--accent)] font-mono">generated_at</code>). Any discrepancy in parameter inputs invalidates the build manifest.
            </p>
          </div>
        </div>
      )}

      {/* Tab 5: Hyperparameters (params.yaml) */}
      {activeTab === "params" && (
        <div className="space-y-4">
          <p className="text-xs text-[var(--tx-1)]">
            Section 0.4 operating rule: <strong className="text-[var(--tx-0)]">Zero hardcoded thresholds.</strong> All quantitative parameters reside in <code className="text-[var(--accent)] font-mono">pipeline/config/params.yaml</code>:
          </p>
          <div className="p-4 rounded-lg bg-[var(--bg-0)] border border-[var(--line)] font-mono text-xs text-[var(--tx-1)] space-y-2 overflow-x-auto">
            <pre className="text-[11px] leading-relaxed text-[#a9b4c4]">
{`data_cleaning:
  max_ffill_days: 3
  jump_sigma: 8.0
  max_stale_pct: 0.20
  stale_after_hours: 36
  min_obs: 750

volatility:
  windows: [10, 20, 60]
  annualization_factor: 252.0

drawdown:
  window: 252

market_state_map:
  trend_window: 60
  trend_threshold: 2.0
  vol_percentile_threshold: 50.0
  momentum_window: 20
  momentum_lag: 20
  hysteresis_days: 3

correlation:
  short_window: 60
  long_window: 250

divergence:
  momentum_window: 20
  percentile_threshold: 95.0
  consecutive_days: 3
  cusum_slack: 0.5
  cusum_threshold: 4.0

liquidity:
  change_windows: [20, 65]
  stablecoin_window: 30

insights:
  cooldown_days: 10
  daily_cap: 8
  severity_bands:
    low: [90.0, 95.0]
    medium: [95.0, 99.0]
    high: [99.0, 100.0]`}
            </pre>
          </div>
        </div>
      )}

      {/* Tab 6: Compliance & Disclaimer */}
      {activeTab === "compliance" && (
        <div className="space-y-4">
          <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-amber-500/30 space-y-3">
            <div className="flex items-center gap-2 text-amber-400 font-mono font-semibold text-xs uppercase tracking-wider">
              <Scale className="w-4 h-4" />
              <span>Descriptive Statistics Only • Non-Advisory Mandate</span>
            </div>
            <p className="text-xs text-[var(--tx-1)] leading-relaxed">
              MacroMonitor is an autonomous, open-source quantitative macro data infrastructure designed exclusively for mathematical research, empirical tracking, and risk monitoring. 
              The system does <strong>not</strong> produce investment recommendations, financial advice, price targets, buy/sell ratings, or automated trading signals.
            </p>
            <p className="text-xs text-[var(--tx-1)] leading-relaxed">
              Nothing published on this website constitutes an offer, solicitation, or recommendation to buy or sell any security, derivative, commodity, or currency. 
              Historical percentiles and regime distributions do not guarantee future market behavior. All users should perform independent diligence or consult a qualified financial advisor before executing trades.
            </p>
            <p className="text-xs text-[var(--tx-1)] leading-relaxed">
              Under the Securities and Exchange Board of India (Investment Advisers) Regulations, 2013, and relevant US SEC / CFTC statutory frameworks, descriptive mathematical summaries of publicly available historical market data do not constitute investment advice.
            </p>
          </div>

          <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-2">
            <h4 className="text-xs font-mono font-semibold text-[var(--tx-0)] uppercase tracking-wider">
              Automated Continuous Integration Safety Scanner
            </h4>
            <p className="text-xs text-[var(--tx-1)] leading-relaxed">
              All narrative briefing strings and insight templates are continuously validated by continuous integration tests (<code className="text-[var(--accent)] font-mono">test_templates_safety.py</code> and <code className="text-[var(--accent)] font-mono">test_stress.py</code>) scanning for forbidden terms (e.g. <em>buy, sell, target price, will rally, will crash, accumulate, bullish recommendation</em>). Any detected advisory claim immediately aborts the deployment build.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
