// web/src/app/india/page.tsx
"use client";

import React, { useEffect, useState } from "react";
import { ChartCard } from "@/components/charts/ChartCard";
import { StatBlock } from "@/components/ui/StatBlock";
import { AuditDrawer, AuditData } from "@/components/ui/AuditDrawer";
import { FiiDiiFlowChart } from "@/components/charts/FiiDiiFlowChart";
import { OiBuildupMap } from "@/components/charts/OiBuildupMap";
import { IndiaFlowsEnvelope } from "@/lib/data/schemas";

export default function IndiaPage() {
  const [dataEnvelope, setDataEnvelope] = useState<IndiaFlowsEnvelope | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Traceability audit drawer state
  const [auditTarget, setAuditTarget] = useState<AuditData | null>(null);
  const [isAuditOpen, setIsAuditOpen] = useState<boolean>(false);

  const handleOpenAudit = (audit: AuditData) => {
    setAuditTarget(audit);
    setIsAuditOpen(true);
  };

  useEffect(() => {
    async function loadIndiaFlows() {
      try {
        const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";
        const res = await fetch(`${basePath}/data/v1/india_flows.json`);
        if (!res.ok) throw new Error(`HTTP ${res.status} while loading India flows data`);
        const json = await res.json();
        setDataEnvelope(json);
      } catch (err: any) {
        setError(err.message || "Failed to load India flows data");
      } finally {
        setLoading(false);
      }
    }
    loadIndiaFlows();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div>
          <div className="text-[11px] font-mono text-[var(--tx-2)] uppercase tracking-wider">Module / India</div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--tx-0)]">India Flows & Positioning Analytics</h1>
        </div>
        <div className="p-12 text-center text-xs font-mono text-[var(--tx-2)] bg-[var(--bg-1)] border border-[var(--line)] rounded-lg">
          Loading India institutional flows and derivatives positioning...
        </div>
      </div>
    );
  }

  if (error || !dataEnvelope) {
    return (
      <div className="space-y-6">
        <div>
          <div className="text-[11px] font-mono text-[var(--tx-2)] uppercase tracking-wider">Module / India</div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--tx-0)]">India Flows & Positioning Analytics</h1>
        </div>
        <div className="p-6 border border-red-500/30 text-xs font-mono text-red-400 bg-[var(--bg-1)] rounded-lg">
          Error loading India flows dataset: {error}
        </div>
      </div>
    );
  }

  const { data, as_of, code_version } = dataEnvelope;
  const audit = data.audit;

  const indiaVixAudit: AuditData = {
    metricName: "India VIX Percentile Ribbon",
    formula: "\\text{Percentile} = \\frac{1}{N}\\sum \\mathbf{1}_{\\{V_s \\le V_t\\}} \\times 100",
    formulaPlain: "Empirical expanding historical percentile rank of India VIX against all past trading days.",
    parameters: { min_obs: 30 },
    inputs: [{ id: "indiavix", name: "India VIX", source: "NSE India", unit: "vol_pts" }],
    window: "Full sample expanding",
    currentValue: data.india_vix,
    percentile: data.india_vix_percentile,
    codeVersion: code_version,
    specUrl: "/methodology",
  };

  const fiiAudit: AuditData = {
    metricName: "FII Institutional Cash Flow",
    formula: "S_{20D} = \\sum_{k=0}^{19} F_{t-k}^{\\text{FII}}, \\quad P = \\text{Rank}(S_{20D})",
    formulaPlain: "Daily FII net cash purchases and 20-day cumulative sum in INR Crores.",
    parameters: { rolling_windows: "5, 20" },
    inputs: [{ id: "fii_cash", name: "FII Cash Market Net", source: "NSE / SEBI", unit: "inr_cr" }],
    window: "20-Day Cumulative Sum",
    currentValue: `₹${data.fii_net_1d.toLocaleString()} Cr`,
    percentile: data.fii_20d_percentile,
    n: audit.n,
    nEff: audit.n_eff,
    codeVersion: code_version,
    specUrl: "/methodology",
  };

  const diiAudit: AuditData = {
    metricName: "DII Institutional Cash Flow",
    formula: "S_{20D} = \\sum_{k=0}^{19} F_{t-k}^{\\text{DII}}, \\quad P = \\text{Rank}(S_{20D})",
    formulaPlain: "Daily DII net cash purchases and 20-day cumulative sum in INR Crores.",
    parameters: { rolling_windows: "5, 20" },
    inputs: [{ id: "dii_cash", name: "DII Cash Market Net", source: "NSE / SEBI", unit: "inr_cr" }],
    window: "20-Day Cumulative Sum",
    currentValue: `₹${data.dii_net_1d.toLocaleString()} Cr`,
    percentile: data.dii_20d_percentile,
    n: audit.n,
    nEff: audit.n_eff,
    codeVersion: code_version,
    specUrl: "/methodology",
  };

  const pcrAudit: AuditData = {
    metricName: "Nifty Put-Call Ratio (PCR)",
    formula: "\\text{PCR} = \\frac{\\sum \\text{Put Open Interest}}{\\sum \\text{Call Open Interest}}",
    formulaPlain: "Total near-month put option open interest divided by call option open interest.",
    parameters: {},
    inputs: [{ id: "nifty_fo", name: "Nifty Option Chain", source: "NSE Derivatives", unit: "ratio" }],
    window: "Daily Close",
    currentValue: data.pcr,
    codeVersion: code_version,
    specUrl: "/methodology",
  };

  const deliveryAudit: AuditData = {
    metricName: "Nifty Volume-Weighted Delivery %",
    formula: "\\text{Delivery \\%} = \\frac{\\sum \\text{Delivered Quantity}}{\\sum \\text{Traded Quantity}} \\times 100",
    formulaPlain: "Volume-weighted delivery percentage across Nifty 50 constituent equities.",
    parameters: {},
    inputs: [{ id: "nifty50_delivery", name: "Nifty Equity Bhavcopy", source: "NSE Cash Market", unit: "percent" }],
    window: "Daily",
    currentValue: `${data.delivery_pct_index}%`,
    percentile: data.delivery_pct_percentile,
    codeVersion: code_version,
    specUrl: "/methodology",
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-[var(--line)]">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">Module / India</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--bg-2)] border border-[var(--line)] text-[var(--accent)]">
              As of {as_of}
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--tx-0)] mt-0.5">
            India Flows & Positioning Analytics
          </h1>
          <p className="text-xs text-[var(--tx-1)] mt-1">
            FII/DII cash market flows, F&O open interest buildup classification, Put-Call Ratio, and delivery percentage.
          </p>
        </div>

        <button
          onClick={() => handleOpenAudit(fiiAudit)}
          className="self-start sm:self-auto flex items-center gap-1.5 text-xs font-mono px-3 py-1.5 rounded bg-[var(--bg-1)] border border-[var(--line)] text-[var(--tx-1)] hover:text-[var(--tx-0)] hover:border-[var(--accent)] transition-colors"
        >
          <span className="w-2 h-2 rounded-full bg-[var(--accent)]"></span>
          Audit Lineage
        </button>
      </div>

      {/* 5 KPI Stat Blocks */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        <StatBlock
          id="india_vix"
          name="India VIX"
          value={data.india_vix}
          change1d={-0.35}
          unit="vol_pts"
          percentile={data.india_vix_percentile}
          sparkline={[15.2, 16.1, 16.8, 17.5, 17.32]}
          onOpenAudit={handleOpenAudit}
          auditPayload={indiaVixAudit}
        />
        <StatBlock
          id="fii_cash"
          name="FII Net Cash (1D)"
          value={data.fii_net_1d}
          change1d={data.fii_net_5d}
          unit="inr_cr"
          percentile={data.fii_20d_percentile}
          sparkline={data.fii_daily.slice(-15)}
          onOpenAudit={handleOpenAudit}
          auditPayload={fiiAudit}
        />
        <StatBlock
          id="dii_cash"
          name="DII Net Cash (1D)"
          value={data.dii_net_1d}
          change1d={data.dii_net_5d}
          unit="inr_cr"
          percentile={data.dii_20d_percentile}
          sparkline={data.dii_daily.slice(-15)}
          onOpenAudit={handleOpenAudit}
          auditPayload={diiAudit}
        />
        <StatBlock
          id="pcr"
          name="Put-Call Ratio (PCR)"
          value={data.pcr}
          change1d={0.02}
          unit="ratio"
          sparkline={[1.02, 1.05, 1.06, 1.07, 1.08]}
          onOpenAudit={handleOpenAudit}
          auditPayload={pcrAudit}
        />
        <StatBlock
          id="delivery_pct"
          name="Nifty Delivery %"
          value={data.delivery_pct_index}
          change1d={1.2}
          unit="pct"
          percentile={data.delivery_pct_percentile}
          sparkline={[51.2, 52.4, 53.0, 53.8, 54.2]}
          onOpenAudit={handleOpenAudit}
          auditPayload={deliveryAudit}
        />
      </div>

      {/* Main Charts: FII/DII Institutional Flows and F&O OI Buildup */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Institutional Flows Chart */}
        <div className="lg:col-span-7">
          <ChartCard
            eyebrow="Cash Market Flows"
            title="FII & DII Institutional Net Purchases"
            subtitle="Daily net purchases and 20-day cumulative absorption in INR Crores (₹ Cr)"
            asOf={as_of}
            auditData={fiiAudit}
            onOpenAudit={handleOpenAudit}
          >
            <FiiDiiFlowChart
              dates={data.flow_dates}
              fiiDaily={data.fii_daily}
              diiDaily={data.dii_daily}
              fiiCum20d={data.fii_cum_20d}
              diiCum20d={data.dii_cum_20d}
            />
          </ChartCard>
        </div>

        {/* F&O OI Buildup Matrix */}
        <div className="lg:col-span-5">
          <ChartCard
            eyebrow="Derivatives Positioning"
            title="F&O Open Interest Buildup"
            subtitle="Underlying price direction vs open interest expansion across indices and key stocks"
            asOf={as_of}
            auditData={pcrAudit}
            onOpenAudit={handleOpenAudit}
          >
            <OiBuildupMap items={data.oi_buildup} pcr={data.pcr} />
          </ChartCard>
        </div>
      </div>

      {/* Narrative & Flow Mechanics Note */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-2">
          <div className="text-xs font-mono font-semibold text-[var(--tx-0)] flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]"></span>
            FII vs DII Absorption Dynamic
          </div>
          <p className="text-xs text-[var(--tx-1)] leading-relaxed">
            Domestic Institutional Investors (DIIs) systematically provide countercyclical liquidity to the Indian cash market via domestic SIP mutual fund inflows. 
            When Foreign Institutional Investors (FIIs) sell during global risk-off regimes, DII net buying cushions drawdown severity, transforming the structural volatility profile of the Nifty 50.
          </p>
        </div>

        <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-2">
          <div className="text-xs font-mono font-semibold text-[var(--tx-0)] flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--warn)]"></span>
            Delivery & Buildup Discipline
          </div>
          <p className="text-xs text-[var(--tx-1)] leading-relaxed">
            Spot delivery percentages above the 60th percentile alongside <span className="text-emerald-400">Long Buildup</span> reflect genuine long-term capital commitment rather than leveraged intraday churning. 
            Conversely, high open interest expansion with falling spot prices (<span className="text-rose-400">Short Buildup</span>) confirms aggressive institutional hedging or directional downside pressure.
          </p>
        </div>
      </div>

      {/* Audit Drawer */}
      <AuditDrawer
        isOpen={isAuditOpen}
        onClose={() => setIsAuditOpen(false)}
        data={auditTarget}
      />
    </div>
  );
}
