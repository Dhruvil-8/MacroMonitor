// web/src/app/liquidity/page.tsx
"use client";

import React, { useEffect, useState } from "react";
import { ChartCard } from "@/components/charts/ChartCard";
import { StatBlock } from "@/components/ui/StatBlock";
import { AuditDrawer, AuditData } from "@/components/ui/AuditDrawer";
import { NetLiquidityChart } from "@/components/charts/NetLiquidityChart";
import { LiquidityEnvelope } from "@/lib/data/schemas";

export default function LiquidityPage() {
  const [dataEnvelope, setDataEnvelope] = useState<LiquidityEnvelope | null>(null);
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
    async function loadLiquidity() {
      try {
        const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";
        const res = await fetch(`${basePath}/data/v1/liquidity.json`);
        if (!res.ok) throw new Error(`HTTP ${res.status} while loading liquidity data`);
        const json = await res.json();
        setDataEnvelope(json);
      } catch (err: any) {
        setError(err.message || "Failed to load liquidity data");
      } finally {
        setLoading(false);
      }
    }
    loadLiquidity();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div>
          <div className="text-[11px] font-mono text-[var(--tx-2)] uppercase tracking-wider">Module / Liquidity</div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--tx-0)]">Global Liquidity & Funding Monitor</h1>
        </div>
        <div className="p-12 text-center text-xs font-mono text-[var(--tx-2)] bg-[var(--bg-1)] border border-[var(--line)] rounded-lg">
          Loading liquidity decomposition...
        </div>
      </div>
    );
  }

  if (error || !dataEnvelope) {
    return (
      <div className="space-y-6">
        <div>
          <div className="text-[11px] font-mono text-[var(--tx-2)] uppercase tracking-wider">Module / Liquidity</div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--tx-0)]">Global Liquidity & Funding Monitor</h1>
        </div>
        <div className="p-6 border border-red-500/30 text-xs font-mono text-red-400 bg-[var(--bg-1)] rounded-lg">
          Error loading liquidity dataset: {error}
        </div>
      </div>
    );
  }

  const { data, as_of, code_version } = dataEnvelope;
  const audit = data.audit;

  const netLiqAudit: AuditData = {
    metricName: "US Net Liquidity",
    formula: "\\text{Net Liquidity} = \\frac{\\text{WALCL}}{1000} - \\frac{\\text{WTREGEN}}{1000} - \\text{RRPONTSYD}",
    formulaPlain: audit.formula,
    parameters: audit.parameters,
    inputs: [
      { id: "fed_assets", name: "Federal Reserve Total Assets", source: "FRED: WALCL", unit: "Millions USD" },
      { id: "tga", name: "Treasury General Account", source: "FRED: WTREGEN", unit: "Millions USD" },
      { id: "rrp", name: "Overnight Reverse Repo", source: "FRED: RRPONTSYD", unit: "Billions USD" },
    ],
    window: "4-Week (20D) and 13-Week (65D) expanding",
    currentValue: `$${data.current_net_liquidity.toLocaleString()}B`,
    percentile: data.change_4w_pct,
    n: audit.n,
    nEff: audit.n_eff,
    codeVersion: code_version,
    specUrl: "/methodology",
  };

  const fedAssetsAudit: AuditData = {
    metricName: "Federal Reserve Total Assets",
    formula: "\\text{Assets} = \\frac{\\text{WALCL}}{1000}",
    formulaPlain: "Fed Total Assets converted from Millions to Billions of USD.",
    parameters: {},
    inputs: [{ id: "fed_assets", name: "Fed Assets", source: "FRED: WALCL", unit: "usd_bn" }],
    window: "Weekly Wednesday H.4.1",
    currentValue: `$${(data.fed_assets[data.fed_assets.length - 1] || 0).toLocaleString()}B`,
    codeVersion: code_version,
    specUrl: "/methodology",
  };

  const tgaAudit: AuditData = {
    metricName: "US Treasury General Account (TGA)",
    formula: "\\text{TGA} = \\frac{\\text{WTREGEN}}{1000}",
    formulaPlain: "Treasury Operating Cash Balance at the Fed in Billions of USD.",
    parameters: {},
    inputs: [{ id: "tga", name: "Treasury General Account", source: "FRED: WTREGEN", unit: "usd_bn" }],
    window: "Weekly Wednesday Financial Statements",
    currentValue: `$${(data.tga[data.tga.length - 1] || 0).toLocaleString()}B`,
    codeVersion: code_version,
    specUrl: "/methodology",
  };

  const rrpAudit: AuditData = {
    metricName: "Overnight Reverse Repurchase Agreements (RRP)",
    formula: "\\text{RRP} = \\text{RRPONTSYD}",
    formulaPlain: "Cash parked by money market funds at the NY Fed overnight facility in Billions of USD.",
    parameters: {},
    inputs: [{ id: "rrp", name: "Reverse Repo", source: "FRED: RRPONTSYD", unit: "usd_bn" }],
    window: "Daily Business Day NY Fed",
    currentValue: `$${(data.rrp[data.rrp.length - 1] || 0).toFixed(2)}B`,
    codeVersion: code_version,
    specUrl: "/methodology",
  };

  const stableAudit: AuditData = {
    metricName: "Total Stablecoins Circulating Supply",
    formula: "\\text{Stablecoins Supply} = \\sum \\text{MarketCap}_{\\text{peggedUSD}}",
    formulaPlain: "Total circulating USD-pegged stablecoin aggregate across all networks.",
    parameters: { window_days: 30 },
    inputs: [{ id: "stablecoins", name: "USD Stablecoin Aggregate", source: "DefiLlama / On-Chain", unit: "usd_bn" }],
    window: "30-Day Rolling Momentum",
    currentValue: `$${data.stablecoins_total_bn.toFixed(1)}B`,
    percentile: 65.0,
    codeVersion: code_version,
    specUrl: "/methodology",
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-[var(--line)]">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">Module / Liquidity</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--bg-2)] border border-[var(--line)] text-[var(--accent)]">
              As of {as_of}
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--tx-0)] mt-0.5">
            Global Liquidity & Funding Monitor
          </h1>
          <p className="text-xs text-[var(--tx-1)] mt-1">
            US Net Liquidity (WALCL − WTREGEN − RRPONTSYD), Treasury absorption, reverse repo exhaustion, and equity overlay.
          </p>
        </div>

        <button
          onClick={() => handleOpenAudit(netLiqAudit)}
          className="self-start sm:self-auto flex items-center gap-1.5 text-xs font-mono px-3 py-1.5 rounded bg-[var(--bg-1)] border border-[var(--line)] text-[var(--tx-1)] hover:text-[var(--tx-0)] hover:border-[var(--accent)] transition-colors"
        >
          <span className="w-2 h-2 rounded-full bg-[var(--accent)]"></span>
          Audit Lineage
        </button>
      </div>

      {/* KPI Top Stat Blocks */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        <StatBlock
          id="net_liq"
          name="US Net Liquidity"
          value={data.current_net_liquidity}
          change1d={data.change_4w}
          unit="$B"
          percentile={data.change_4w_pct}
          sparkline={data.net_liquidity.slice(-15)}
          onOpenAudit={handleOpenAudit}
          auditPayload={netLiqAudit}
        />
        <StatBlock
          id="fed_assets"
          name="Fed Total Assets"
          value={data.fed_assets[data.fed_assets.length - 1] || 0}
          change1d={0}
          unit="$B"
          sparkline={data.fed_assets.slice(-15)}
          onOpenAudit={handleOpenAudit}
          auditPayload={fedAssetsAudit}
        />
        <StatBlock
          id="tga"
          name="Treasury Cash (TGA)"
          value={data.tga[data.tga.length - 1] || 0}
          change1d={0}
          unit="$B"
          sparkline={data.tga.slice(-15)}
          onOpenAudit={handleOpenAudit}
          auditPayload={tgaAudit}
        />
        <StatBlock
          id="rrp"
          name="Reverse Repo (RRP)"
          value={data.rrp[data.rrp.length - 1] || 0}
          change1d={0}
          unit="$B"
          sparkline={data.rrp.slice(-15)}
          onOpenAudit={handleOpenAudit}
          auditPayload={rrpAudit}
        />
        <StatBlock
          id="stablecoins"
          name="Total Stablecoins"
          value={data.stablecoins_total_bn}
          change1d={data.stablecoins_30d_pct}
          unit="$B"
          percentile={70}
          sparkline={[275, 277, 278, 280, 282.58]}
          onOpenAudit={handleOpenAudit}
          auditPayload={stableAudit}
        />
      </div>

      {/* Main Liquidity Decomposition & Overlay */}
      <ChartCard
        eyebrow="Funding Dynamics"
        title="Federal Reserve Net Liquidity Decomposition & Equity Overlay"
        subtitle="Fed Total Assets (+) minus Treasury Cash at Fed (−) minus Overnight Reverse Repo (−), overlaid with S&P 500"
        asOf={as_of}
        auditData={netLiqAudit}
        onOpenAudit={handleOpenAudit}
      >
        <NetLiquidityChart
          dates={data.dates}
          netLiquidity={data.net_liquidity}
          fedAssets={data.fed_assets}
          tga={data.tga}
          rrp={data.rrp}
          spxOverlay={data.spx_overlay}
          rollingCorr60d={data.rolling_corr_60d}
        />
      </ChartCard>

      {/* Narrative & Methodology Note */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-2">
          <div className="text-xs font-mono font-semibold text-[var(--tx-0)] flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]"></span>
            Mechanics of Fed Net Liquidity
          </div>
          <p className="text-xs text-[var(--tx-1)] leading-relaxed">
            Net Liquidity measures the amount of commercial bank reserves circulating in the financial system. 
            When the Federal Reserve expands its balance sheet (<span className="text-[var(--pos)]">Fed Assets</span>), liquidity is added. 
            When the Treasury builds cash reserves (<span className="text-[var(--warn)]">TGA</span>) or money market funds park cash in the Fed's overnight facility (<span className="text-[var(--neg)]">RRP</span>), liquidity is drained from private credit markets.
          </p>
        </div>

        <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-2">
          <div className="text-xs font-mono font-semibold text-[var(--tx-0)] flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--warn)]"></span>
            Empirical Caution & Causality
          </div>
          <p className="text-xs text-[var(--tx-1)] leading-relaxed">
            Association, not mechanical causation: While changes in central bank reserves historically track shifts in equity market multiples and risk-asset volatility, net liquidity is not a mechanical deterministic trading trigger. Shifts in foreign FX reserves, commercial bank loan creation, and leverage also drive broad financial conditions.
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
