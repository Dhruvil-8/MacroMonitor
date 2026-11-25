"use client";

import { useEffect, useState } from "react";
import { GlobalSlowEnvelope, ReerValuationItem, ForeignTreasuryHoldingItem, CentralBankAssetItem } from "@/lib/data/schemas";
import { loadGlobalSlow } from "@/lib/data/client";
import { HealthDot } from "@/components/ui/HealthDot";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { Clock, DollarSign, Globe, Landmark, Shield, TrendingDown, TrendingUp } from "lucide-react";

export default function GlobalSlowPage() {
  const [data, setData] = useState<GlobalSlowEnvelope | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadGlobalSlow()
      .then((res) => {
        if (res.status === "error") {
          setError(res.error || "Failed to load slow global data");
        } else {
          setData(res.data);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) {
    return (
      <div className="rounded-[6px] border border-rose-500 bg-[#0c1017] p-5 text-sm text-rose-400">
        Failed to load Slow Global Macro data: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-96 rounded-[6px] border border-[#1d2633] bg-[#0c1017] animate-pulse flex items-center justify-center font-mono text-xs text-[#6f7c90]">
        Loading BIS Real Effective Exchange Rates & US Treasury TIC Data...
      </div>
    );
  }

  const gs = (data as any)?.data || data;

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-12">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1d2633]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[#6f7c90]">
            Structural Macro &amp; Reserve Adequacy
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[#e8edf5] mt-0.5">
            Slow Global Layer: REER, Foreign Holdings & Central Banks
          </h1>
          <p className="text-xs text-[#8c9bb0] mt-1">
            Low-frequency structural backdrop: BIS real effective exchange rates, US Treasury foreign holders (TIC), global central bank balance sheets, and India FX adequacy.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot status="ok" label="BIS & TIC Synchronized" />
          <DataAgeBadge asOf={data.as_of || gs.as_of || ""} />
        </div>
      </div>

      {/* Lag Banner */}
      <div className="p-3 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] text-xs font-mono text-[var(--tx-2)] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-sky-400 shrink-0" />
          <span><b>Publication Lag Disclosure:</b> {gs.lag_label}</span>
        </div>
        <span className="text-[10px] text-[var(--tx-3)] font-semibold">Strict Point-in-Time Alignment</span>
      </div>

      {/* Narrative Card */}
      <div className="p-4 bg-[var(--bg-0)] rounded-lg border border-[var(--line)] font-mono text-xs text-[var(--tx-1)]">
        {gs.interpretation}
      </div>

      {/* 1. Currency Valuation Matrix (BIS REER) */}
      <div className="bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)] space-y-3 font-mono text-xs">
        <div className="flex items-center justify-between border-b border-[var(--line)] pb-2">
          <div className="flex items-center gap-2">
            <Globe className="w-4 h-4 text-[var(--accent)]" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--tx-0)]">
              Currency Valuation Matrix (BIS Real Effective Exchange Rates)
            </h3>
          </div>
          <span className="text-[10px] text-[var(--tx-2)]">Expanding Historical Percentiles • Base 100</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {gs.reer?.map((r: ReerValuationItem) => (
            <div key={r.id} className="p-3 bg-[var(--bg-1)] rounded border border-[var(--line)] space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-bold text-[var(--tx-0)]">{r.name}</span>
                <span
                  className={`text-[9px] px-1.5 py-0.2 rounded font-bold uppercase border ${
                    r.valuation === "overvalued"
                      ? "bg-rose-950/70 text-rose-400 border-rose-800"
                      : r.valuation === "undervalued"
                      ? "bg-emerald-950/70 text-emerald-400 border-emerald-800"
                      : "bg-slate-900 text-slate-300 border-slate-700"
                  }`}
                >
                  {r.valuation}
                </span>
              </div>
              <div className="text-xl font-bold text-[var(--tx-0)]">{r.current_index.toFixed(1)}</div>
              <div className="flex justify-between text-[11px] text-[var(--tx-2)] pt-1 border-t border-[var(--line)]">
                <span>Rank: <strong className="text-[var(--accent)]">{r.percentile.toFixed(0)}th pct</strong></span>
                <span>1Y Δ: <b className={r.chg_1y >= 0 ? "text-emerald-400" : "text-rose-400"}>{r.chg_1y > 0 ? "+" : ""}{r.chg_1y.toFixed(1)}</b></span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 2. Foreign Holdings of US Treasuries (TIC) & Central Bank Assets */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 font-mono text-xs">
        {/* TIC Foreign Holders */}
        <div className="bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)] space-y-3">
          <div className="flex items-center justify-between border-b border-[var(--line)] pb-2">
            <div className="flex items-center gap-2">
              <Landmark className="w-4 h-4 text-sky-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--tx-0)]">
                Foreign Treasury Holdings (US Treasury TIC)
              </h3>
            </div>
            <span className="text-[10px] text-[var(--tx-2)]">Major Custody Holders</span>
          </div>

          <div className="divide-y divide-[var(--line)]">
            {gs.foreign_treasury_holdings?.map((h: ForeignTreasuryHoldingItem) => (
              <div key={h.country} className="py-2 flex items-center justify-between">
                <div>
                  <div className="font-bold text-[var(--tx-0)]">{h.country}</div>
                  <div className="text-[10px] text-[var(--tx-2)]">{h.share_pct}% of total foreign stock</div>
                </div>
                <div className="text-right">
                  <div className="font-bold text-[var(--tx-0)]">${h.holdings_bn.toFixed(1)}B</div>
                  <div className={`text-[10px] ${h.chg_1y_bn >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    1Y: {h.chg_1y_bn > 0 ? "+" : ""}${h.chg_1y_bn.toFixed(1)}B
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Global Central Bank Balance Sheets */}
        <div className="bg-[var(--bg-0)] p-4 rounded-lg border border-[var(--line)] space-y-3">
          <div className="flex items-center justify-between border-b border-[var(--line)] pb-2">
            <div className="flex items-center gap-2">
              <DollarSign className="w-4 h-4 text-emerald-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--tx-0)]">
                Central Bank Assets (USD Trillions)
              </h3>
            </div>
            <span className="text-[10px] text-[var(--tx-2)]">Fed, ECB, BOJ Total Assets</span>
          </div>

          <div className="divide-y divide-[var(--line)]">
            {gs.central_bank_assets?.map((cb: CentralBankAssetItem) => (
              <div key={cb.central_bank} className="py-2.5 flex items-center justify-between">
                <div>
                  <div className="font-bold text-[var(--tx-0)]">{cb.central_bank}</div>
                  <div className="text-[10px] text-[var(--tx-2)]">As of: {cb.as_of}</div>
                </div>
                <div className="text-right">
                  <div className="font-bold text-emerald-400 text-base">${cb.assets_trillion_usd.toFixed(2)}T</div>
                  <div className={`text-[10px] ${cb.chg_1y_pct >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    1Y: {cb.chg_1y_pct > 0 ? "+" : ""}{cb.chg_1y_pct.toFixed(1)}% (QT / Runoff)
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* India Macro Adequacy */}
          <div className="pt-2 border-t border-[var(--line)] space-y-2">
            <h4 className="text-[11px] font-bold uppercase text-[var(--tx-0)] flex items-center gap-1.5">
              <Shield className="w-3.5 h-3.5 text-teal-400" /> India Reserve Adequacy (RBI)
            </h4>
            <div className="grid grid-cols-2 gap-2 text-[11px]">
              <div className="p-2 bg-[var(--bg-1)] rounded border border-[var(--line)]">
                <span className="text-[var(--tx-2)]">FX Reserves:</span>
                <div className="font-bold text-[var(--tx-0)] text-sm">${gs.india_macro?.fx_reserves_bn_usd}B</div>
                <div className="text-[10px] text-teal-400">~{gs.india_macro?.import_cover_months} Mos Import Cover</div>
              </div>
              <div className="p-2 bg-[var(--bg-1)] rounded border border-[var(--line)]">
                <span className="text-[var(--tx-2)]">12M FPI Net Flows:</span>
                <div className={`font-bold text-sm ${gs.india_macro?.fpi_net_total_12m_bn >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {gs.india_macro?.fpi_net_total_12m_bn > 0 ? "+" : ""}${gs.india_macro?.fpi_net_total_12m_bn}B
                </div>
                <div className="text-[10px] text-[var(--tx-2)]">Equity: ${gs.india_macro?.fpi_equity_12m_bn}B • Debt: +${gs.india_macro?.fpi_debt_12m_bn}B</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
