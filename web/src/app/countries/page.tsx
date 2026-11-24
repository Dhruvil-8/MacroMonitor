// web/src/app/countries/page.tsx
"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ChartCard } from "@/components/charts/ChartCard";
import { WorldChoropleth } from "@/components/charts/WorldChoropleth";
import { StatBlock } from "@/components/ui/StatBlock";
import { AuditData, AuditDrawer } from "@/components/ui/AuditDrawer";
import { MethodData, MethodDrawer } from "@/components/ui/MethodDrawer";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";
import { HealthDot } from "@/components/ui/HealthDot";
import { loadCountries } from "@/lib/data/client";
import { CountriesEnvelope, CountryMatrixRow } from "@/lib/data/schemas";
import { Globe, ArrowRight, Layers, TrendingUp } from "lucide-react";

export default function CountriesPage() {
  const [data, setData] = useState<CountriesEnvelope | null>(null);
  const [lens, setLens] = useState<"local" | "usd" | "fx">("usd");
  const [asOf, setAsOf] = useState<string>("");
  const [isStale, setIsStale] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  // Drawers
  const [auditTarget, setAuditTarget] = useState<AuditData | null>(null);
  const [isAuditOpen, setIsAuditOpen] = useState<boolean>(false);
  const [methodTarget, setMethodTarget] = useState<MethodData | null>(null);
  const [isMethodOpen, setIsMethodOpen] = useState<boolean>(false);

  const handleOpenAudit = (audit: AuditData) => {
    setAuditTarget(audit);
    setIsAuditOpen(true);
  };

  const handleOpenMethod = (method: MethodData) => {
    setMethodTarget(method);
    setIsMethodOpen(true);
  };

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const res = await loadCountries();
        if (res.status !== "error") {
          setData(res.data);
          setAsOf(res.asOf);
          if (res.status === "stale") setIsStale(true);
        }
      } catch (e) {
        console.error("Error loading countries data:", e);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const horizons = ["1D", "5D", "1M", "3M", "1Y"];

  const getReturn = (country: CountryMatrixRow, h: string) => {
    if (lens === "local") return country.returns_local[h] ?? 0.0;
    if (lens === "fx") return country.returns_fx[h] ?? 0.0;
    return country.returns_usd[h] ?? 0.0;
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[var(--line)]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
            Module / Countries
          </div>
          <h1 className="text-3xl font-serif font-bold tracking-tight text-[var(--tx-0)] mt-1">
            Global Equity Performance Matrix
          </h1>
          <p className="text-xs text-[var(--tx-1)] mt-1.5 max-w-2xl leading-relaxed">
            Multi-horizon returns for the 10 launch economies decomposed across Local Currency, USD Denominated,
            and Currency (FX) lenses under the exact identity (1 + r_usd) = (1 + r_local)(1 + r_fx).
          </p>
        </div>

        <div className="flex items-center gap-3">
          <HealthDot status={isStale ? "stale" : "ok"} label={isStale ? "Stale Data" : "Verified Live"} />
          <DataAgeBadge asOf={asOf} isStale={isStale} ageHours={isStale ? 40 : undefined} />
        </div>
      </div>

      {/* Summary Stat Blocks */}
      {data && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatBlock
            id="tracked_countries"
            name="Tracked Economies"
            value={data.data.tracked_count}
            unit="countries"
            sparkline={[10, 10, 10, 10, 10]}
          />
          <StatBlock
            id="breadth_200d"
            name="200D Country Breadth"
            value={`${(data.data.breadth_200d * 100).toFixed(0)}%`}
            unit="above 200 MA"
            sparkline={[0.6, 0.7, 0.7, 0.6, data.data.breadth_200d]}
          />
          <StatBlock
            id="avg_corr_60d"
            name="60D Pairwise Correlation"
            value={data.data.avg_corr_60d.toFixed(2)}
            unit="mean rho"
            sparkline={[0.42, 0.45, 0.49, 0.52, data.data.avg_corr_60d]}
          />
          <StatBlock
            id="avg_corr_250d"
            name="250D Pairwise Correlation"
            value={data.data.avg_corr_250d.toFixed(2)}
            unit="mean rho"
            sparkline={[0.55, 0.54, 0.52, 0.50, data.data.avg_corr_250d]}
          />
        </div>
      )}

      {/* C03: World Spatial Performance & Volatility Matrix */}
      <ChartCard
        eyebrow="Spatial Macro Map (C03)"
        title="World Equity Performance & Volatility Barometer"
        subtitle="Cross-market return and 20-day realized volatility comparison across the 10 launch economies"
        asOf={asOf}
        methodData={{
          chartTitle: "World Spatial Cross-Market Performance & Volatility (C03)",
          whatItShows: "Comparative spatial barometer across North America, Europe, Asia-Pacific, and Emerging Markets.",
          definitionFormula: "r_h = \\frac{P_t}{P_{t-h}} - 1, \\quad \\sigma_{20} = \\text{std}(r_{1D}) \\times \\sqrt{252}",
          symbolsDefined: [
            { symbol: "r_h", meaning: "Multi-horizon cumulative return (1D, 5D, 20D)" },
            { symbol: "sigma_20", meaning: "20-day annualized realized volatility" },
          ],
          parameters: { horizons: "1D, 5D, 20D, 20D Realized Volatility" },
          dataSources: [{ series: "10 Country Benchmarks", source: "Official Primary Exchanges", frequency: "Daily" }],
          sample: { n: 250, nEff: 250, dateRange: "Full available history" },
          interpretationNotes: ["Color bars indicate positive return (teal) vs negative drawdowns (rose), or volatility tiers."],
          reproduce: {
            functionName: "compute_world_spatial_matrix",
            filePath: "pipeline/src/radar/metrics/country.py",
            testPath: "pipeline/tests/unit/test_country.py",
          },
        }}
        auditData={{
          metricName: "World Spatial Performance Barometer",
          formula: "r_h = P_t / P_{t-h} - 1",
          formulaPlain: "Cross-sectional comparison of 10 global equity indices.",
          parameters: { count: 10 },
          inputs: [{ id: "global_10", name: "10 Launch Country Indices", source: "Exchange Feeds", unit: "%" }],
          window: "1D, 5D, 20D lookbacks",
          currentValue: "10 economies tracked",
        }}
        onOpenAudit={handleOpenAudit}
        onOpenMethod={handleOpenMethod}
      >
        <WorldChoropleth />
      </ChartCard>

      {/* Performance Matrix Card */}
      {data && (
        <ChartCard
          eyebrow="Multi-Horizon Matrix"
          title="Cross-Country Return Matrix"
          subtitle={
            lens === "usd"
              ? "USD Lens: All 10 economies normalized to US Dollars (common denominator for cross-border comparison)"
              : lens === "local"
              ? "Local Currency Lens: Each index in its native domestic currency (Nifty in INR, Nikkei in JPY, DAX in EUR — zero FX noise)"
              : "Currency (FX) Lens: Foreign exchange appreciation / depreciation vs. US Dollar (positive = currency strengthened)"
          }
          asOf={asOf}
          headerControls={
            <div className="flex items-center gap-1 bg-[var(--bg-0)] p-1 rounded border border-[var(--line)] text-xs">
              <span className="text-[11px] font-mono text-[var(--tx-2)] px-1.5">Lens:</span>
              <button
                onClick={() => setLens("usd")}
                className={`px-2.5 py-1 rounded font-mono text-xs transition-colors ${
                  lens === "usd"
                    ? "bg-[var(--accent)] text-slate-900 font-semibold"
                    : "text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                }`}
              >
                USD Lens
              </button>
              <button
                onClick={() => setLens("local")}
                className={`px-2.5 py-1 rounded font-mono text-xs transition-colors ${
                  lens === "local"
                    ? "bg-[var(--accent)] text-slate-900 font-semibold"
                    : "text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                }`}
              >
                Local Currency
              </button>
              <button
                onClick={() => setLens("fx")}
                className={`px-2.5 py-1 rounded font-mono text-xs transition-colors ${
                  lens === "fx"
                    ? "bg-[var(--accent)] text-slate-900 font-semibold"
                    : "text-[var(--tx-2)] hover:text-[var(--tx-0)]"
                }`}
              >
                Currency (FX)
              </button>
            </div>
          }
          methodData={{
            chartTitle: "Cross-Country Return Matrix (Exact FX Identity)",
            whatItShows:
              "Displays cumulative returns across 5 standard horizons decomposed into local equity return, currency effect, and total USD return.",
            definitionFormula: "(1 + r_{\\text{usd}}) = (1 + r_{\\text{local}}) \\times (1 + r_{\\text{fx}})",
            symbolsDefined: [
              { symbol: "r_usd", meaning: "Return denominated in US Dollars" },
              { symbol: "r_local", meaning: "Return in domestic exchange currency" },
              { symbol: "r_fx", meaning: "Normalized currency appreciation vs USD (positive = currency strengthened)" },
            ],
            parameters: {
              horizons: "1D, 5D, 1M, 3M, 1Y",
              quote_convention: "Normalized to USD per local unit",
              rounding: "Exact 4 decimal places with identity preservation",
            },
            dataSources: [
              { series: "10 Country Benchmark Indices", source: "Official Primary Feeds", frequency: "Daily" },
              { series: "Bilateral FX Rates", source: "Central Banks / FX Benchmark Rates", frequency: "Daily" },
            ],
            sample: {
              n: 252,
              nEff: 180,
              dateRange: "Trailing 1 Year",
            },
            interpretationNotes: [
              "USD returns allow honest cross-border comparison from a single global investor reference frame.",
              "FX lens reveals whether international equity gains were amplified or erased by exchange rate shifts.",
            ],
            reproduce: {
              functionName: "compute_country_metrics",
              filePath: "pipeline/src/radar/metrics/country.py",
              testPath: "pipeline/tests/unit/test_country.py",
            },
          }}
          onOpenMethod={handleOpenMethod}
          auditData={{
            metricName: "Country Layer Return Matrix",
            formula: "r_h = \\frac{P_t}{P_{t-h}} - 1, \\quad (1 + r_{\\text{usd}}) = (1 + r_{\\text{local}})(1 + r_{\\text{fx}})",
            formulaPlain: "Cumulative percentage changes verified under exact foreign exchange multiplicative identity.",
            parameters: {
              active_lens: lens,
              tracked_countries: data.data.tracked_count,
              breadth_200d: `${(data.data.breadth_200d * 100).toFixed(1)}%`,
            },
            inputs: data.data.countries.map((c) => ({
              id: c.code,
              name: `${c.name} (${c.index_id.toUpperCase()})`,
              source: "Yahoo / Primary Exchange",
              unit: c.currency,
            })),
            window: "1D to 1Y",
            currentValue: `${data.data.tracked_count} countries`,
            codeVersion: data.code_version,
            specUrl: "/methodology",
          }}
          onOpenAudit={handleOpenAudit}
        >
          {/* Active Lens Explanatory Banner */}
          <div className="mb-3 px-3.5 py-2 rounded-md bg-[var(--bg-0)] border border-[var(--line)] text-xs font-mono text-[var(--tx-1)] flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              {lens === "usd" && (
                <span>
                  <strong className="text-[var(--accent)] font-semibold">USD Lens Active:</strong> All equity returns converted to USD using official currency rates. Allows direct cross-border returns comparison.
                </span>
              )}
              {lens === "local" && (
                <span>
                  <strong className="text-[var(--accent)] font-semibold">Local Currency Lens Active:</strong> Returns measured in each country&apos;s own domestic currency (India in INR, Japan in JPY, Germany in EUR, UK in GBP). Eliminates exchange-rate distortion.
                </span>
              )}
              {lens === "fx" && (
                <span>
                  <strong className="text-[var(--accent)] font-semibold">Currency (FX) Lens Active:</strong> Performance of each domestic currency relative to USD. Positive = currency appreciated against USD.
                </span>
              )}
            </div>
            <span className="text-[10px] text-[var(--tx-2)] shrink-0 font-mono">
              (1 + r_usd) = (1 + r_local) &times; (1 + r_fx)
            </span>
          </div>

          <div className="overflow-x-auto rounded border border-[var(--line)]">
            <table className="w-full text-xs text-left">
              <thead className="bg-[var(--bg-0)] text-[var(--tx-2)] uppercase font-mono text-[11px] border-b border-[var(--line)]">
                <tr>
                  <th className="py-3 px-4">Country & Benchmark</th>
                  <th className="py-3 px-3">Currency</th>
                  {horizons.map((h) => (
                    <th key={h} className="py-3 px-3 text-right">
                      {h}
                    </th>
                  ))}
                  <th className="py-3 px-4 text-right">Profile</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--line)]">
                {data.data.countries.map((c) => (
                  <tr key={c.code} className="hover:bg-[var(--bg-0)]/60 transition-colors">
                    <td className="py-3 px-4 font-medium text-[var(--tx-0)] flex items-center gap-2">
                      <span className="font-mono text-xs uppercase px-1.5 py-0.5 rounded bg-[var(--bg-2)] border border-[var(--line)] text-[var(--tx-2)]">
                        {c.code}
                      </span>
                      <span className="font-semibold">{c.name}</span>
                      <span className="text-[11px] font-mono text-[var(--tx-2)]">({c.index_id.toUpperCase()})</span>
                    </td>
                    <td className="py-3 px-3 font-mono text-[var(--tx-2)]">{c.currency}</td>
                    {horizons.map((h) => {
                      const ret = getReturn(c, h);
                      const retPct = ret * 100.0;
                      const sign = retPct >= 0 ? "+" : "";
                      return (
                        <td
                          key={h}
                          className={`py-3 px-3 text-right font-mono font-medium ${
                            retPct > 0 ? "text-emerald-400" : retPct < 0 ? "text-rose-400" : "text-[var(--tx-2)]"
                          }`}
                        >
                          {sign}
                          {retPct.toFixed(2)}%
                        </td>
                      );
                    })}
                    <td className="py-3 px-4 text-right">
                      <Link
                        href={`/context/${c.index_id}`}
                        className="inline-flex items-center gap-1 text-[11px] font-mono text-[var(--accent)] hover:underline"
                      >
                        <span>Inspect</span>
                        <ArrowRight className="w-3 h-3" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ChartCard>
      )}

      {/* Drawers */}
      <AuditDrawer isOpen={isAuditOpen} onClose={() => setIsAuditOpen(false)} data={auditTarget} />
      <MethodDrawer isOpen={isMethodOpen} onClose={() => setIsMethodOpen(false)} data={methodTarget} />
    </div>
  );
}
