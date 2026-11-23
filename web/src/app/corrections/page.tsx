import { ShieldCheck } from "lucide-react";

const CORRECTIONS_LOG = [
  {
    date: "2026-11-21",
    scope: "Macro Universe (v2.0)",
    finding: "Discontinued legacy FRED OECD MEI IDs (EA19CPALTT01GYM, GBRPROINDMISMEI, JPNCPIALLMINMEI, INDPROINDMISMEI) halted updates between 2021 and 2024.",
    remediation: "Replaced Eurozone with direct ECB SDMX monthly feeds, UK with ONS CSV + OECD CLI + SONIA, and downgraded Japan & India to Grade C (raw active indicators only, zero forced regime labels).",
  },
  {
    date: "2026-11-21",
    scope: "SEC Form 13F-HR Units (v2.0)",
    finding: "SEC EDGAR Form 13F XML <value> transitioned from thousands of dollars pre-2023 to exact whole USD dollars post-2023, and multiple <infoTable> rows share the same CUSIP across sub-managers.",
    remediation: "Parser sums all <infoTable> value and share rows per CUSIP using exact USD dollar scaling for current filings.",
  },
  {
    date: "2026-11-21",
    scope: "SEC Form 4 Filtering (v2.0)",
    finding: "Routine Rule 10b5-1 scheduled executive transactions and tax-withholding dispositions inflate naive insider counts.",
    remediation: "Form 4 parser restricts clusters strictly to code 'P' open-market purchases excluding <aff10b5One> and footnote plan disclosures.",
  },
  {
    date: "2026-11-21",
    scope: "NSE India Direct API (v1.1 / v2.0)",
    finding: "Direct NSE India JSON endpoints return HTTP 403 Akamai WAF challenge to headless batch runners.",
    remediation: "Marked direct FII/DII JSON explicitly unavailable in status metadata while sourcing daily Nifty 50 sectors and India VIX reliably via equity/volatility feeds.",
  },
  {
    date: "2026-11-21",
    scope: "ALFRED US Macro Point-in-Time (v2.0)",
    finding: "ALFRED CSV ignores comma-separated vintage lists and returns only the first requested date.",
    remediation: "Batch ingestion queries single vintage_date snapshots concurrently and caches historical vintages permanently so 'As Known Then' never leaks subsequent BEA/BLS revisions.",
  },
];

export default function CorrectionsPage() {
  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-12">
      <div>
        <div className="text-[10px] font-mono uppercase tracking-widest text-[#6f7c90]">
          Public Audit Trail & Provenance
        </div>
        <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-[#e8edf5] mt-0.5">
          Corrections, Retractions &amp; Data-Integrity Audit Log
        </h1>
        <p className="text-xs sm:text-sm text-[#8c9bb0] mt-1">
          Public ledger of source discontinuations, unit-scaling fixes, WAF exclusions, and methodological constraints across MacroMonitor.
        </p>
      </div>

      <div className="rounded-[6px] border border-[#1d2633] bg-[#0c1017] p-4 flex items-start gap-2.5 text-xs text-[#8c9bb0]">
        <ShieldCheck className="w-4 h-4 text-[#2dd4bf] mt-0.5 shrink-0" />
        <div className="leading-relaxed">
          <strong className="text-[#e8edf5]">No Silent Restatements:</strong> Whenever an upstream feed changes format, halts publication, or exhibits revision bias, MacroMonitor documents the finding here and degrades quality grades transparently rather than extrapolating missing data.
        </div>
      </div>

      <div className="rounded-[6px] border border-[#1d2633] bg-[#0c1017] overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead className="bg-[#07090d] border-b border-[#1d2633] text-[#6f7c90] text-left">
            <tr>
              <th className="py-2.5 px-4 font-medium whitespace-nowrap">Date</th>
              <th className="py-2.5 px-4 font-medium whitespace-nowrap">Scope</th>
              <th className="py-2.5 px-4 font-medium">Audit Finding</th>
              <th className="py-2.5 px-4 font-medium">Applied Remediation</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#1d2633]">
            {CORRECTIONS_LOG.map((entry, idx) => (
              <tr key={idx} className="align-top hover:bg-[#121823]/40">
                <td className="py-2.5 px-4 font-mono tabular-nums text-[#6f7c90] whitespace-nowrap">{entry.date}</td>
                <td className="py-2.5 px-4 font-semibold text-[#e8edf5] whitespace-nowrap">{entry.scope}</td>
                <td className="py-2.5 px-4 text-[#8c9bb0] leading-relaxed">{entry.finding}</td>
                <td className="py-2.5 px-4 text-[#e8edf5] leading-relaxed">{entry.remediation}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
