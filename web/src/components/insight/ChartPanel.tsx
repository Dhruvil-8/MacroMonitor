"use client";

import { ReactNode, useMemo, useState } from "react";
import { BarChart2, Download, FileCheck2, Info, Share2, Table2, X } from "lucide-react";
import { AuditPayload, useRadarStore } from "@/store/useRadarStore";

export interface ChartTableData {
  headers: string[];
  rows: (string | number | boolean | null | undefined)[][];
}

interface ChartPanelProps {
  title: string;
  subtitle: string;
  infoText: string;
  insights: string[];
  asOf: string;
  windowLabel: string;
  sourcesLabel: string;
  n: number | string;
  nEff: number | string;
  audit: AuditPayload;
  onDownloadCsv: () => void;
  getTableData?: () => ChartTableData;
  headerRightExtra?: ReactNode;
  embedKey?: string;
  children: ReactNode;
}

export function ChartPanel({
  title,
  subtitle,
  infoText,
  insights,
  asOf,
  windowLabel,
  sourcesLabel,
  n,
  nEff,
  audit,
  onDownloadCsv,
  getTableData,
  headerRightExtra,
  embedKey,
  children,
}: ChartPanelProps) {
  const [showInfo, setShowInfo] = useState(false);
  const [showTable, setShowTable] = useState(false);
  const openAudit = useRadarStore((s) => s.openAudit);
  const openShare = useRadarStore((s) => s.openShare);

  const tablePayload = useMemo(() => {
    if (!showTable || !getTableData) return null;
    const raw = getTableData();
    const limit = 120;
    const total = raw.rows.length;
    const displayRows = total > limit ? raw.rows.slice(-limit).reverse() : [...raw.rows].reverse();
    return { headers: raw.headers, rows: displayRows, total, truncated: total > limit };
  }, [showTable, getTableData]);

  return (
    <section className="rounded-[6px] border border-[#1d2633] bg-[#0c1017] p-4 sm:p-5 flex flex-col gap-3">
      {/* Header row */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2">
        <div className="min-w-0">
          <h2 className="text-base sm:text-lg font-semibold tracking-tight text-[#e8edf5]">
            {title}
          </h2>
          <p className="text-xs sm:text-sm text-[#8c9bb0] mt-0.5">{subtitle}</p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          {headerRightExtra}
          {getTableData && (
            <button
              type="button"
              onClick={() => setShowTable((v) => !v)}
              aria-pressed={showTable}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-[6px] border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                showTable
                  ? "border-[#38bdf8] bg-[#121823] text-[#e8edf5]"
                  : "border-[#1d2633] text-[#8c9bb0] hover:text-[#e8edf5] hover:bg-[#121823]"
              }`}
            >
              {showTable ? <BarChart2 className="w-3.5 h-3.5" /> : <Table2 className="w-3.5 h-3.5" />}
              <span>{showTable ? "Chart" : "Table"}</span>
            </button>
          )}
          <button
            type="button"
            onClick={() => setShowInfo((v) => !v)}
            aria-expanded={showInfo}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-[6px] border border-[#1d2633] text-[#8c9bb0] hover:text-[#e8edf5] hover:bg-[#121823] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            {showInfo ? <X className="w-3.5 h-3.5" /> : <Info className="w-3.5 h-3.5" />}
            <span>Info</span>
          </button>
          <button
            type="button"
            onClick={() => openAudit(audit)}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-[6px] border border-[#1d2633] text-[#8c9bb0] hover:text-[#e8edf5] hover:bg-[#121823] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <FileCheck2 className="w-3.5 h-3.5" />
            <span>Audit</span>
          </button>
          <button
            type="button"
            onClick={() =>
              openShare({
                chartKey: embedKey || audit.specAnchor || "state_map",
                title,
                subtitle,
                headlineMetric: insights[0] || title,
                percentileSummary: audit.percentileSummary,
                asOf,
                n,
                nEff,
              })
            }
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-[6px] border border-[#1d2633] text-[#8c9bb0] hover:text-[#e8edf5] hover:bg-[#121823] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Share2 className="w-3.5 h-3.5" />
            <span>Share</span>
          </button>
          <button
            type="button"
            onClick={onDownloadCsv}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-[6px] border border-[#1d2633] text-[#8c9bb0] hover:text-[#e8edf5] hover:bg-[#121823] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download CSV</span>
          </button>
        </div>
      </div>

      {/* Collapsible Info Note */}
      {showInfo && (
        <div className="rounded-[6px] border border-[#1d2633] bg-[#07090d] px-3 py-2.5 text-xs leading-relaxed text-[#8c9bb0]">
          {infoText}
        </div>
      )}

      {/* Main Chart Surface or Accessible Data Table */}
      {showTable && tablePayload ? (
        <div className="flex flex-col gap-2">
          <div className="overflow-x-auto max-h-[360px] overflow-y-auto rounded-[6px] border border-[#1d2633]">
            <table className="w-full text-xs border-collapse">
              <caption className="sr-only">Tabular statistical data for {title}</caption>
              <thead className="sticky top-0 bg-[#07090d] border-b border-[#1d2633] text-[#6f7c90] text-left">
                <tr>
                  {tablePayload.headers.map((h) => (
                    <th key={h} scope="col" className="py-2 px-3 font-medium whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1d2633] font-mono tabular-nums">
                {tablePayload.rows.map((row, rIdx) => (
                  <tr key={rIdx} className="hover:bg-[#121823]/50">
                    {row.map((cell, cIdx) => (
                      <td key={cIdx} className="py-1.5 px-3 text-[#e8edf5] whitespace-nowrap">
                        {cell === null || cell === undefined || cell === "" ? "—" : String(cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-[11px] font-mono text-[#6f7c90]">
            {tablePayload.truncated
              ? `Displaying latest 120 of ${tablePayload.total.toLocaleString()} rows (most recent first). Use Download CSV for complete history.`
              : `Displaying all ${tablePayload.total.toLocaleString()} rows (most recent first).`}
          </div>
        </div>
      ) : (
        <div className="w-full">{children}</div>
      )}

      {/* Deterministic Rules Insight Strip */}
      <div className="rounded-[6px] border border-[#1d2633] bg-[#07090d] px-3.5 py-2.5">
        <div className="flex flex-col gap-1">
          {insights.slice(0, 3).map((line, idx) => (
            <p key={idx} className="text-xs sm:text-[13px] leading-relaxed text-[#e8edf5]">
              {line}
            </p>
          ))}
        </div>
      </div>

      {/* Metadata Footer */}
      <footer className="flex flex-wrap items-center gap-x-3 gap-y-1 pt-1 border-t border-[#1d2633] text-[11px] text-[#6f7c90] font-mono tabular-nums">
        <span>As of {asOf}</span>
        <span aria-hidden="true">·</span>
        <span>Window: {windowLabel}</span>
        <span aria-hidden="true">·</span>
        <span>Sources: {sourcesLabel}</span>
        <span aria-hidden="true">·</span>
        <span>
          N = {typeof n === "number" ? n.toLocaleString() : n}, N_eff ={" "}
          {typeof nEff === "number" ? nEff.toLocaleString() : nEff}
        </span>
      </footer>
    </section>
  );
}
