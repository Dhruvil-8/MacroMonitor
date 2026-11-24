// web/src/components/charts/ChartCard.tsx
"use client";

import React, { useState } from "react";
import { ShieldCheck, HelpCircle, Table, BarChart2 } from "lucide-react";
import { InsightStrip } from "../insight/InsightStrip";
import { AuditData } from "../ui/AuditDrawer";
import { MethodData } from "../ui/MethodDrawer";

interface ChartCardProps {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  insightHeadline?: string;
  insightDetail?: string;
  sourceText?: string;
  asOf?: string;
  auditData?: AuditData;
  onOpenAudit?: (data: AuditData) => void;
  methodData?: MethodData;
  onOpenMethod?: (data: MethodData) => void;
  dataTable?: React.ReactNode;
  children: React.ReactNode;
  headerControls?: React.ReactNode;
  className?: string;
}

export const ChartCard: React.FC<ChartCardProps> = ({
  eyebrow,
  title,
  subtitle,
  insightHeadline,
  insightDetail,
  sourceText = "Federal Reserve, Yahoo Finance, NSE",
  asOf,
  auditData,
  onOpenAudit,
  methodData,
  onOpenMethod,
  dataTable,
  children,
  headerControls,
  className = "",
}) => {
  const [showTable, setShowTable] = useState(false);

  return (
    <div
      className={`p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] shadow-sm flex flex-col justify-between ${className}`}
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3 pb-4 border-b border-[var(--line)]">
        <div>
          {eyebrow && (
            <span className="text-[10px] uppercase font-mono tracking-widest text-[var(--tx-2)] block mb-1">
              {eyebrow}
            </span>
          )}
          <h3 className="text-base font-semibold text-[var(--tx-0)] tracking-tight">
            {title}
          </h3>
          {subtitle && (
            <p className="text-xs text-[var(--tx-2)] mt-0.5">{subtitle}</p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {headerControls}

          {dataTable && (
            <button
              onClick={() => setShowTable(!showTable)}
              className={`p-1.5 rounded transition-colors border border-[var(--line)] ${
                showTable
                  ? "bg-[var(--accent)]/10 text-[var(--accent)] border-[var(--accent)]"
                  : "hover:bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              }`}
              title={showTable ? "Show visual chart" : "Show data table (accessible view)"}
              aria-label={showTable ? "Switch to chart" : "Switch to data table"}
            >
              {showTable ? <BarChart2 className="w-4 h-4" /> : <Table className="w-4 h-4" />}
            </button>
          )}

          {methodData && onOpenMethod && (
            <button
              onClick={() => onOpenMethod(methodData)}
              className="p-1.5 rounded hover:bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--accent)] transition-colors border border-[var(--line)]"
              title="Methodology, KaTeX formula & statistical specification"
              aria-label={`Methodology for ${title}`}
            >
              <HelpCircle className="w-4 h-4" />
            </button>
          )}

          {auditData && onOpenAudit && (
            <button
              onClick={() => onOpenAudit(auditData)}
              className="p-1.5 rounded hover:bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--accent)] transition-colors border border-[var(--line)]"
              title="Open traceability audit"
              aria-label={`Audit ${title}`}
            >
              <ShieldCheck className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Chart Viewport or Accessible Data Table */}
      <div className="my-4 flex-1 min-h-[340px] flex flex-col justify-center">
        {showTable && dataTable ? (
          <div className="overflow-x-auto max-h-[380px] p-2 rounded bg-[var(--bg-0)] border border-[var(--line)] text-xs font-mono">
            {dataTable}
          </div>
        ) : (
          children
        )}
      </div>

      {/* Footer & Insight Strip */}
      <div className="space-y-3 pt-3 border-t border-[var(--line)]">
        {insightHeadline && (
          <InsightStrip headline={insightHeadline} detail={insightDetail} />
        )}

        <div className="flex items-center justify-between text-[11px] font-mono text-[var(--tx-2)]">
          <span>Sources: {sourceText}</span>
          {asOf && <span>As of: {asOf}</span>}
        </div>
      </div>
    </div>
  );
};
