// web/src/components/charts/ChartCard.tsx
"use client";

import React, { useEffect, useRef, useState } from "react";
import { ShieldCheck, HelpCircle, Table, BarChart2, Download, Camera } from "lucide-react";
import { InsightStrip } from "../insight/InsightStrip";
import { AuditData } from "../ui/AuditDrawer";
import { MethodData } from "../ui/MethodDrawer";
import { fetchManifest } from "@/lib/data/client";

function formatFetchTimestamp(iso?: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (isNaN(d.getTime())) return null;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

interface ChartCardProps {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  insightHeadline?: string;
  insightDetail?: string;
  sourceText?: string;
  asOf?: string;
  fetchedAt?: string;
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
  fetchedAt,
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
  const [manifestFetchedAt, setManifestFetchedAt] = useState<string | null>(null);
  const [hasCanvas, setHasCanvas] = useState(false);
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const hiddenTableRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (fetchedAt) return;
    let active = true;
    fetchManifest().then((res) => {
      if (!active) return;
      if (res.status === "ok" || res.status === "stale") {
        setManifestFetchedAt(res.data.fetched_at ?? null);
      }
    });
    return () => {
      active = false;
    };
  }, [fetchedAt]);

  useEffect(() => {
    const checkCanvas = () => {
      const canvas = chartContainerRef.current?.querySelector("canvas");
      setHasCanvas(Boolean(canvas));
    };
    checkCanvas();
    const timer = setTimeout(checkCanvas, 600);
    return () => clearTimeout(timer);
  }, [children, showTable]);

  const formattedFetch = formatFetchTimestamp(fetchedAt ?? manifestFetchedAt);

  const handleDownloadPng = () => {
    const srcCanvas = chartContainerRef.current?.querySelector("canvas");
    if (!srcCanvas) return;

    const padX = 24;
    const headerH = subtitle ? 58 : 42;
    const footerH = 30;
    const out = document.createElement("canvas");
    out.width = srcCanvas.width + padX * 2;
    out.height = srcCanvas.height + headerH + footerH;
    const ctx = out.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#0c1017";
    ctx.fillRect(0, 0, out.width, out.height);

    ctx.fillStyle = "#e8edf5";
    ctx.font = "600 16px system-ui, -apple-system, sans-serif";
    ctx.fillText(title, padX, 26);

    if (subtitle) {
      ctx.fillStyle = "#8c9bb0";
      ctx.font = "12px system-ui, -apple-system, sans-serif";
      ctx.fillText(subtitle, padX, 45);
    }

    ctx.drawImage(srcCanvas, padX, headerH);

    ctx.fillStyle = "#6f7c90";
    ctx.font = "11px monospace";
    const leftMeta = `MacroMonitor • dhruvil-8.github.io/MacroMonitor`;
    const rightMeta = asOf ? `As of: ${asOf}` : "";
    ctx.fillText(leftMeta, padX, out.height - 10);
    if (rightMeta) {
      const m = ctx.measureText(rightMeta);
      ctx.fillText(rightMeta, out.width - padX - m.width, out.height - 10);
    }

    const slug = title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
    const link = document.createElement("a");
    link.download = `${slug || "chart"}.png`;
    link.href = out.toDataURL("image/png");
    link.click();
  };

  const handleDownloadCsv = () => {
    const root = showTable ? chartContainerRef.current : hiddenTableRef.current;
    const tableEl = root?.querySelector("table");
    if (!tableEl) return;

    const rows = Array.from(tableEl.querySelectorAll("tr")).map((tr) =>
      Array.from(tr.querySelectorAll("th, td"))
        .map((cell) => {
          const text = (cell.textContent || "").replace(/\s+/g, " ").trim();
          return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
        })
        .join(",")
    );
    const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const slug = title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
    const link = document.createElement("a");
    link.href = url;
    link.download = `${slug || "data"}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div
      className={`p-6 rounded-2xl bg-[#0e121a]/90 backdrop-blur-md border border-white/[0.08] hover:border-white/[0.14] shadow-[0_8px_30px_rgba(0,0,0,0.35)] transition-all duration-200 flex flex-col justify-between ${className}`}
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3 pb-4 border-b border-white/[0.07]">
        <div>
          {eyebrow && (
            <span className="text-[10px] uppercase font-medium tracking-[0.12em] text-sky-400/90 block mb-1">
              {eyebrow}
            </span>
          )}
          <h3 className="text-[17px] font-semibold text-[var(--tx-0)] tracking-tight leading-snug">
            {title}
          </h3>
          {subtitle && (
            <p className="text-xs text-[var(--tx-2)] mt-1 leading-relaxed">{subtitle}</p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {headerControls}

          {hasCanvas && !showTable && (
            <button
              onClick={handleDownloadPng}
              className="p-1.5 rounded hover:bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--tx-0)] transition-colors border border-[var(--line)]"
              title="Download chart as PNG"
              aria-label={`Download PNG for ${title}`}
            >
              <Camera className="w-4 h-4" />
            </button>
          )}

          {dataTable && (
            <>
              <button
                onClick={handleDownloadCsv}
                className="p-1.5 rounded hover:bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--tx-0)] transition-colors border border-[var(--line)]"
                title="Download underlying data as CSV"
                aria-label={`Download CSV for ${title}`}
              >
                <Download className="w-4 h-4" />
              </button>
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
            </>
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
      <div ref={chartContainerRef} className="my-4 flex-1 min-h-[340px] flex flex-col justify-center">
        {showTable && dataTable ? (
          <div className="overflow-x-auto max-h-[380px] p-2 rounded bg-[var(--bg-0)] border border-[var(--line)] text-xs font-mono">
            {dataTable}
          </div>
        ) : (
          children
        )}
      </div>

      {dataTable && !showTable && (
        <div ref={hiddenTableRef} className="hidden" aria-hidden="true">
          {dataTable}
        </div>
      )}

      {/* Footer & Insight Strip */}
      <div className="space-y-3 pt-3 border-t border-[var(--line)]">
        {insightHeadline && (
          <InsightStrip headline={insightHeadline} detail={insightDetail} />
        )}

        <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-[var(--tx-2)]">
          <span>Sources: {sourceText}</span>
          {(asOf || formattedFetch) && (
            <span>
              {asOf ? `As of: ${asOf}` : ""}
              {asOf && formattedFetch ? " · " : ""}
              {formattedFetch ? `Last fetch: ${formattedFetch}` : ""}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
