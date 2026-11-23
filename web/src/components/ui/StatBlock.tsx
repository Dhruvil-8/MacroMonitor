// web/src/components/ui/StatBlock.tsx
"use client";

import React from "react";
import { Sparkline } from "./Sparkline";
import { ShieldAlert, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { AuditData } from "./AuditDrawer";

interface StatBlockProps {
  id: string;
  name: string;
  value: number | string;
  change1d?: number | null;
  unit: string;
  percentile?: number | null;
  sparkline?: number[];
  onOpenAudit?: (audit: AuditData) => void;
  auditPayload?: AuditData;
}

export const StatBlock: React.FC<StatBlockProps> = ({
  id,
  name,
  value,
  change1d,
  unit,
  percentile,
  sparkline,
  onOpenAudit,
  auditPayload,
}) => {
  const isPositive = (change1d ?? 0) > 0;
  const isNegative = (change1d ?? 0) < 0;

  const formattedValue =
    typeof value === "string"
      ? value
      : unit === "bps"
      ? `${value.toFixed(1)} bps`
      : unit === "pct"
      ? `${value.toFixed(2)}%`
      : value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  return (
    <div className="group relative p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] hover:border-[var(--line-strong)] transition-all">
      <div className="flex items-start justify-between">
        <div>
          <span className="text-xs uppercase font-mono tracking-wider text-[var(--tx-2)] block">
            {name}
          </span>
          <div className="text-xl font-mono font-semibold text-[var(--tx-0)] mt-1 tracking-tight">
            {formattedValue}
          </div>
        </div>

        {/* Audit trigger */}
        {onOpenAudit && auditPayload && (
          <button
            onClick={() => onOpenAudit(auditPayload)}
            className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--accent)] transition-all"
            title="Open traceability audit"
            aria-label={`Audit ${name}`}
          >
            <ShieldAlert className="w-4 h-4" />
          </button>
        )}
      </div>

      <div className="mt-3 flex items-end justify-between">
        <div className="space-y-1">
          {change1d !== null && change1d !== undefined && (
            <div
              className={`inline-flex items-center gap-0.5 text-xs font-mono font-medium ${
                isPositive
                  ? "text-[var(--pos)]"
                  : isNegative
                  ? "text-[var(--neg)]"
                  : "text-[var(--neutral)]"
              }`}
            >
              {isPositive ? (
                <ArrowUpRight className="w-3.5 h-3.5" />
              ) : isNegative ? (
                <ArrowDownRight className="w-3.5 h-3.5" />
              ) : null}
              <span>{isPositive ? `+${change1d.toFixed(2)}%` : `${change1d.toFixed(2)}%`}</span>
            </div>
          )}

          {percentile !== null && percentile !== undefined && (
            <div className="text-[11px] font-mono text-[var(--tx-2)]">
              Pctile: <span className="text-[var(--tx-1)] font-medium">{percentile.toFixed(0)}th</span>
            </div>
          )}
        </div>

        {sparkline && sparkline.length > 0 && (
          <Sparkline data={sparkline} width={90} height={24} />
        )}
      </div>
    </div>
  );
};
