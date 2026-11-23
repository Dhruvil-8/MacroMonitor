// web/src/components/insight/ObservationCard.tsx
"use client";

import React from "react";
import { ObservationItem } from "@/lib/data/schemas";
import { ShieldCheck, ArrowRight } from "lucide-react";
import { AuditData } from "../ui/AuditDrawer";

interface ObservationCardProps {
  observation: ObservationItem;
  onOpenAudit: (data: AuditData) => void;
}

export const ObservationCard: React.FC<ObservationCardProps> = ({
  observation,
  onOpenAudit,
}) => {
  const isHigh = observation.severity === "high";
  const isMed = observation.severity === "medium";

  const severityBadgeClass = isHigh
    ? "bg-[var(--neg)]/10 text-[var(--neg)] border-[var(--neg)]/30"
    : isMed
    ? "bg-[var(--warn)]/10 text-[var(--warn)] border-[var(--warn)]/30"
    : "bg-[var(--accent)]/10 text-[var(--accent)] border-[var(--accent)]/30";

  return (
    <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] hover:border-[var(--line-strong)] transition-all space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider border ${severityBadgeClass}`}
          >
            {observation.severity} severity
          </span>
          <span className="text-[11px] font-mono text-[var(--tx-2)]">
            Rule: {observation.rule_id}
          </span>
        </div>

        <button
          onClick={() =>
            onOpenAudit({
              metricName: `Observation Audit: ${observation.headline}`,
              formula: "Deterministic rules evaluation against historical distributions",
              formulaPlain: observation.body,
              parameters: observation.metrics,
              inputs: observation.inputs.map((inp) => ({
                id: inp,
                name: inp.toUpperCase(),
                source: "Radar Engine",
                unit: "metric",
              })),
              window: observation.window,
              currentValue: observation.metrics.vol20 ?? observation.metrics.spread_z ?? "Active",
              percentile: observation.metrics.percentile,
              n: observation.metrics.n,
              nEff: observation.metrics.n_eff,
              codeVersion: observation.code_version,
              specUrl: "/methodology",
            })
          }
          className="p-1 rounded hover:bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--accent)] transition-colors"
          title="Audit observation provenance"
          aria-label={`Audit ${observation.headline}`}
        >
          <ShieldCheck className="w-4 h-4" />
        </button>
      </div>

      {/* Headline & Body */}
      <div>
        <h4 className="text-sm font-semibold text-[var(--tx-0)] tracking-tight">
          {observation.headline}
        </h4>
        <p className="text-xs text-[var(--tx-1)] mt-1 leading-relaxed">
          {observation.body}
        </p>
      </div>

      {/* Metadata strip */}
      <div className="pt-2 border-t border-[var(--line)] flex items-center justify-between text-[11px] font-mono text-[var(--tx-2)]">
        <span>Asset: {observation.asset.toUpperCase()}</span>
        <span>As of: {observation.as_of}</span>
      </div>
    </div>
  );
};
