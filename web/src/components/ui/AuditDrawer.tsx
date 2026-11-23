// web/src/components/ui/AuditDrawer.tsx
"use client";

import React from "react";
import { X, ExternalLink, ShieldCheck, Database, Sliders } from "lucide-react";

export interface AuditData {
  metricName: string;
  formula: string;
  formulaPlain: string;
  parameters: Record<string, any>;
  inputs: Array<{ id: string; name: string; source: string; unit: string }>;
  window: string;
  currentValue: string | number;
  percentile?: number;
  n?: number;
  nEff?: number;
  codeVersion?: string;
  specUrl?: string;
}

interface AuditDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  data: AuditData | null;
}

export const AuditDrawer: React.FC<AuditDrawerProps> = ({ isOpen, onClose, data }) => {
  if (!isOpen || !data) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm transition-opacity">
      <div
        className="w-full max-w-md h-full bg-[var(--bg-1)] border-l border-[var(--line)] shadow-2xl flex flex-col overflow-hidden animate-in slide-in-from-right duration-200"
        role="dialog"
        aria-modal="true"
        aria-labelledby="audit-title"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--line)] bg-[var(--bg-0)]">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-[var(--accent)]" />
            <span className="text-xs uppercase font-mono tracking-wider text-[var(--tx-2)]">Traceability Audit</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--tx-0)] transition-colors"
            aria-label="Close audit panel"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-sm">
          <div>
            <h2 id="audit-title" className="text-lg font-semibold text-[var(--tx-0)] tracking-tight">
              {data.metricName}
            </h2>
            <p className="text-xs text-[var(--tx-2)] mt-1">
              Window: <span className="font-mono text-[var(--tx-1)]">{data.window}</span>
            </p>
          </div>

          {/* Mathematical Specification */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-[var(--tx-1)] uppercase tracking-wider">
              <span>Mathematical Definition</span>
            </div>
            <div className="p-3.5 rounded bg-[var(--bg-2)] border border-[var(--line)] font-mono text-xs text-[var(--tx-0)] overflow-x-auto">
              {data.formula}
            </div>
            <p className="text-xs text-[var(--tx-2)] leading-relaxed">
              {data.formulaPlain}
            </p>
          </div>

          {/* Sample Size & Statistical Power */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-[var(--tx-1)] uppercase tracking-wider">
              <span>Sample Size & Autocorrelation</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)]">
                <div className="text-[11px] text-[var(--tx-2)] uppercase">Raw Observations (N)</div>
                <div className="text-base font-mono font-medium text-[var(--tx-0)] mt-0.5">
                  {data.n ? data.n.toLocaleString() : "—"}
                </div>
              </div>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)]">
                <div className="text-[11px] text-[var(--tx-2)] uppercase">Effective N (N_eff)</div>
                <div className="text-base font-mono font-medium text-[var(--accent)] mt-0.5">
                  {data.nEff ? data.nEff.toLocaleString() : "—"}
                </div>
              </div>
            </div>
            <p className="text-[11px] text-[var(--tx-2)]">
              N_eff adjusts for overlapping windows and lag-1 autocorrelation (rho_1).
            </p>
          </div>

          {/* Parameters */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-[var(--tx-1)] uppercase tracking-wider">
              <Sliders className="w-3.5 h-3.5 text-[var(--tx-2)]" />
              <span>Hyperparameters</span>
            </div>
            <div className="rounded border border-[var(--line)] divide-y divide-[var(--line)]">
              {Object.entries(data.parameters).map(([key, val]) => (
                <div key={key} className="flex justify-between items-center px-3 py-2 text-xs">
                  <span className="text-[var(--tx-2)] font-mono">{key}</span>
                  <span className="text-[var(--tx-0)] font-mono font-medium">{String(val)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Inputs & Sources */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-[var(--tx-1)] uppercase tracking-wider">
              <Database className="w-3.5 h-3.5 text-[var(--tx-2)]" />
              <span>Input Series</span>
            </div>
            <div className="space-y-2">
              {data.inputs.map((inp) => (
                <div key={inp.id} className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] flex justify-between items-center">
                  <div>
                    <div className="text-xs font-medium text-[var(--tx-0)]">{inp.name}</div>
                    <div className="text-[11px] font-mono text-[var(--tx-2)]">ID: {inp.id}</div>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] px-2 py-0.5 rounded bg-[var(--bg-2)] border border-[var(--line)] text-[var(--tx-1)] font-mono uppercase">
                      {inp.source}
                    </span>
                    <div className="text-[10px] text-[var(--tx-2)] font-mono mt-0.5">{inp.unit}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-[var(--line)] bg-[var(--bg-0)] flex items-center justify-between text-xs text-[var(--tx-2)]">
          <span className="font-mono text-[11px]">Version: {data.codeVersion || "v1.0.0"}</span>
          <a
            href={data.specUrl || "/methodology"}
            className="flex items-center gap-1 text-[var(--accent)] hover:underline font-medium"
          >
            <span>View Full Specification</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>
    </div>
  );
};
