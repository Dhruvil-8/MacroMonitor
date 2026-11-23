// web/src/components/ui/MethodDrawer.tsx
"use client";

import React from "react";
import { X, BookOpen, ExternalLink, Sliders, Database, AlertCircle, Code2 } from "lucide-react";

export interface MethodData {
  chartTitle: string;
  whatItShows: string;
  definitionFormula: string;
  symbolsDefined?: Array<{ symbol: string; meaning: string }>;
  parameters: Record<string, string | number>;
  dataSources: Array<{ series: string; source: string; frequency: string }>;
  sample: {
    n: number | string;
    nEff?: number | string;
    dateRange: string;
  };
  interpretationNotes: string[];
  reproduce: {
    functionName: string;
    filePath: string;
    testPath: string;
  };
}

interface MethodDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  data: MethodData | null;
}

export const MethodDrawer: React.FC<MethodDrawerProps> = ({ isOpen, onClose, data }) => {
  if (!isOpen || !data) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm transition-opacity">
      <div
        className="w-full max-w-lg h-full bg-[var(--bg-1)] border-l border-[var(--line)] shadow-2xl flex flex-col overflow-hidden animate-in slide-in-from-right duration-200"
        role="dialog"
        aria-modal="true"
        aria-labelledby="method-drawer-title"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--line)] bg-[var(--bg-0)]">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-[var(--accent)]" />
            <span className="text-xs uppercase font-mono tracking-wider text-[var(--tx-2)]">Methodology & Specification</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--tx-0)] transition-colors"
            aria-label="Close method drawer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-sm">
          {/* 1. Title & What it shows */}
          <div>
            <h2 id="method-drawer-title" className="text-lg font-serif font-semibold text-[var(--tx-0)] tracking-tight">
              {data.chartTitle}
            </h2>
            <p className="text-xs text-[var(--tx-1)] mt-2 leading-relaxed bg-[var(--bg-2)]/60 p-3 rounded border border-[var(--line)]">
              {data.whatItShows}
            </p>
          </div>

          {/* 2. Mathematical Definition */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-[var(--tx-1)] uppercase tracking-wider">
              <span>Mathematical Formula</span>
            </div>
            <div className="p-3.5 rounded bg-[var(--bg-0)] border border-[var(--line)] font-mono text-xs text-[var(--accent)] overflow-x-auto">
              {data.definitionFormula}
            </div>
            {data.symbolsDefined && data.symbolsDefined.length > 0 && (
              <div className="space-y-1 pt-1">
                {data.symbolsDefined.map((sym, i) => (
                  <div key={i} className="text-xs text-[var(--tx-2)] flex items-start gap-2">
                    <span className="font-mono text-[var(--tx-1)] shrink-0">{sym.symbol}:</span>
                    <span>{sym.meaning}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 3. Parameters */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-[var(--tx-1)] uppercase tracking-wider">
              <Sliders className="w-3.5 h-3.5 text-[var(--tx-2)]" />
              <span>Statistical Parameters</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              {Object.entries(data.parameters).map(([k, v]) => (
                <div key={k} className="p-2.5 rounded bg-[var(--bg-0)] border border-[var(--line)] flex justify-between">
                  <span className="text-[var(--tx-2)]">{k}</span>
                  <span className="text-[var(--tx-0)] font-semibold">{String(v)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* 4. Sample Size & Effective N */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-[var(--tx-1)] uppercase tracking-wider">
              <Database className="w-3.5 h-3.5 text-[var(--tx-2)]" />
              <span>Sample & Autocorrelation</span>
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)]">
                <div className="text-[11px] text-[var(--tx-2)] uppercase">Observations (N)</div>
                <div className="text-base font-mono font-medium text-[var(--tx-0)] mt-0.5">{data.sample.n}</div>
              </div>
              <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)]">
                <div className="text-[11px] text-[var(--tx-2)] uppercase">Effective N (Neff)</div>
                <div className="text-base font-mono font-medium text-[var(--accent)] mt-0.5">
                  {data.sample.nEff ?? "—"}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-[var(--tx-2)] font-mono">Date Range: {data.sample.dateRange}</div>
          </div>

          {/* 5. Interpretation Notes */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-[var(--tx-1)] uppercase tracking-wider">
              <AlertCircle className="w-3.5 h-3.5 text-[var(--tx-2)]" />
              <span>Interpretation & Limitations</span>
            </div>
            <ul className="space-y-1.5 text-xs text-[var(--tx-2)] list-disc pl-4 leading-relaxed">
              {data.interpretationNotes.map((note, idx) => (
                <li key={idx}>{note}</li>
              ))}
            </ul>
          </div>

          {/* 6. Reproduce in Code */}
          <div className="space-y-2 pt-2 border-t border-[var(--line)]">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-[var(--tx-1)] uppercase tracking-wider">
              <Code2 className="w-3.5 h-3.5 text-[var(--tx-2)]" />
              <span>Reproduce & Audit Code</span>
            </div>
            <div className="p-3 rounded bg-[var(--bg-0)] border border-[var(--line)] space-y-1 text-xs font-mono">
              <div className="text-[var(--tx-0)] font-semibold">{data.reproduce.functionName}</div>
              <div className="text-[var(--tx-2)] text-[11px]">{data.reproduce.filePath}</div>
              <div className="text-[var(--tx-2)] text-[11px]">Test: {data.reproduce.testPath}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
