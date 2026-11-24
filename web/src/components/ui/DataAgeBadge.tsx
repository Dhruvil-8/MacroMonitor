// web/src/components/ui/DataAgeBadge.tsx
"use client";

import React from "react";
import { AlertCircle, Calendar } from "lucide-react";

interface DataAgeBadgeProps {
  asOf: string;
  isStale?: boolean;
  ageHours?: number;
}

export const DataAgeBadge: React.FC<DataAgeBadgeProps> = ({ asOf, isStale, ageHours }) => {
  if (!asOf) return null;
  return (
    <div className="flex items-center gap-2">
      <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[var(--bg-1)] border border-[var(--line)] text-xs text-[var(--tx-2)]">
        <Calendar className="w-3.5 h-3.5 text-[var(--tx-2)]" />
        <span className="font-mono text-[11px]">As of {asOf}</span>
      </div>
      {isStale && (
        <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-[var(--warn)]/10 border border-[var(--warn)]/30 text-[11px] text-[var(--warn)] font-medium">
          <AlertCircle className="w-3 h-3" />
          <span>Stale ({ageHours}h)</span>
        </div>
      )}
    </div>
  );
};
