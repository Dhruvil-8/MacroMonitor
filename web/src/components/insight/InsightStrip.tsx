// web/src/components/insight/InsightStrip.tsx
"use client";

import React from "react";
import { Info } from "lucide-react";

interface InsightStripProps {
  headline: string;
  detail?: string;
  className?: string;
}

export const InsightStrip: React.FC<InsightStripProps> = ({
  headline,
  detail,
  className = "",
}) => {
  return (
    <div
      className={`px-3.5 py-2.5 rounded bg-[var(--bg-0)] border border-[var(--line)] flex items-start gap-2.5 text-xs ${className}`}
    >
      <Info className="w-4 h-4 text-[var(--accent)] shrink-0 mt-0.5" />
      <div className="leading-relaxed">
        <span className="font-medium text-[var(--tx-0)]">{headline}</span>
        {detail && <span className="text-[var(--tx-2)] ml-1.5">{detail}</span>}
      </div>
    </div>
  );
};
