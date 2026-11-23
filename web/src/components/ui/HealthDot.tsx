// web/src/components/ui/HealthDot.tsx
"use client";

import React from "react";

interface HealthDotProps {
  status: "ok" | "stale" | "failed" | string;
  className?: string;
  label?: string;
}

export const HealthDot: React.FC<HealthDotProps> = ({ status, className = "", label }) => {
  const colorClass =
    status === "ok"
      ? "bg-[var(--pos)]"
      : status === "stale"
      ? "bg-[var(--warn)]"
      : "bg-[var(--neg)]";

  const statusText =
    status === "ok"
      ? "Pipeline Live & Verified"
      : status === "stale"
      ? "Data Stale (>36h)"
      : "Health Check Failed";

  return (
    <div className={`inline-flex items-center gap-1.5 text-xs text-[var(--tx-2)] ${className}`} title={statusText}>
      <span className={`w-2 h-2 rounded-full ${colorClass} animate-pulse`} />
      {label && <span className="font-mono text-[11px]">{label}</span>}
    </div>
  );
};
