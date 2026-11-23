// web/src/components/ui/Sparkline.tsx
"use client";

import React from "react";

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  className?: string;
}

export const Sparkline: React.FC<SparklineProps> = ({
  data,
  width = 120,
  height = 28,
  color = "var(--accent)",
  className = "",
}) => {
  if (!data || data.length < 2) {
    return <div style={{ width, height }} className="bg-[var(--bg-2)]/30 rounded" />;
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min === 0 ? 1 : max - min;
  const padding = 2;
  const effectiveHeight = height - padding * 2;

  const points = data
    .map((val, idx) => {
      const x = (idx / (data.length - 1)) * width;
      const y = height - padding - ((val - min) / range) * effectiveHeight;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const lastPoint = points.split(" ").slice(-1)[0];
  const [lastX, lastY] = lastPoint ? lastPoint.split(",") : [width, height / 2];

  // Determine direction
  const isUp = data[data.length - 1] >= data[0];
  const strokeColor = color === "var(--accent)" ? (isUp ? "var(--pos)" : "var(--neg)") : color;

  return (
    <svg
      width={width}
      height={height}
      className={`overflow-visible ${className}`}
      aria-hidden="true"
    >
      <polyline
        fill="none"
        stroke={strokeColor}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
      <circle
        cx={lastX}
        cy={lastY}
        r="2"
        fill={strokeColor}
      />
    </svg>
  );
};
