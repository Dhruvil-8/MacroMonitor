// web/src/components/charts/StressGauge.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";

interface StressGaugeProps {
  score: number;
  regime: string;
  components: Record<string, number>;
  missingComponents: string[];
}

export const StressGauge: React.FC<StressGaugeProps> = ({
  score,
  regime,
  components,
  missingComponents,
}) => {
  const getRegimeColor = (val: number) => {
    if (val < 40) return "#22c55e"; // green
    if (val < 70) return "#38bdf8"; // blue
    if (val < 85) return "#f59e0b"; // amber
    return "#ef4444"; // red
  };

  const option = useMemo(() => {
    return {
      backgroundColor: "transparent",
      series: [
        {
          type: "gauge",
          startAngle: 180,
          endAngle: 0,
          min: 0,
          max: 100,
          splitNumber: 5,
          radius: "100%",
          center: ["50%", "70%"],
          axisLine: {
            lineStyle: {
              width: 14,
              color: [
                [0.4, "#22c55e"],
                [0.7, "#38bdf8"],
                [0.85, "#f59e0b"],
                [1.0, "#ef4444"],
              ],
            },
          },
          pointer: {
            itemStyle: {
              color: "#e8edf5",
            },
            length: "60%",
            width: 4,
          },
          axisTick: { show: false },
          splitLine: {
            length: 14,
            lineStyle: { color: "#0d131d", width: 2 },
          },
          axisLabel: {
            color: "#6f7c90",
            fontSize: 10,
            distance: -28,
            formatter: (v: number) => (v === 0 || v === 50 || v === 100 ? `${v}` : ""),
          },
          detail: {
            valueAnimation: true,
            formatter: () => `${score.toFixed(1)}`,
            color: "#e8edf5",
            fontSize: 24,
            fontWeight: 700,
            offsetCenter: [0, "-15%"],
          },
          data: [{ value: score }],
        },
      ],
    };
  }, [score]);

  return (
    <div className="space-y-4">
      {/* Gauge and Status */}
      <div className="flex flex-col items-center">
        <div className="h-[140px] w-[240px]">
          <ReactECharts option={option} style={{ height: "100%", width: "100%" }} notMerge={true} />
        </div>
        <div
          className="px-2.5 py-0.5 rounded text-xs font-mono font-semibold uppercase tracking-wider"
          style={{
            backgroundColor: `${getRegimeColor(score)}15`,
            color: getRegimeColor(score),
            border: `1px solid ${getRegimeColor(score)}40`,
          }}
        >
          {regime}
        </div>
      </div>

      {/* Component Breakdown Bars */}
      <div className="space-y-2 pt-2 border-t border-[var(--line)]">
        <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--tx-2)]">
          Component Percentiles (Zero Imputation)
        </div>
        <div className="space-y-1.5 font-mono text-xs">
          {Object.entries(components).map(([name, val]) => (
            <div key={name} className="space-y-0.5">
              <div className="flex justify-between items-center text-[11px]">
                <span className="capitalize text-[var(--tx-1)]">{name}</span>
                <span className="font-semibold text-[var(--tx-0)]">{val.toFixed(1)}%</span>
              </div>
              <div className="h-1.5 w-full bg-[var(--bg-2)] rounded overflow-hidden">
                <div
                  className="h-full rounded transition-all duration-500"
                  style={{
                    width: `${val}%`,
                    backgroundColor: getRegimeColor(val),
                  }}
                />
              </div>
            </div>
          ))}
        </div>

        {missingComponents.length > 0 && (
          <div className="pt-2 text-[10px] font-mono text-[var(--tx-2)]">
            Missing / Dropped:{" "}
            {missingComponents.map((c) => (
              <span key={c} className="underline uppercase mr-1">
                {c}
              </span>
            ))}
            (no synthetic imputation)
          </div>
        )}
      </div>
    </div>
  );
};
