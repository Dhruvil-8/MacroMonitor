// web/src/components/charts/DistributionHistogram.tsx
"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";

interface DistributionHistogramProps {
  bins: number[];
  counts: number[];
  kdeGrid?: number[];
  kdeDensity?: number[];
  normalDensity?: number[];
  currentValue: number;
  percentile: number;
  unit: string;
}

export const DistributionHistogram: React.FC<DistributionHistogramProps> = ({
  bins,
  counts,
  kdeGrid,
  kdeDensity,
  normalDensity,
  currentValue,
  percentile,
  unit,
}) => {
  const option = useMemo(() => {
    if (!bins || bins.length === 0 || !counts || counts.length === 0) {
      return {};
    }

    const formattedBins = bins.map((b) => (b > 0 ? `+${(b * 100).toFixed(1)}%` : `${(b * 100).toFixed(1)}%`));
    const hasKde = Boolean(kdeGrid && kdeGrid.length > 0 && kdeDensity && kdeDensity.length > 0);

    // Normalizing KDE & Normal densities to frequency scale
    // count_equiv = density * totalCount * binWidth
    const totalCount = counts.reduce((a, b) => a + b, 0);
    const binWidth = bins.length > 1 ? Math.abs(bins[1] - bins[0]) : 0.01;
    const scaleFactor = totalCount * binWidth;

    // Map each bin center to the closest grid value in kdeGrid so the curve spans all bins
    const scaledKde = bins.map((b) => {
      if (!hasKde || !kdeGrid || !kdeDensity) return 0;
      let closestIdx = 0;
      let minDiff = Infinity;
      for (let k = 0; k < kdeGrid.length; k++) {
        const diff = Math.abs(kdeGrid[k] - b);
        if (diff < minDiff) {
          minDiff = diff;
          closestIdx = k;
        }
      }
      return Number(((kdeDensity[closestIdx] || 0) * scaleFactor).toFixed(1));
    });

    const scaledNorm = bins.map((b) => {
      if (!normalDensity || normalDensity.length === 0) return 0;
      const refGrid = (kdeGrid && kdeGrid.length === normalDensity.length) ? kdeGrid : null;
      let closestIdx = 0;
      let minDiff = Infinity;
      if (refGrid) {
        for (let k = 0; k < refGrid.length; k++) {
          const diff = Math.abs(refGrid[k] - b);
          if (diff < minDiff) {
            minDiff = diff;
            closestIdx = k;
          }
        }
      } else {
        // Linear index mapping
        closestIdx = Math.min(normalDensity.length - 1, Math.floor((bins.indexOf(b) / bins.length) * normalDensity.length));
      }
      return Number(((normalDensity[closestIdx] || 0) * scaleFactor).toFixed(1));
    });

    // Place current realization marker accurately based on percentile rank
    const clampedPct = Math.max(0, Math.min(100, percentile));
    const currentBinIdx = Math.max(
      0,
      Math.min(bins.length - 1, Math.round((clampedPct / 100) * (bins.length - 1)))
    );

    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: {
        top: 35,
        right: 25,
        bottom: 30,
        left: 35,
        containLabel: true,
      },
      legend: {
        data: ["Empirical Histogram", "Gaussian KDE (Silverman)", "Normal Equivalent N(μ, σ²)"],
        textStyle: { color: "#8b949e", fontSize: 11, fontFamily: "monospace" },
        top: 0,
        right: 10,
        icon: "roundRect",
      },
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(10, 14, 20, 0.95)",
        borderColor: "rgba(255, 255, 255, 0.12)",
        textStyle: { color: "#e6edf3", fontSize: 11, fontFamily: "var(--font-mono)" },
        formatter: (params: any[]) => {
          if (!params || params.length === 0) return "";
          const item = params[0];
          const histVal = params.find((p) => p.seriesName === "Empirical Histogram")?.value ?? 0;
          const kdeVal = params.find((p) => p.seriesName.includes("KDE"))?.value;
          const normVal = params.find((p) => p.seriesName.includes("Normal"))?.value;

          let html = `<div style="font-weight: 600; color: #58a6ff; margin-bottom: 4px;">Return Bin: ${item.name}</div>`;
          html += `<div style="color: #8b949e;">Observed: <b style="color: #e6edf3;">${histVal} sessions</b></div>`;
          if (kdeVal !== undefined) {
            html += `<div><span style="color: #3fb6a8;">KDE Density (equiv):</span> <b>${kdeVal}</b></div>`;
          }
          if (normVal !== undefined) {
            html += `<div><span style="color: #a371f7;">Normal Bell:</span> <b>${normVal}</b></div>`;
          }
          return html;
        },
      },
      xAxis: {
        type: "category",
        data: formattedBins,
        axisLine: { lineStyle: { color: "rgba(255, 255, 255, 0.15)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "var(--font-mono)", interval: 3 },
      },
      yAxis: {
        type: "value",
        name: "Sessions",
        nameTextStyle: { color: "#8b949e", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(255, 255, 255, 0.05)" } },
        axisLabel: { color: "#8b949e", fontSize: 10, fontFamily: "var(--font-mono)" },
      },
      series: [
        {
          name: "Empirical Histogram",
          type: "bar",
          data: counts,
          itemStyle: {
            color: "rgba(88, 166, 255, 0.35)",
            borderColor: "#58a6ff",
            borderWidth: 1,
            borderRadius: [2, 2, 0, 0],
          },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: {
              color: "#f0883e",
              type: "solid",
              width: 2,
            },
            label: {
              show: true,
              formatter: `Today (${percentile.toFixed(0)}th pct)`,
              color: "#f0883e",
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              position: "insideEndTop",
            },
            data: [{ xAxis: currentBinIdx }],
          },
        },
        ...(hasKde
          ? [
              {
                name: "Gaussian KDE (Silverman)",
                type: "line",
                smooth: true,
                data: scaledKde,
                lineStyle: { color: "#3fb6a8", width: 2 },
                itemStyle: { color: "#3fb6a8" },
                symbol: "none",
              },
            ]
          : []),
        ...(scaledNorm.length > 0
          ? [
              {
                name: "Normal Equivalent N(μ, σ²)",
                type: "line",
                smooth: true,
                data: scaledNorm,
                lineStyle: { color: "#a371f7", width: 1.5, type: "dashed" },
                itemStyle: { color: "#a371f7" },
                symbol: "none",
              },
            ]
          : []),
      ],
    };
  }, [bins, counts, kdeGrid, kdeDensity, normalDensity, percentile]);

  return (
    <div className="space-y-2">
      <div className="w-full h-[280px]">
        <ReactECharts
          option={option}
          style={{ height: "100%", width: "100%" }}
          opts={{ renderer: "canvas" }}
          notMerge={true}
          lazyUpdate={true}
        />
      </div>
      <div className="flex justify-between items-center text-[11px] font-mono text-[var(--tx-2)] px-1 pt-1 border-t border-[var(--line)]">
        <span>Empirical Daily Return Distribution ({bins.length} bins)</span>
        <span>Current Percentile: <strong className="text-[var(--accent)]">{percentile.toFixed(0)}th</strong></span>
      </div>
    </div>
  );
};
