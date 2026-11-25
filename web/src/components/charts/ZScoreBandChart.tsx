// web/src/components/charts/ZScoreBandChart.tsx
// C21: Standardized z-score band chart with outlier thresholds (Section 5.1)
"use client";

import React, { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { loadCompactSeries } from "@/lib/data/client";

interface ZScoreBandChartProps {
  seriesName?: string;
  assetId?: string;
  dates?: string[];
  zScores?: number[];
}

export const ZScoreBandChart: React.FC<ZScoreBandChartProps> = ({
  seriesName,
  assetId = "spx",
  dates: propDates,
  zScores: propZScores,
}) => {
  const [loadedSeries, setLoadedSeries] = useState<{
    name: string;
    dates: string[];
    zScores: number[];
  } | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (propDates && propZScores && propDates.length > 0) return;
    let active = true;
    setLoading(true);
    setError(null);
    loadCompactSeries(assetId)
      .then((res) => {
        if (!active) return;
        if (res.status === "error") {
          setError(res.error || "Failed to load series");
          return;
        }
        const d = res.data.data.d;
        const v = res.data.data.v;
        const isRate = assetId === "us10y" || assetId === "us2y" || assetId === "us10y_real";

        // Compute daily stationary changes (log returns or yield diffs)
        const rets: { date: string; ret: number }[] = [];
        for (let i = 1; i < v.length; i++) {
          const prev = v[i - 1];
          const curr = v[i];
          if (prev !== null && curr !== null && (isRate || (prev > 0 && curr > 0))) {
            const r = isRate ? curr - prev : Math.log(curr / prev);
            rets.push({ date: d[i], ret: r });
          }
        }

        // Compute expanding z-score of daily innovations (anti-lookahead: mean & std up to t-1, min 30 obs)
        const outDates: string[] = [];
        const outZ: number[] = [];
        let sum = 0;
        let sumSq = 0;
        let count = 0;
        const startIdx = Math.max(30, rets.length - 60);

        for (let i = 0; i < rets.length; i++) {
          const r = rets[i].ret;
          if (i >= startIdx && count >= 30) {
            const mean = sum / count;
            const variance = Math.max(1e-12, sumSq / count - mean * mean);
            const std = Math.sqrt(variance);
            const z = (r - mean) / std;
            outDates.push(rets[i].date.slice(5));
            outZ.push(parseFloat(Math.max(-5, Math.min(5, z)).toFixed(2)));
          }
          sum += r;
          sumSq += r * r;
          count += 1;
        }

        setLoadedSeries({
          name: seriesName || `${res.data.data.name} Daily Innovation Z-Score`,
          dates: outDates,
          zScores: outZ,
        });
      })
      .catch((e) => {
        if (active) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [assetId, propDates, propZScores, seriesName]);

  const { dates, zScores, activeName } = useMemo(() => {
    if (propDates && propZScores && propDates.length > 0) {
      return {
        dates: propDates,
        zScores: propZScores,
        activeName: seriesName || "Daily Innovation Z-Score",
      };
    }
    if (loadedSeries) {
      return {
        dates: loadedSeries.dates,
        zScores: loadedSeries.zScores,
        activeName: loadedSeries.name,
      };
    }
    return { dates: [], zScores: [], activeName: seriesName || "Daily Innovation Z-Score" };
  }, [propDates, propZScores, loadedSeries, seriesName]);

  if (error) {
    return (
      <div className="h-[280px] flex items-center justify-center text-xs font-mono text-[var(--neg)]">
        Failed to load z-score series: {error}
      </div>
    );
  }

  if (loading || dates.length === 0) {
    return (
      <div className="h-[280px] flex items-center justify-center text-xs font-mono text-[var(--tx-2)] animate-pulse">
        Computing trailing 60-day innovation z-scores from series data...
      </div>
    );
  }

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0c1017",
      borderColor: "#1d2633",
      textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
      formatter: (params: any) => {
        const item = params[0];
        const val = item.value;
        const alertTag = Math.abs(val) >= 2.0 ? `<span style="color:#e8765a;font-weight:bold"> [Outlier |z| >= 2.0]</span>` : "";
        return `
          <div style="font-weight:600;margin-bottom:2px">${item.axisValue}</div>
          <div style="color:#6f7c90">Standardized Z: <span style="color:${val >= 0 ? '#3fb6a8' : '#e8765a'}">${val >= 0 ? '+' : ''}${val}&sigma;</span>${alertTag}</div>
        `;
      },
    },
    grid: { left: 45, right: 30, top: 30, bottom: 35 },
    xAxis: {
      type: "category",
      data: dates,
      axisLine: { lineStyle: { color: "#1d2633" } },
      axisLabel: { color: "#6f7c90", fontFamily: "monospace", fontSize: 10 },
    },
    yAxis: {
      type: "value",
      min: -4.0,
      max: 4.0,
      axisLine: { lineStyle: { color: "#1d2633" } },
      splitLine: { lineStyle: { color: "#121823" } },
      axisLabel: {
        color: "#6f7c90",
        fontFamily: "monospace",
        fontSize: 10,
        formatter: (val: number) => `${val > 0 ? "+" : ""}${val}&sigma;`,
      },
    },
    series: [
      {
        name: activeName,
        type: "line",
        data: zScores,
        smooth: false,
        symbol: "circle",
        symbolSize: 4,
        lineStyle: { width: 1.8, color: "#6ea8ff" },
        itemStyle: {
          color: (params: any) => (Math.abs(params.value) >= 2.0 ? "#e8765a" : "#6ea8ff"),
        },
        markLine: {
          symbol: "none",
          data: [
            { yAxis: 3.0, lineStyle: { color: "#e8765a", type: "dashed", width: 1 } },
            { yAxis: 2.0, lineStyle: { color: "#e6b94d", type: "dashed", width: 1 } },
            { yAxis: 0.0, lineStyle: { color: "#2f6fb4", type: "solid", width: 1 } },
            { yAxis: -2.0, lineStyle: { color: "#e6b94d", type: "dashed", width: 1 } },
            { yAxis: -3.0, lineStyle: { color: "#e8765a", type: "dashed", width: 1 } },
          ],
        },
        markArea: {
          itemStyle: { color: "rgba(232, 118, 90, 0.06)" },
          data: [
            [{ yAxis: 2.0 }, { yAxis: 4.0 }],
            [{ yAxis: -4.0 }, { yAxis: -2.0 }],
          ],
        },
      },
    ],
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs font-mono text-[var(--tx-2)]">
        <span>Standardized Innovation Z-Score (60D Lookback)</span>
        <span className="text-amber-400">&plusmn;2&sigma; / &plusmn;3&sigma; Threshold Bands</span>
      </div>
      <ReactECharts option={option} style={{ height: "300px", width: "100%" }} />
    </div>
  );
};
