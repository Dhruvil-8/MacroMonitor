// web/src/components/charts/NetLiquidityChart.tsx
"use client";

import React, { useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

interface NetLiquidityChartProps {
  dates: string[];
  netLiquidity: number[];
  fedAssets: number[];
  tga: number[];
  rrp: number[];
  spxOverlay: number[];
  rollingCorr60d: number[];
}

export const NetLiquidityChart: React.FC<NetLiquidityChartProps> = ({
  dates,
  netLiquidity,
  fedAssets,
  tga,
  rrp,
  spxOverlay,
  rollingCorr60d,
}) => {
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [showDecomp, setShowDecomp] = useState<boolean>(true);

  const option = useMemo(() => {
    const seriesList: any[] = [
      {
        name: "Net Liquidity",
        type: "line",
        yAxisIndex: 0,
        data: netLiquidity,
        lineStyle: { width: 2.5, color: "#38bdf8" },
        showSymbol: false,
        z: 10,
      },
    ];

    if (showDecomp) {
      seriesList.push(
        {
          name: "Fed Assets",
          type: "line",
          yAxisIndex: 0,
          data: fedAssets,
          lineStyle: { width: 1.2, color: "#22c55e", type: "dashed" },
          showSymbol: false,
          z: 5,
        },
        {
          name: "TGA Balance",
          type: "line",
          yAxisIndex: 0,
          data: tga,
          lineStyle: { width: 1.2, color: "#f59e0b", type: "dotted" },
          showSymbol: false,
          z: 5,
        },
        {
          name: "Reverse Repo (RRP)",
          type: "line",
          yAxisIndex: 0,
          data: rrp,
          lineStyle: { width: 1.2, color: "#ec4899", type: "dotted" },
          showSymbol: false,
          z: 5,
        }
      );
    }

    if (showOverlay) {
      seriesList.push({
        name: "S&P 500 Index",
        type: "line",
        yAxisIndex: 1,
        data: spxOverlay,
        lineStyle: { width: 1.8, color: "#a855f7" },
        showSymbol: false,
        z: 8,
      });
    }

    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0d131d",
        borderColor: "#232f42",
        textStyle: { color: "#e8edf5", fontSize: 12, fontFamily: "var(--font-mono)" },
        formatter: (params: any[]) => {
          if (!params || params.length === 0) return "";
          let html = `<div style="font-weight:600;margin-bottom:4px;border-bottom:1px solid #232f42;padding-bottom:3px;">${params[0].axisValue}</div>`;
          params.forEach((p) => {
            const isSpx = p.seriesName.includes("S&P 500");
            const unit = isSpx ? "pts" : "$B";
            const valStr = isSpx ? p.value.toLocaleString() : `$${p.value.toLocaleString()} B`;
            html += `<div style="display:flex;justify-content:space-between;gap:12px;margin-top:2px;">
              <span style="color:${p.color};">${p.seriesName}:</span>
              <span style="font-weight:600;">${valStr}</span>
            </div>`;
          });
          return html;
        },
      },
      legend: {
        bottom: 0,
        textStyle: { color: "#a9b4c4", fontSize: 11 },
        itemWidth: 14,
        itemHeight: 8,
      },
      grid: {
        top: 24,
        left: 55,
        right: showOverlay ? 55 : 20,
        bottom: 45,
      },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: { lineStyle: { color: "#232f42" } },
        axisLabel: {
          color: "#6f7c90",
          fontSize: 10,
          formatter: (val: string) => {
            const parts = val.split("-");
            return parts.length >= 2 ? `${parts[0].slice(2)}/${parts[1]}` : val;
          },
        },
        splitLine: { show: false },
      },
      yAxis: [
        {
          type: "value",
          name: "USD ($B)",
          nameTextStyle: { color: "#6f7c90", fontSize: 10, align: "right" },
          axisLabel: {
            color: "#6f7c90",
            fontSize: 10,
            formatter: (v: number) => `$${v.toFixed(0)}B`,
          },
          splitLine: {
            show: true,
            lineStyle: { color: "#232f42", opacity: 0.5, type: "dashed" },
          },
        },
        {
          type: "value",
          name: "S&P 500",
          nameTextStyle: { color: "#6f7c90", fontSize: 10, align: "left" },
          show: showOverlay,
          position: "right",
          axisLabel: {
            color: "#6f7c90",
            fontSize: 10,
            formatter: (v: number) => v.toFixed(0),
          },
          splitLine: { show: false },
        },
      ],
      series: seriesList,
    };
  }, [dates, netLiquidity, fedAssets, tga, rrp, spxOverlay, showOverlay, showDecomp]);

  const corrOption = useMemo(() => {
    return {
      backgroundColor: "transparent",
      animationDuration: 300,
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0d131d",
        borderColor: "#232f42",
        textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "var(--font-mono)" },
        formatter: (params: any[]) => {
          if (!params || params.length === 0) return "";
          const p = params[0];
          return `<div>${p.axisValue}: <span style="font-weight:600;color:#38bdf8;">ρ = ${p.value}</span></div>`;
        },
      },
      grid: {
        top: 10,
        left: 55,
        right: showOverlay ? 55 : 20,
        bottom: 25,
      },
      xAxis: {
        type: "category",
        data: dates,
        axisLine: { lineStyle: { color: "#232f42" } },
        axisLabel: { show: false },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        min: -1,
        max: 1,
        interval: 0.5,
        axisLabel: {
          color: "#6f7c90",
          fontSize: 9,
          formatter: (v: number) => v.toFixed(1),
        },
        splitLine: {
          show: true,
          lineStyle: { color: "#232f42", opacity: 0.4 },
        },
      },
      series: [
        {
          name: "60D Correlation",
          type: "line",
          data: rollingCorr60d,
          lineStyle: { width: 1.5, color: "#38bdf8" },
          areaStyle: {
            color: "#38bdf8",
            opacity: 0.08,
          },
          showSymbol: false,
        },
      ],
    };
  }, [dates, rollingCorr60d, showOverlay]);

  return (
    <div className="space-y-4">
      {/* Control bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
              showOverlay ? "bg-[#232f42] text-[#e8edf5] border border-[#38bdf8]" : "bg-[#131b26] text-[#6f7c90]"
            }`}
          >
            {showOverlay ? "Hide S&P 500 Overlay" : "Show S&P 500 Overlay"}
          </button>
          <button
            onClick={() => setShowDecomp(!showDecomp)}
            className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
              showDecomp ? "bg-[#232f42] text-[#e8edf5] border border-[#22c55e]" : "bg-[#131b26] text-[#6f7c90]"
            }`}
          >
            {showDecomp ? "Hide Components (Fed, TGA, RRP)" : "Show Components"}
          </button>
        </div>
        <div className="text-[11px] font-mono text-[#6f7c90]">
          Formula: <span className="text-[#38bdf8]">Net Liq = Fed - TGA - RRP</span>
        </div>
      </div>

      {/* Main Liquidity Chart */}
      <div className="h-[360px] w-full">
        <ReactECharts option={option} style={{ height: "100%", width: "100%" }} notMerge={true} />
      </div>

      {/* Rolling Correlation Strip */}
      <div className="pt-2 border-t border-[#232f42]">
        <div className="flex items-center justify-between text-[11px] font-mono text-[#6f7c90] mb-1">
          <span>Rolling 60-Day Correlation: Net Liquidity vs S&P 500</span>
          <span className="text-[10px] text-[#a9b4c4]">Association, not mechanical causation</span>
        </div>
        <div className="h-[90px] w-full">
          <ReactECharts option={corrOption} style={{ height: "100%", width: "100%" }} notMerge={true} />
        </div>
      </div>
    </div>
  );
};
