// web/src/components/charts/MarketStateMap.tsx
"use client";

import React, { useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { StateMapItem } from "@/lib/data/schemas";

interface MarketStateMapProps {
  assets: StateMapItem[];
  onSelectAsset?: (asset: string) => void;
}

export const MarketStateMap: React.FC<MarketStateMapProps> = ({
  assets,
  onSelectAsset,
}) => {
  const [selectedGroup, setSelectedGroup] = useState<string>("all");

  const groups = useMemo(() => {
    const set = new Set(assets.map((a) => a.group));
    return ["all", ...Array.from(set)];
  }, [assets]);

  const filteredAssets = useMemo(() => {
    if (selectedGroup === "all") return assets;
    return assets.filter((a) => a.group === selectedGroup);
  }, [assets, selectedGroup]);

  // Color mapping based on quadrant
  const getQuadrantColor = (quadrant: string) => {
    switch (quadrant) {
      case "Orderly trend":
        return "#3fb6a8"; // --pos (teal)
      case "Violent trend":
        return "#e8765a"; // --neg (coral)
      case "Quiet range":
        return "#6ea8ff"; // --accent (blue)
      case "Choppy":
      default:
        return "#e6b94d"; // --warn (amber)
    }
  };

  const option = useMemo(() => {
    return {
      backgroundColor: "transparent",
      animationDuration: 400,
      grid: {
        top: 30,
        right: 40,
        bottom: 50,
        left: 55,
        containLabel: true,
      },
      tooltip: {
        trigger: "item",
        backgroundColor: "#0c1017",
        borderColor: "#1d2633",
        textStyle: { color: "#e8edf5", fontSize: 12 },
        formatter: (params: any) => {
          const item = params.data.item as StateMapItem;
          return `
            <div style="font-family: monospace; padding: 4px;">
              <div style="font-weight: 600; font-size: 13px; color: #fff; margin-bottom: 4px;">${item.name} (${item.asset.toUpperCase()})</div>
              <div style="color: #6f7c90; font-size: 11px; margin-bottom: 6px;">${item.group} • ${item.region}</div>
              <div style="display: flex; justify-content: space-between; gap: 16px; margin-bottom: 2px;">
                <span style="color: #a9b4c4;">Regime:</span>
                <span style="font-weight: 600; color: ${getQuadrantColor(item.quadrant)}">${item.quadrant}</span>
              </div>
              <div style="display: flex; justify-content: space-between; gap: 16px; margin-bottom: 2px;">
                <span style="color: #a9b4c4;">Trend Strength (t):</span>
                <span>${item.trend_t > 0 ? "+" : ""}${item.trend_t.toFixed(2)}</span>
              </div>
              <div style="display: flex; justify-content: space-between; gap: 16px; margin-bottom: 2px;">
                <span style="color: #a9b4c4;">Vol Percentile:</span>
                <span>${item.vol_percentile.toFixed(0)}th</span>
              </div>
              <div style="display: flex; justify-content: space-between; gap: 16px;">
                <span style="color: #a9b4c4;">20D Momentum:</span>
                <span>${item.momentum_20d > 0 ? "+" : ""}${(item.momentum_20d * 100).toFixed(1)}%</span>
              </div>
            </div>
          `;
        },
      },
      xAxis: {
        type: "value",
        name: "Trend Strength (60D signed t-stat)",
        nameLocation: "middle",
        nameGap: 30,
        nameTextStyle: { color: "#6f7c90", fontSize: 11 },
        scale: true,
        min: (value: { min: number; max: number }) => Math.floor(Math.min(value.min, -4) - 1),
        max: (value: { min: number; max: number }) => Math.ceil(Math.max(value.max, 4) + 1),
        splitLine: {
          lineStyle: { color: "#1d2633", opacity: 0.6 },
        },
        axisLine: {
          lineStyle: { color: "#2a3647" },
        },
        axisLabel: {
          color: "#6f7c90",
          fontFamily: "monospace",
          fontSize: 11,
        },
      },
      yAxis: {
        type: "value",
        name: "Volatility Percentile (20D Vol)",
        nameLocation: "middle",
        nameGap: 40,
        nameTextStyle: { color: "#6f7c90", fontSize: 11 },
        min: 0,
        max: 100,
        splitLine: {
          lineStyle: { color: "#1d2633", opacity: 0.6 },
        },
        axisLine: {
          lineStyle: { color: "#2a3647" },
        },
        axisLabel: {
          color: "#6f7c90",
          fontFamily: "monospace",
          fontSize: 11,
          formatter: "{value}%",
        },
      },
      series: [
        {
          type: "scatter",
          symbolSize: (_val: any, params: any) => {
            const mom = params?.data?.item?.momentum_20d != null ? Math.abs(params.data.item.momentum_20d) : 0.03;
            return Math.max(12, Math.min(28, 14 + mom * 80));
          },
          label: {
            show: true,
            formatter: (params: any) => params?.data?.item?.asset?.toUpperCase() || "",
            position: "right",
            color: "#8b98a5",
            fontSize: 9,
            fontFamily: "monospace",
            distance: 4,
          },
          data: filteredAssets.map((a) => ({
            value: [a.trend_t, a.vol_percentile],
            item: a,
            itemStyle: {
              color: getQuadrantColor(a.quadrant),
              opacity: 0.85,
              borderColor: "#121823",
              borderWidth: 1.5,
            },
          })),
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: {
              color: "#2a3647",
              type: "dashed",
              width: 1,
            },
            data: [
              { xAxis: 2.0 },
              { xAxis: -2.0 },
              { yAxis: 50.0 },
            ],
          },
        },
      ],
    };
  }, [filteredAssets]);

  return (
    <div className="space-y-4">
      {/* Category Filter Tabs */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs">
        {groups.map((grp) => (
          <button
            key={grp}
            onClick={() => setSelectedGroup(grp)}
            className={`px-2.5 py-1 rounded font-mono text-[11px] transition-colors ${
              selectedGroup === grp
                ? "bg-[var(--accent)] text-white font-medium"
                : "bg-[var(--bg-2)] text-[var(--tx-2)] hover:text-[var(--tx-0)]"
            }`}
          >
            {grp.toUpperCase()}
          </button>
        ))}
      </div>

      {/* ECharts Instance */}
      <div className="w-full h-[360px]">
        <ReactECharts
          option={option}
          style={{ height: "100%", width: "100%" }}
          onEvents={{
            click: (params: any) => {
              if (params.data?.item && onSelectAsset) {
                onSelectAsset(params.data.item.asset);
              }
            },
          }}
        />
      </div>

      {/* Quadrant Legend */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-[var(--line)] text-xs">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[var(--pos)]" />
          <span className="text-[var(--tx-1)]">Orderly Trend</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[var(--neg)]" />
          <span className="text-[var(--tx-1)]">Violent Trend</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[var(--accent)]" />
          <span className="text-[var(--tx-1)]">Quiet Range</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[var(--warn)]" />
          <span className="text-[var(--tx-1)]">Choppy</span>
        </div>
      </div>
    </div>
  );
};
