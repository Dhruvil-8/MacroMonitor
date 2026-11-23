"use client";

import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { useRadarStore } from "@/store/useRadarStore";

interface EChartBoxProps {
  option: EChartsOption;
  height?: number | string;
  ariaLabel?: string;
  onPointClick?: (params: unknown) => void;
}

export function EChartBox({ option, height = 360, ariaLabel = "Statistical chart", onPointClick }: EChartBoxProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const theme = useRadarStore((s) => s.theme);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const chart = echarts.init(el, undefined, { renderer: "canvas" });
    chartRef.current = chart;

    const ro = new ResizeObserver(() => {
      chart.resize();
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.setOption(option, { notMerge: true, lazyUpdate: true });
  }, [option, theme]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !onPointClick) return;
    chart.on("click", onPointClick);
    return () => {
      chart.off("click", onPointClick);
    };
  }, [onPointClick]);

  return (
    <div
      ref={containerRef}
      role="img"
      aria-label={ariaLabel}
      tabIndex={0}
      style={{ height: typeof height === "number" ? `${height}px` : height }}
      className="w-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 rounded-[4px]"
    />
  );
}
