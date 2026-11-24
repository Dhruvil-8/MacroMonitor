// web/src/components/charts/WorldChoropleth.tsx
// C03: World spatial performance and stress map with metric selector (Section 5.1)
"use client";

import React, { useState, useEffect, useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { CountryMatrixRow } from "@/lib/data/schemas";
import { loadCountries, loadVolLandscape } from "@/lib/data/client";

interface CountryPoint {
  code: string;
  name: string;
  indexId: string;
  return1d: number;
  return5d: number;
  return20d: number;
  vol20d: number;
}

const FALLBACK_COUNTRIES: CountryPoint[] = [
  { code: "US", name: "United States", indexId: "spx", return1d: 0.0473, return5d: -0.0242, return20d: -0.0507, vol20d: 32.0 },
  { code: "CA", name: "Canada", indexId: "tsse", return1d: 0.0256, return5d: -0.0499, return20d: -0.0706, vol20d: 23.1 },
  { code: "BR", name: "Brazil", indexId: "ibov", return1d: -0.0063, return5d: -0.0506, return20d: -0.3277, vol20d: 18.5 },
  { code: "GB", name: "United Kingdom", indexId: "ftse", return1d: 0.0087, return5d: -0.0474, return20d: -0.0444, vol20d: 14.8 },
  { code: "DE", name: "Germany", indexId: "dax", return1d: 0.0161, return5d: -0.0547, return20d: -0.0211, vol20d: 21.8 },
  { code: "JP", name: "Japan", indexId: "n225", return1d: 0.0405, return5d: 0.0014, return20d: -0.0683, vol20d: 39.0 },
  { code: "CN", name: "China", indexId: "ssec", return1d: 0.0071, return5d: -0.0076, return20d: -0.0359, vol20d: 14.1 },
  { code: "IN", name: "India", indexId: "nifty50", return1d: -0.0199, return5d: 0.0137, return20d: 0.1463, vol20d: 15.6 },
  { code: "KR", name: "South Korea", indexId: "kospi", return1d: 0.0734, return5d: 0.0106, return20d: -0.0914, vol20d: 60.6 },
  { code: "AU", name: "Australia", indexId: "asx200", return1d: 0.0136, return5d: -0.0361, return20d: -0.0685, vol20d: 16.1 },
];

export interface WorldChoroplethProps {
  countries?: CountryMatrixRow[];
  lens?: "usd" | "local" | "fx";
}

export const WorldChoropleth: React.FC<WorldChoroplethProps> = ({
  countries: propCountries,
  lens = "usd",
}) => {
  const [metric, setMetric] = useState<"1D" | "5D" | "20D" | "vol">("1D");
  const [loadedCountries, setLoadedCountries] = useState<CountryMatrixRow[] | null>(null);
  const [volMap, setVolMap] = useState<Record<string, number>>({});

  useEffect(() => {
    if (!propCountries) {
      loadCountries().then((res) => {
        if (res.status !== "error") {
          setLoadedCountries(res.data.data.countries);
        }
      });
    }
    loadVolLandscape().then((res) => {
      if (res.status !== "error" && res.data.data?.landscape) {
        const vm: Record<string, number> = {};
        for (const item of res.data.data.landscape) {
          if (item.asset) {
            vm[item.asset.toLowerCase()] = (item.vol20 || 0) * 100;
          }
        }
        setVolMap(vm);
      }
    });
  }, [propCountries]);

  const activeCountriesList = propCountries || loadedCountries;

  const dataPoints: CountryPoint[] = useMemo(() => {
    if (!activeCountriesList || activeCountriesList.length === 0) {
      return FALLBACK_COUNTRIES;
    }
    return activeCountriesList.map((c) => {
      const retDict = lens === "local" ? c.returns_local : lens === "fx" ? c.returns_fx : c.returns_usd;
      const r1 = retDict["1D"] ?? 0;
      const r5 = retDict["5D"] ?? 0;
      const r20 = retDict["1M"] ?? retDict["20D"] ?? 0;
      const vol =
        volMap[c.index_id.toLowerCase()] ??
        FALLBACK_COUNTRIES.find((fc) => fc.indexId === c.index_id.toLowerCase())?.vol20d ??
        15.0;

      return {
        code: c.code.toUpperCase(),
        name: c.name,
        indexId: c.index_id,
        return1d: r1,
        return5d: r5,
        return20d: r20,
        vol20d: vol,
      };
    });
  }, [activeCountriesList, lens, volMap]);

  const getValue = (c: CountryPoint) => {
    if (metric === "1D") return c.return1d * 100;
    if (metric === "5D") return c.return5d * 100;
    if (metric === "20D") return c.return20d * 100;
    return c.vol20d;
  };

  const isVol = metric === "vol";

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "item",
      backgroundColor: "#0c1017",
      borderColor: "#1d2633",
      textStyle: { color: "#e8edf5", fontSize: 11, fontFamily: "monospace" },
      formatter: (params: any) => {
        const c = dataPoints[params.dataIndex];
        if (!c) return "";
        return `
          <div style="font-weight:600;margin-bottom:4px;color:#e8edf5">${c.name} (${c.code})</div>
          <div style="color:#6f7c90">1D Daily Return: <span style="color:${c.return1d >= 0 ? "#3fb6a8" : "#e8765a"}">${(c.return1d * 100).toFixed(2)}%</span></div>
          <div style="color:#6f7c90">5D Return (1W): <span style="color:${c.return5d >= 0 ? "#3fb6a8" : "#e8765a"}">${(c.return5d * 100).toFixed(2)}%</span></div>
          <div style="color:#6f7c90">20D Return (1M): <span style="color:${c.return20d >= 0 ? "#3fb6a8" : "#e8765a"}">${(c.return20d * 100).toFixed(2)}%</span></div>
          <div style="color:#6f7c90">20D Realized Vol: <span style="color:#e6b94d">${c.vol20d.toFixed(1)}%</span></div>
        `;
      },
    },
    grid: { left: 40, right: 30, top: 40, bottom: 40 },
    xAxis: {
      type: "category",
      data: dataPoints.map((c) => c.code),
      axisLine: { lineStyle: { color: "#1d2633" } },
      axisLabel: { color: "#a9b4c4", fontFamily: "monospace", fontSize: 11 },
    },
    yAxis: {
      type: "value",
      axisLine: { lineStyle: { color: "#1d2633" } },
      splitLine: { lineStyle: { color: "#121823" } },
      axisLabel: {
        color: "#6f7c90",
        fontFamily: "monospace",
        fontSize: 10,
        formatter: (val: number) => `${val.toFixed(1)}%`,
      },
    },
    series: [
      {
        name: isVol ? "20D Realized Volatility" : `${metric} Return`,
        type: "bar",
        barWidth: "40%",
        data: dataPoints.map((c) => {
          const val = getValue(c);
          return {
            value: val,
            itemStyle: {
              color: isVol
                ? val > 25
                  ? "#e8765a"
                  : val > 18
                  ? "#e6b94d"
                  : "#3fb6a8"
                : val >= 0
                ? "#3fb6a8"
                : "#e8765a",
              borderRadius: [3, 3, 0, 0],
            },
          };
        }),
      },
    ],
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <span className="text-[11px] font-mono text-[var(--tx-2)]">
          Spatial Cross-Market Horizon ({lens.toUpperCase()} Lens):
        </span>
        <div className="flex items-center gap-1 bg-[var(--bg-0)] p-1 rounded border border-[var(--line)] text-xs font-mono">
          {(["1D", "5D", "20D", "vol"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={`px-2 py-0.5 rounded transition ${
                metric === m
                  ? "bg-[var(--accent)] text-slate-900 font-semibold"
                  : "text-[var(--tx-2)] hover:text-[var(--tx-0)]"
              }`}
            >
              {m === "vol" ? "20D Vol" : `${m} Return`}
            </button>
          ))}
        </div>
      </div>
      <ReactECharts option={option} style={{ height: "300px", width: "100%" }} />
    </div>
  );
};
