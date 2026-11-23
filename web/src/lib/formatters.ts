export function fmtPct(val: number | null | undefined, digits = 1, signed = false): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  const pct = val * 100;
  const prefix = signed && pct > 0 ? "+" : "";
  return `${prefix}${pct.toFixed(digits)}%`;
}

export function fmtPercentile(val: number | null | undefined): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  const rounded = Math.round(val);
  const mod100 = rounded % 100;
  let suffix = "th";
  if (mod100 < 11 || mod100 > 13) {
    if (rounded % 10 === 1) suffix = "st";
    else if (rounded % 10 === 2) suffix = "nd";
    else if (rounded % 10 === 3) suffix = "rd";
  }
  return `${rounded}${suffix}`;
}

export function fmtNum(val: number | null | undefined, digits = 2, signed = false): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  const prefix = signed && val > 0 ? "+" : "";
  return `${prefix}${val.toFixed(digits)}`;
}

export function fmtMetricValue(metricKey: string, val: number | null | undefined, isRate = false): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  if (metricKey === "vol20") {
    return isRate ? `${val.toFixed(2)} pts` : fmtPct(val, 1, false);
  }
  if (metricKey === "ret_20d") {
    return isRate ? `${val > 0 ? "+" : ""}${(val * 100).toFixed(0)} bps` : fmtPct(val, 2, true);
  }
  if (metricKey === "drawdown_252d") {
    return fmtPct(val, 1, false);
  }
  return fmtNum(val, 2, true);
}

export function fmtBillions(val: number | null | undefined, signed = false, digits = 1): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  const absFormatted = `$${Math.abs(val).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}B`;
  if (val < 0) return `-${absFormatted}`;
  if (signed && val > 0) return `+${absFormatted}`;
  return absFormatted;
}

export function directionArrow(val: number | null | undefined): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "·";
  if (val > 0) return "▲";
  if (val < 0) return "▼";
  return "◆";
}

export function groupLabel(group: string): string {
  const labels: Record<string, string> = {
    equity_global: "Global Equities",
    equity_index: "Global Equities",
    equity_india: "India Sectors",
    india_equity: "India Equities",
    india_constituents: "India Constituents",
    equity_us_sector: "US Sectors",
    sector: "US Sectors",
    rates: "Rates & Duration",
    rate: "Rates & Duration",
    fx: "FX Spot",
    commodities: "Commodities",
    commodity: "Commodities",
    risk: "Risk Gauges",
    credit: "Credit Proxy",
    cross_asset: "Cross-Asset",
    macro: "Macro Stress",
  };
  return labels[group] ?? group;
}

export function downloadCsv(filename: string, headers: string[], rows: (string | number | boolean | null | undefined)[][]): void {
  const escape = (v: string | number | boolean | null | undefined): string => {
    if (v === null || v === undefined) return "";
    const str = String(v);
    if (str.includes(",") || str.includes('"') || str.includes("\n")) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };
  const csv = [headers.map(escape).join(","), ...rows.map((r) => r.map(escape).join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
