// web/src/app/context/[id]/page.tsx
import React from "react";
import { ContextClientView } from "./ContextClientView";

export function generateStaticParams() {
  const ALL_SERIES_IDS = [
    "spx", "ndx", "rut", "dji", "rsp", "dax", "ftse", "n225", "hsi", "ssec", "eem",
    "tsse", "ibov", "kospi", "asx200",
    "xlk", "xlf", "xlv", "xly", "xli", "xlp", "xle", "xlu", "xlre", "xlb", "xlc",
    "nifty50", "niftybank", "sensex", "niftymid", "niftysmall", "niftyit", "niftypharma",
    "niftyfmcg", "niftyauto", "niftymetal", "niftyrealty", "indiavix",
    "us2y", "us5y", "us10y", "us30y", "us2s10s", "us10y_real", "us10y_breakeven", "sofr",
    "dxy", "eurusd", "usdjpy", "gbpusd", "usdinr", "usdcny", "usdcad", "usdbrl", "usdkrw", "audusd",
    "brent", "wti", "gold", "silver", "copper", "natgas",
    "vix", "hy_spread", "ig_spread",
    "fed_assets", "tga", "rrp"
  ];
  return ALL_SERIES_IDS.map((id) => ({ id }));
}

export default function ContextPage({ params }: { params: { id: string } }) {
  return <ContextClientView id={params.id} />;
}
