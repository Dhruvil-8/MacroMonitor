// web/src/app/risk/page.tsx
import React from "react";
import { Metadata } from "next";
import { RiskClientView } from "./RiskClientView";

export const metadata: Metadata = {
  title: "Risk Lab | MacroMonitor — Macro Statistics",
  description:
    "Quantitative risk diagnostics: realized volatility cones, fat-tailed Expected Shortfall vs Gaussian VaR, ARCH volatility clustering, underwater drawdowns, PCA market synchronization, and downside stress correlation premiums.",
};

export default function RiskPage() {
  return (
    <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <RiskClientView />
    </main>
  );
}
