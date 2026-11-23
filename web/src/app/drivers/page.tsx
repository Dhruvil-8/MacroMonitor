// web/src/app/drivers/page.tsx
import React from "react";
import { Metadata } from "next";
import { DriverClientView } from "./DriverClientView";

export const metadata: Metadata = {
  title: "Macro Drivers | MacroMonitor — Macro Statistics",
  description:
    "Cross-asset driver sensitivity matrix, rolling regressions with Newey-West HAC errors, stability screening, and forest plots for equity markets.",
};

export default function DriversPage() {
  return (
    <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <DriverClientView />
    </main>
  );
}
