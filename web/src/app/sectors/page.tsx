// web/src/app/sectors/page.tsx
import React from "react";
import { Metadata } from "next";
import { SectorClientView } from "./SectorClientView";

export const metadata: Metadata = {
  title: "Sectors | MacroMonitor — Macro Statistics",
  description:
    "Macro sector attribution, relative rotation cycles, and multi-horizon performance heatmaps for US and India markets.",
};

export default function SectorsPage() {
  return (
    <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <SectorClientView />
    </main>
  );
}
