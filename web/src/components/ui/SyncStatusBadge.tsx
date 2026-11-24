"use client";

import React, { useEffect, useState } from "react";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

export function SyncStatusBadge() {
  const [syncedDate, setSyncedDate] = useState<string>("2026-11-24");

  useEffect(() => {
    async function loadDate() {
      try {
        const res = await fetch(`${basePath}/data/v1/manifest.json`);
        if (res.ok) {
          const json = await res.json();
          if (json.as_of) {
            setSyncedDate(json.as_of);
          }
        }
      } catch (e) {
        // Fallback to static as_of
      }
    }
    loadDate();
  }, []);

  return (
    <div className="hidden sm:flex items-center space-x-1.5 px-2 py-1 rounded bg-[#121823] border border-[#1d2633]">
      <span className="w-2 h-2 rounded-full bg-[#3fb6a8]"></span>
      <span className="text-[#a9b4c4] text-[11px]">Synced: {syncedDate}</span>
    </div>
  );
}
