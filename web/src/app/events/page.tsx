"use client";

import { useEffect, useState } from "react";
import { EventStudyEnvelope } from "@/lib/data/schemas";
import { loadEvents } from "@/lib/data/client";
import { ReactionRibbon } from "@/components/charts/ReactionRibbon";
import { HealthDot } from "@/components/ui/HealthDot";
import { DataAgeBadge } from "@/components/ui/DataAgeBadge";

export default function EventsPage() {
  const [eventType, setEventType] = useState<"fomc" | "rbi_mpc">("fomc");
  const [data, setData] = useState<EventStudyEnvelope | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    setError(null);
    loadEvents(eventType)
      .then((res) => {
        if (res.status === "error") {
          setError(res.error || "Failed to load event study data");
        } else {
          setData(res.data);
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [eventType]);

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto pb-12">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[#1d2633]">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-widest text-[#6f7c90]">
            Central Bank Policy Reactions • Module P6
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[#e8edf5] mt-0.5">
            Event Studies: Central Bank Decision Days
          </h1>
          <p className="text-xs text-[#8c9bb0] mt-1">
            Abnormal returns and cumulative abnormal return (CAR [-5, +10] days) around policy announcement dates with block-bootstrap intervals.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Institution Selector */}
          <div className="flex rounded border border-[#1d2633] overflow-hidden font-mono text-xs">
            <button
              onClick={() => setEventType("fomc")}
              className={`px-3 py-1.5 transition-colors ${
                eventType === "fomc"
                  ? "bg-[var(--accent)] text-white font-medium"
                  : "bg-[#0c1017] text-[#8c9bb0] hover:text-[#e8edf5]"
              }`}
            >
              Federal Reserve (FOMC)
            </button>
            <button
              onClick={() => setEventType("rbi_mpc")}
              className={`px-3 py-1.5 transition-colors ${
                eventType === "rbi_mpc"
                  ? "bg-[var(--accent)] text-white font-medium"
                  : "bg-[#0c1017] text-[#8c9bb0] hover:text-[#e8edf5]"
              }`}
            >
              RBI MPC (India)
            </button>
          </div>
          <HealthDot status="ok" label="Calendars Synchronized" />
          <DataAgeBadge asOf={data?.as_of || "2026-11-20"} />
        </div>
      </div>

      {error ? (
        <div className="rounded-[6px] border border-rose-500 bg-[#0c1017] p-5 text-sm text-rose-400">
          Failed to load event study: {error}
        </div>
      ) : !data ? (
        <div className="h-96 rounded-[6px] border border-[#1d2633] bg-[#0c1017] animate-pulse flex items-center justify-center font-mono text-xs text-[#6f7c90]">
          Loading {eventType.toUpperCase()} event study and CAR bootstrap distribution...
        </div>
      ) : (
        <ReactionRibbon data={data} />
      )}
    </div>
  );
}
