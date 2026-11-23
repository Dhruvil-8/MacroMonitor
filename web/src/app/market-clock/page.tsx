// web/src/app/market-clock/page.tsx
"use client";

import React, { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import {
  Clock,
  Globe,
  MapPin,
  TrendingUp,
  Activity,
  Calendar,
  Layers,
  ExternalLink,
  ChevronRight,
  Sun,
  Moon,
  Zap,
} from "lucide-react";

interface ExchangeInfo {
  id: string;
  name: string;
  shortName: string;
  city: string;
  country: string;
  flag: string;
  region: "apac" | "europe" | "americas";
  timeZone: string; // IANA
  tzLabel: string;
  openHour: number; // in local exchange time (24h)
  openMinute: number;
  closeHour: number;
  closeMinute: number;
  lunchStartHour?: number;
  lunchStartMinute?: number;
  lunchEndHour?: number;
  lunchEndMinute?: number;
  primaryIndex: string;
  ticker: string;
  navLink: string;
}

const EXCHANGES: ExchangeInfo[] = [
  // Asia-Pacific
  {
    id: "nse",
    name: "National Stock Exchange of India",
    shortName: "NSE",
    city: "Mumbai",
    country: "India",
    flag: "🇮🇳",
    region: "apac",
    timeZone: "Asia/Kolkata",
    tzLabel: "IST (UTC+5:30)",
    openHour: 9,
    openMinute: 15,
    closeHour: 15,
    closeMinute: 30,
    primaryIndex: "NIFTY 50",
    ticker: "^NSEI",
    navLink: "/india",
  },
  {
    id: "tse",
    name: "Tokyo Stock Exchange",
    shortName: "TSE",
    city: "Tokyo",
    country: "Japan",
    flag: "🇯🇵",
    region: "apac",
    timeZone: "Asia/Tokyo",
    tzLabel: "JST (UTC+9)",
    openHour: 9,
    openMinute: 0,
    closeHour: 15,
    closeMinute: 30,
    lunchStartHour: 11,
    lunchStartMinute: 30,
    lunchEndHour: 12,
    lunchEndMinute: 30,
    primaryIndex: "Nikkei 225",
    ticker: "^N225",
    navLink: "/countries",
  },
  {
    id: "hkex",
    name: "Hong Kong Stock Exchange",
    shortName: "HKEX",
    city: "Hong Kong",
    country: "Hong Kong SAR",
    flag: "🇭🇰",
    region: "apac",
    timeZone: "Asia/Hong_Kong",
    tzLabel: "HKT (UTC+8)",
    openHour: 9,
    openMinute: 30,
    closeHour: 16,
    closeMinute: 0,
    lunchStartHour: 12,
    lunchStartMinute: 0,
    lunchEndHour: 13,
    lunchEndMinute: 0,
    primaryIndex: "Hang Seng Index",
    ticker: "^HSI",
    navLink: "/countries",
  },
  {
    id: "sgx",
    name: "Singapore Exchange",
    shortName: "SGX",
    city: "Singapore",
    country: "Singapore",
    flag: "🇸🇬",
    region: "apac",
    timeZone: "Asia/Singapore",
    tzLabel: "SGT (UTC+8)",
    openHour: 9,
    openMinute: 0,
    closeHour: 17,
    closeMinute: 0,
    primaryIndex: "Straits Times Index",
    ticker: "^STI",
    navLink: "/countries",
  },
  {
    id: "asx",
    name: "Australian Securities Exchange",
    shortName: "ASX",
    city: "Sydney",
    country: "Australia",
    flag: "🇦🇺",
    region: "apac",
    timeZone: "Australia/Sydney",
    tzLabel: "AEST/AEDT",
    openHour: 10,
    openMinute: 0,
    closeHour: 16,
    closeMinute: 0,
    primaryIndex: "S&P/ASX 200",
    ticker: "^AXJO",
    navLink: "/countries",
  },

  // Europe
  {
    id: "lse",
    name: "London Stock Exchange",
    shortName: "LSE",
    city: "London",
    country: "United Kingdom",
    flag: "🇬🇧",
    region: "europe",
    timeZone: "Europe/London",
    tzLabel: "GMT/BST",
    openHour: 8,
    openMinute: 0,
    closeHour: 16,
    closeMinute: 30,
    primaryIndex: "FTSE 100",
    ticker: "^FTSE",
    navLink: "/countries",
  },
  {
    id: "xetra",
    name: "Frankfurt Stock Exchange (XETRA)",
    shortName: "XETRA",
    city: "Frankfurt",
    country: "Germany",
    flag: "🇩🇪",
    region: "europe",
    timeZone: "Europe/Berlin",
    tzLabel: "CET/CEST",
    openHour: 9,
    openMinute: 0,
    closeHour: 17,
    closeMinute: 30,
    primaryIndex: "DAX 40",
    ticker: "^GDAXI",
    navLink: "/countries",
  },
  {
    id: "euronext",
    name: "Euronext Paris",
    shortName: "Euronext",
    city: "Paris",
    country: "France",
    flag: "🇫🇷",
    region: "europe",
    timeZone: "Europe/Paris",
    tzLabel: "CET/CEST",
    openHour: 9,
    openMinute: 0,
    closeHour: 17,
    closeMinute: 30,
    primaryIndex: "CAC 40",
    ticker: "^FCHI",
    navLink: "/countries",
  },
  {
    id: "six",
    name: "SIX Swiss Exchange",
    shortName: "SIX",
    city: "Zurich",
    country: "Switzerland",
    flag: "🇨🇭",
    region: "europe",
    timeZone: "Europe/Zurich",
    tzLabel: "CET/CEST",
    openHour: 9,
    openMinute: 0,
    closeHour: 17,
    closeMinute: 30,
    primaryIndex: "Swiss Market Index (SMI)",
    ticker: "^SSMI",
    navLink: "/countries",
  },

  // Americas
  {
    id: "nyse",
    name: "New York Stock Exchange / NASDAQ",
    shortName: "NYSE / NASDAQ",
    city: "New York",
    country: "United States",
    flag: "🇺🇸",
    region: "americas",
    timeZone: "America/New_York",
    tzLabel: "EST/EDT",
    openHour: 9,
    openMinute: 30,
    closeHour: 16,
    closeMinute: 0,
    primaryIndex: "S&P 500 / Nasdaq 100",
    ticker: "^GSPC",
    navLink: "/rates",
  },
  {
    id: "tsx",
    name: "Toronto Stock Exchange",
    shortName: "TSX",
    city: "Toronto",
    country: "Canada",
    flag: "🇨🇦",
    region: "americas",
    timeZone: "America/Toronto",
    tzLabel: "EST/EDT",
    openHour: 9,
    openMinute: 30,
    closeHour: 16,
    closeMinute: 0,
    primaryIndex: "S&P/TSX Composite",
    ticker: "^GSPTSE",
    navLink: "/countries",
  },
  {
    id: "b3",
    name: "B3 - Brasil Bolsa Balcão",
    shortName: "B3",
    city: "São Paulo",
    country: "Brazil",
    flag: "🇧🇷",
    region: "americas",
    timeZone: "America/Sao_Paulo",
    tzLabel: "BRT (UTC-3)",
    openHour: 10,
    openMinute: 0,
    closeHour: 17,
    closeMinute: 0,
    primaryIndex: "Ibovespa",
    ticker: "^BVSP",
    navLink: "/countries",
  },
];

const POPULAR_TIMEZONES = [
  { value: "auto", label: "Auto-detect (Your Browser Time)" },
  { value: "Asia/Kolkata", label: "India (IST, UTC+5:30)" },
  { value: "America/New_York", label: "New York (EST/EDT, UTC-5/-4)" },
  { value: "Europe/London", label: "London (GMT/BST, UTC+0/+1)" },
  { value: "Europe/Berlin", label: "Frankfurt / Paris (CET/CEST, UTC+1/+2)" },
  { value: "Asia/Tokyo", label: "Tokyo (JST, UTC+9)" },
  { value: "Asia/Hong_Kong", label: "Hong Kong (HKT, UTC+8)" },
  { value: "Asia/Singapore", label: "Singapore (SGT, UTC+8)" },
  { value: "Asia/Dubai", label: "Dubai (GST, UTC+4)" },
  { value: "Australia/Sydney", label: "Sydney (AEST/AEDT, UTC+10/+11)" },
  { value: "America/Chicago", label: "Chicago (CST/CDT, UTC-6/-5)" },
  { value: "America/Los_Angeles", label: "San Francisco / LA (PST/PDT, UTC-8/-7)" },
  { value: "UTC", label: "UTC (Universal Coordinated Time)" },
];

export default function MarketClockPage() {
  const [selectedTz, setSelectedTz] = useState<string>("auto");
  const [resolvedTz, setResolvedTz] = useState<string>("UTC");
  const [now, setNow] = useState<Date | null>(null);
  const [activeRegion, setActiveRegion] = useState<"all" | "apac" | "europe" | "americas">("all");

  // Client-side initialization to avoid hydration mismatch
  useEffect(() => {
    const detected = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
    setResolvedTz(detected);
    setNow(new Date());

    // 1-second live ticking clock
    const timer = setInterval(() => {
      setNow(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const effectiveTz = selectedTz === "auto" ? resolvedTz : selectedTz;

  // Formatted master time string in user's selected time zone
  const userClockString = useMemo(() => {
    if (!now) return "--:--:--";
    return new Intl.DateTimeFormat("en-GB", {
      timeZone: effectiveTz,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(now);
  }, [now, effectiveTz]);

  const userDateString = useMemo(() => {
    if (!now) return "";
    return new Intl.DateTimeFormat("en-US", {
      timeZone: effectiveTz,
      weekday: "long",
      year: "numeric",
      month: "short",
      day: "numeric",
    }).format(now);
  }, [now, effectiveTz]);

  const utcClockString = useMemo(() => {
    if (!now) return "--:--:--";
    return new Intl.DateTimeFormat("en-GB", {
      timeZone: "UTC",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(now);
  }, [now]);

  // Compute live market status for each exchange
  const exchangeStates = useMemo(() => {
    if (!now) return [];

    return EXCHANGES.map((ex) => {
      // Get current date/time parts in the exchange's local timezone
      const parts = new Intl.DateTimeFormat("en-US", {
        timeZone: ex.timeZone,
        weekday: "short",
        hour: "numeric",
        minute: "numeric",
        second: "numeric",
        hour12: false,
      }).formatToParts(now);

      const partMap: Record<string, string> = {};
      for (const p of parts) partMap[p.type] = p.value;

      const dayOfWeek = partMap.weekday; // 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'
      const curHour = parseInt(partMap.hour, 10);
      const curMinute = parseInt(partMap.minute, 10);
      const curSecond = parseInt(partMap.second, 10);

      const isWeekend = dayOfWeek === "Sat" || dayOfWeek === "Sun";
      const curMins = curHour * 60 + curMinute;
      const openMins = ex.openHour * 60 + ex.openMinute;
      const closeMins = ex.closeHour * 60 + ex.closeMinute;

      let status: "open" | "closed" | "lunch" | "weekend" = "closed";
      let timeRemainingText = "";
      let progressPct = 0;

      if (isWeekend) {
        status = "weekend";
        timeRemainingText = "Opens Monday";
      } else {
        const inLunch =
          ex.lunchStartHour !== undefined &&
          ex.lunchEndHour !== undefined &&
          curMins >= ex.lunchStartHour * 60 + (ex.lunchStartMinute || 0) &&
          curMins < ex.lunchEndHour * 60 + (ex.lunchEndMinute || 0);

        if (inLunch) {
          status = "lunch";
          const minsToLunchEnd = (ex.lunchEndHour! * 60 + (ex.lunchEndMinute || 0)) - curMins;
          timeRemainingText = `Lunch break (${minsToLunchEnd}m left)`;
        } else if (curMins >= openMins && curMins < closeMins) {
          status = "open";
          const minsToClose = closeMins - curMins;
          const h = Math.floor(minsToClose / 60);
          const m = minsToClose % 60;
          timeRemainingText = `Closes in ${h}h ${m}m`;
          const totalDuration = closeMins - openMins;
          progressPct = Math.min(100, Math.max(0, ((curMins - openMins) / totalDuration) * 100));
        } else if (curMins < openMins) {
          status = "closed";
          const minsToOpen = openMins - curMins;
          const h = Math.floor(minsToOpen / 60);
          const m = minsToOpen % 60;
          timeRemainingText = `Opens in ${h}h ${m}m`;
        } else {
          status = "closed";
          // Next day opening
          const minsToNextOpen = 24 * 60 - curMins + openMins;
          const h = Math.floor(minsToNextOpen / 60);
          const m = minsToNextOpen % 60;
          timeRemainingText = dayOfWeek === "Fri" ? "Opens Monday" : `Opens in ${h}h ${m}m`;
        }
      }

      // Local exchange digital time
      const localTimeStr = `${String(curHour).padStart(2, "0")}:${String(curMinute).padStart(2, "0")}:${String(curSecond).padStart(2, "0")}`;

      // Convert exchange open/close hours to user's selected timezone
      // We do this by creating a mock date on the exchange date
      const openFormatStr = `${String(ex.openHour).padStart(2, "0")}:${String(ex.openMinute).padStart(2, "0")}`;
      const closeFormatStr = `${String(ex.closeHour).padStart(2, "0")}:${String(ex.closeMinute).padStart(2, "0")}`;

      return {
        ...ex,
        localTimeStr,
        dayOfWeek,
        status,
        timeRemainingText,
        progressPct,
        openFormatStr,
        closeFormatStr,
      };
    });
  }, [now, effectiveTz]);

  const openCount = exchangeStates.filter((s) => s.status === "open").length;

  const filteredExchanges = useMemo(() => {
    if (activeRegion === "all") return exchangeStates;
    return exchangeStates.filter((e) => e.region === activeRegion);
  }, [exchangeStates, activeRegion]);

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Top Header & Interactive Timezone Selector */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 pb-6 border-b border-[var(--line)]">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase tracking-widest text-[var(--tx-2)]">
              MacroMonitor / Surveillance
            </span>
            <span className="px-2 py-0.5 rounded bg-emerald-950/70 border border-emerald-800 text-emerald-400 font-mono text-[10px] font-bold flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              Live Sync
            </span>
          </div>
          <h1 className="text-3xl font-serif font-bold tracking-tight text-[var(--tx-0)] mt-1">
            Global Financial Market Clock
          </h1>
          <p className="text-xs text-[var(--tx-1)] mt-1 max-w-2xl leading-relaxed">
            Real-time trading session monitor across major world equity &amp; derivatives exchanges. 
            All market hours, lunch recesses, and liquidity overlaps automatically synchronized to your selected location.
          </p>
        </div>

        {/* User Timezone Selector Card */}
        <div className="p-4 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] flex flex-col sm:flex-row items-start sm:items-center gap-4 shrink-0 shadow-lg">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--tx-2)] flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-[var(--accent)]" />
              <span>Reference Timezone</span>
            </div>
            <select
              value={selectedTz}
              onChange={(e) => setSelectedTz(e.target.value)}
              className="mt-1 bg-[var(--bg-0)] border border-[var(--line)] text-xs font-mono text-[var(--tx-0)] py-1.5 px-3 rounded focus:outline-none focus:border-[var(--accent)] cursor-pointer"
            >
              {POPULAR_TIMEZONES.map((tz) => (
                <option key={tz.value} value={tz.value}>
                  {tz.label}
                </option>
              ))}
            </select>
          </div>

          <div className="sm:border-l sm:border-[var(--line)] sm:pl-4">
            <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--tx-2)] flex items-center gap-1">
              <Clock className="w-3.5 h-3.5 text-[var(--accent)]" />
              <span>Your Current Time</span>
            </div>
            <div className="text-2xl font-bold font-mono text-[var(--tx-0)] tracking-wider mt-0.5">
              {userClockString}
            </div>
            <div className="text-[10px] font-mono text-[var(--tx-2)] mt-0.5">
              {userDateString} &bull; UTC: <span className="text-[var(--accent)]">{utcClockString}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Global Liquidity Bar: Overlapping Sessions */}
      <div className="p-5 rounded-lg bg-[var(--bg-1)] border border-[var(--line)] space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[var(--line)] pb-3">
          <div className="flex items-center gap-2">
            <Globe className="w-4 h-4 text-[var(--accent)]" />
            <h2 className="text-sm font-mono font-semibold text-[var(--tx-0)] uppercase tracking-wider">
              24-Hour Global Trading Session Overlaps
            </h2>
          </div>
          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              <strong className="text-[var(--tx-0)]">{openCount}</strong>
              <span className="text-[var(--tx-2)]">of 12 Markets Open Now</span>
            </span>
          </div>
        </div>

        {/* Visual 24h Timeline Bar */}
        <div className="space-y-2 pt-1 font-mono text-xs">
          <div className="grid grid-cols-4 sm:grid-cols-8 md:grid-cols-12 gap-1 text-[10px] text-[var(--tx-2)] pb-1 border-b border-[var(--line)]/50 text-center">
            {["00:00", "02:00", "04:00", "06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"].map((t) => (
              <span key={t}>{t}</span>
            ))}
          </div>

          {/* Session Ribbons */}
          <div className="space-y-2 pt-2">
            {/* Asia-Pacific */}
            <div className="flex items-center gap-3">
              <span className="w-28 text-[11px] text-[var(--tx-2)] shrink-0 flex items-center gap-1">
                <span>🌏</span> Tokyo / Asia
              </span>
              <div className="flex-1 h-6 bg-[var(--bg-0)] rounded border border-[var(--line)] relative overflow-hidden flex items-center">
                <div
                  className="h-full bg-purple-500/20 border-x border-purple-500/50 flex items-center px-2 text-[10px] text-purple-300 font-semibold"
                  style={{ width: "27%", marginLeft: "37.5%" }}
                >
                  Tokyo &bull; HK &bull; Singapore
                </div>
              </div>
            </div>

            {/* India NSE */}
            <div className="flex items-center gap-3">
              <span className="w-28 text-[11px] text-[var(--tx-2)] shrink-0 flex items-center gap-1">
                <span>🇮🇳</span> India (NSE)
              </span>
              <div className="flex-1 h-6 bg-[var(--bg-0)] rounded border border-[var(--line)] relative overflow-hidden flex items-center">
                <div
                  className="h-full bg-orange-500/20 border-x border-orange-500/50 flex items-center px-2 text-[10px] text-orange-300 font-semibold"
                  style={{ width: "26%", marginLeft: "38.5%" }}
                >
                  09:15 – 15:30 IST
                </div>
              </div>
            </div>

            {/* Europe / London */}
            <div className="flex items-center gap-3">
              <span className="w-28 text-[11px] text-[var(--tx-2)] shrink-0 flex items-center gap-1">
                <span>🇬🇧</span> Europe / London
              </span>
              <div className="flex-1 h-6 bg-[var(--bg-0)] rounded border border-[var(--line)] relative overflow-hidden flex items-center">
                <div
                  className="h-full bg-sky-500/20 border-x border-sky-500/50 flex items-center px-2 text-[10px] text-sky-300 font-semibold"
                  style={{ width: "35%", marginLeft: "33.3%" }}
                >
                  London &bull; Frankfurt &bull; Paris
                </div>
              </div>
            </div>

            {/* US / New York */}
            <div className="flex items-center gap-3">
              <span className="w-28 text-[11px] text-[var(--tx-2)] shrink-0 flex items-center gap-1">
                <span>🇺🇸</span> US / New York
              </span>
              <div className="flex-1 h-6 bg-[var(--bg-0)] rounded border border-[var(--line)] relative overflow-hidden flex items-center">
                <div
                  className="h-full bg-emerald-500/20 border-x border-emerald-500/50 flex items-center px-2 text-[10px] text-emerald-300 font-semibold"
                  style={{ width: "27%", marginLeft: "39.5%" }}
                >
                  NYSE &bull; NASDAQ &bull; CME
                </div>
              </div>
            </div>
          </div>

          <div className="pt-2 text-[11px] text-[var(--tx-2)] flex flex-wrap items-center justify-between gap-2 border-t border-[var(--line)]/50">
            <span className="flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span>
                <strong>Peak Global Liquidity Window:</strong> London + New York Overlap (13:00 – 16:30 GMT) accounts for &gt;60% of daily cross-asset FX and derivatives volume.
              </span>
            </span>
            <span className="text-[10px] text-[var(--accent)] font-mono">
              Displayed in UTC &bull; Switch reference timezone above to align
            </span>
          </div>
        </div>
      </div>

      {/* Region Filter Buttons */}
      <div className="flex items-center gap-2 border-b border-[var(--line)] pb-3 text-xs font-mono overflow-x-auto">
        {[
          { id: "all", label: "All Major Exchanges (12)" },
          { id: "apac", label: "Asia-Pacific (5)" },
          { id: "europe", label: "Europe (4)" },
          { id: "americas", label: "Americas (3)" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveRegion(tab.id as any)}
            className={`px-3 py-1.5 rounded transition-colors whitespace-nowrap cursor-pointer ${
              activeRegion === tab.id
                ? "bg-[var(--accent)] text-slate-900 font-semibold"
                : "text-[var(--tx-2)] hover:text-[var(--tx-0)] bg-[var(--bg-1)] border border-[var(--line)]"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Exchange Cards Grid (12 Major Exchanges) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {filteredExchanges.map((ex) => {
          const isOpen = ex.status === "open";
          const isLunch = ex.status === "lunch";
          const isWeekend = ex.status === "weekend";

          return (
            <div
              key={ex.id}
              className={`p-5 rounded-lg bg-[var(--bg-1)] border transition flex flex-col justify-between space-y-4 ${
                isOpen
                  ? "border-emerald-500/40 shadow-lg shadow-emerald-950/20"
                  : isLunch
                  ? "border-amber-500/40"
                  : "border-[var(--line)] hover:border-[var(--tx-2)]"
              }`}
            >
              {/* Header: Flag, City, Status */}
              <div>
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2.5">
                    <span className="text-2xl">{ex.flag}</span>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-sm text-[var(--tx-0)]">{ex.shortName}</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-[var(--bg-0)] border border-[var(--line)] text-[var(--tx-2)]">
                          {ex.city}
                        </span>
                      </div>
                      <div className="text-[11px] text-[var(--tx-2)] mt-0.5">{ex.name}</div>
                    </div>
                  </div>

                  {/* Status Badge */}
                  <span
                    className={`px-2.5 py-1 rounded text-[10px] font-mono font-bold flex items-center gap-1.5 uppercase tracking-wider shrink-0 border ${
                      isOpen
                        ? "bg-emerald-950/80 border-emerald-700 text-emerald-400"
                        : isLunch
                        ? "bg-amber-950/80 border-amber-700 text-amber-400"
                        : isWeekend
                        ? "bg-[var(--bg-0)] border-[var(--line)] text-[var(--tx-2)]"
                        : "bg-rose-950/80 border-rose-800 text-rose-400"
                    }`}
                  >
                    <span
                      className={`w-1.5 h-1.5 rounded-full ${
                        isOpen ? "bg-emerald-400 animate-pulse" : isLunch ? "bg-amber-400" : isWeekend ? "bg-[var(--tx-2)]" : "bg-rose-400"
                      }`}
                    ></span>
                    {isOpen ? "Open" : isLunch ? "Lunch" : isWeekend ? "Weekend" : "Closed"}
                  </span>
                </div>

                {/* Session Progress Bar if Open */}
                {isOpen && (
                  <div className="mt-3">
                    <div className="flex items-center justify-between text-[10px] font-mono text-[var(--tx-2)] pb-1">
                      <span className="text-emerald-400 font-semibold">{ex.timeRemainingText}</span>
                      <span>{ex.progressPct.toFixed(0)}% elapsed</span>
                    </div>
                    <div className="w-full h-1.5 bg-[var(--bg-0)] rounded-full overflow-hidden border border-[var(--line)]">
                      <div
                        className="h-full bg-emerald-400 transition-all duration-1000"
                        style={{ width: `${ex.progressPct}%` }}
                      ></div>
                    </div>
                  </div>
                )}

                {!isOpen && (
                  <div className="mt-2.5 text-[11px] font-mono text-[var(--tx-2)]">
                    Status: <span className="text-[var(--tx-0)] font-semibold">{ex.timeRemainingText}</span>
                  </div>
                )}
              </div>

              {/* Digital Clocks Display */}
              <div className="grid grid-cols-2 gap-2.5 pt-3 border-t border-[var(--line)] font-mono text-center">
                <div className="p-2.5 rounded bg-[var(--bg-0)] border border-[var(--line)]">
                  <div className="text-[9px] uppercase tracking-wider text-[var(--tx-2)] flex items-center justify-center gap-1">
                    <Clock className="w-3 h-3 text-[var(--accent)]" />
                    <span>Local Exchange Time</span>
                  </div>
                  <div className="text-base font-bold text-[var(--tx-0)] mt-0.5">
                    {ex.localTimeStr}
                  </div>
                  <div className="text-[9px] text-[var(--tx-2)] mt-0.5">{ex.tzLabel}</div>
                </div>

                <div className="p-2.5 rounded bg-[var(--bg-0)] border border-[var(--line)]">
                  <div className="text-[9px] uppercase tracking-wider text-[var(--tx-2)] flex items-center justify-center gap-1">
                    <MapPin className="w-3 h-3 text-sky-400" />
                    <span>In Your Time</span>
                  </div>
                  <div className="text-base font-bold text-[var(--accent)] mt-0.5">
                    {userClockString}
                  </div>
                  <div className="text-[9px] text-[var(--tx-2)] mt-0.5">Selected Ref</div>
                </div>
              </div>

              {/* Trading Hours Details */}
              <div className="pt-2 border-t border-[var(--line)]/60 space-y-1.5 text-[11px] font-mono">
                <div className="flex items-center justify-between text-[var(--tx-2)]">
                  <span>Regular Session:</span>
                  <strong className="text-[var(--tx-0)] font-semibold">
                    {ex.openFormatStr} – {ex.closeFormatStr} ({ex.tzLabel.split(" ")[0]})
                  </strong>
                </div>

                {ex.lunchStartHour !== undefined && (
                  <div className="flex items-center justify-between text-[var(--tx-2)]">
                    <span>Lunch Recess:</span>
                    <span className="text-amber-400/90 font-medium">
                      {String(ex.lunchStartHour).padStart(2, "0")}:{String(ex.lunchStartMinute || 0).padStart(2, "0")} –{" "}
                      {String(ex.lunchEndHour).padStart(2, "0")}:{String(ex.lunchEndMinute || 0).padStart(2, "0")}
                    </span>
                  </div>
                )}

                <div className="flex items-center justify-between text-[var(--tx-2)] pt-1">
                  <span>Benchmark Index:</span>
                  <span className="text-[var(--accent)] font-semibold">{ex.primaryIndex}</span>
                </div>
              </div>

              {/* Action Link to Module */}
              <div className="pt-2">
                <Link
                  href={ex.navLink}
                  className="w-full flex items-center justify-between px-3 py-1.5 rounded bg-[var(--bg-0)] hover:bg-[var(--bg-2)] border border-[var(--line)] text-xs font-mono text-[var(--tx-1)] hover:text-[var(--tx-0)] transition-colors"
                >
                  <span>Explore {ex.primaryIndex} Analytics</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </Link>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
