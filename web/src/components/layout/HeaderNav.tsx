'use client';

import React, { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ChevronDown } from 'lucide-react';

interface NavItem {
  label: string;
  href: string;
  description?: string;
  badge?: string;
}

interface NavGroup {
  name: string;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    name: 'Markets',
    items: [
      { label: 'Today (Market Pulse)', href: '/', description: 'Global market strip, attribution & state map' },
      { label: 'Asset Compare & Stress', href: '/compare', description: 'Compare any 2–4 assets & test portfolio macro shocks' },
      { label: 'Country Profiles', href: '/countries', description: '10 global equity markets & FX performance' },
      { label: 'Sector Breakdown', href: '/sectors', description: '11 GICS sector performance and dispersion' },
      { label: 'Historical Analogues', href: '/analogues', description: 'Nearest-neighbor historical macro trajectory search' },
      { label: 'Cross-Asset Radar', href: '/radar', description: 'Divergence feed, CUSUM detector & Z-score bands' },
      { label: 'Cross-Asset Matrix', href: '/cross-asset', description: 'Pairwise correlation and lead-lag matrix' },
    ],
  },
  {
    name: 'Drivers & Rates',
    items: [
      { label: 'Macro Drivers', href: '/drivers', description: 'Time-varying multi-factor beta sensitivities' },
      { label: 'Yield Curve Lab', href: '/rates', description: 'Sovereign curve anatomy & global spreads' },
      { label: 'Futures Positioning', href: '/positioning', description: 'CFTC speculative crowding & percentiles' },
      { label: 'Central Bank Events', href: '/events', description: 'FOMC & RBI MPC policy reaction studies' },
    ],
  },
  {
    name: 'Risk & Vol',
    items: [
      { label: 'Risk Lab', href: '/risk', description: 'Volatility cones, drawdowns & EVT tail risk' },
      { label: 'Implied Volatility', href: '/volatility', description: 'VIX term structure, SKEW & variance risk premium' },
      { label: 'Connectedness', href: '/connectedness', description: 'Cross-market shock spillover network' },
    ],
  },
  {
    name: 'Macro & Liquidity',
    items: [
      { label: 'Net Liquidity', href: '/liquidity', description: 'Fed balance sheet, TGA & reverse repo' },
      { label: 'Global Credit & Reserves', href: '/global', description: 'BIS REER, Treasury TIC holdings & central banks' },
      { label: 'Energy Fundamentals', href: '/fundamentals', description: 'EIA 5-year seasonal inventory bands' },
      { label: 'India Flows', href: '/india', description: 'FII/DII equity & debt flow monitor' },
    ],
  },
  {
    name: 'Methodology & Audit',
    items: [
      { label: 'Track Record', href: '/track-record', description: 'Out-of-sample validation vs base rates' },
      { label: 'Methodology', href: '/methodology', description: 'Mathematical formulas & statistical specifications' },
      { label: 'Source Data Health', href: '/data-health', description: 'Upstream feed freshness & circuit breaker status' },
      { label: 'Corrections Log', href: '/corrections', description: 'Transparent error & revision audit log' },
    ],
  },
];

export function HeaderNav() {
  const pathname = usePathname();
  const [openGroup, setOpenGroup] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpenGroup(null);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Close dropdown on route change
  useEffect(() => {
    setOpenGroup(null);
  }, [pathname]);

  const isGroupActive = (group: NavGroup) => {
    return group.items.some((item) => pathname === item.href || (item.href !== '/' && pathname?.startsWith(item.href)));
  };

  return (
    <div ref={containerRef} className="flex items-center space-x-1 relative">
      {NAV_GROUPS.map((group) => {
        const isOpen = openGroup === group.name;
        const active = isGroupActive(group);

        return (
          <div key={group.name} className="relative">
            <button
              type="button"
              onClick={() => setOpenGroup(isOpen ? null : group.name)}
              onMouseEnter={() => setOpenGroup(group.name)}
              className={`flex items-center space-x-1 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                active
                  ? 'text-[#38bdf8] bg-[#121c29]'
                  : isOpen
                  ? 'text-[#e8edf5] bg-[#121823]'
                  : 'text-[#a9b4c4] hover:text-[#e8edf5] hover:bg-[#121823]'
              }`}
            >
              <span>{group.name}</span>
              <ChevronDown
                className={`w-3 h-3 transition-transform duration-200 ${
                  isOpen ? 'rotate-180 text-[#38bdf8]' : 'text-[#6f7c90]'
                }`}
              />
            </button>

            {/* Dropdown Menu */}
            {isOpen && (
              <div
                onMouseLeave={() => setOpenGroup(null)}
                className="absolute top-full left-0 mt-1 w-64 rounded-lg bg-[#0c1017] border border-[#1d2633] shadow-2xl p-2 z-50 animate-in fade-in slide-in-from-top-1 duration-150"
              >
                <div className="space-y-1">
                  {group.items.map((item) => {
                    const isItemActive = pathname === item.href;
                    return (
                      <Link
                        key={item.href}
                        href={item.href}
                        className={`block px-3 py-2 rounded text-xs transition-colors ${
                          isItemActive
                            ? 'bg-[#182638] text-[#38bdf8]'
                            : 'hover:bg-[#121823] text-[#cbd5e1] hover:text-[#e8edf5]'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium">{item.label}</span>
                          {item.badge && (
                            <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-[#16202c] border border-[#26354a] text-[#38bdf8]">
                              {item.badge}
                            </span>
                          )}
                        </div>
                        {item.description && (
                          <div className="text-[10px] text-[#6f7c90] mt-0.5 leading-snug line-clamp-1">
                            {item.description}
                          </div>
                        )}
                      </Link>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function MobileNav() {
  const pathname = usePathname();
  const [openGroup, setOpenGroup] = useState<string | null>(null);

  return (
    <div className="space-y-2 py-2">
      {NAV_GROUPS.map((group) => {
        const isOpen = openGroup === group.name;
        return (
          <div key={group.name} className="border-b border-[#1d2633] pb-1.5">
            <button
              onClick={() => setOpenGroup(isOpen ? null : group.name)}
              className="w-full flex items-center justify-between py-1 text-xs font-semibold text-[#89a4c7]"
            >
              <span>{group.name}</span>
              <ChevronDown className={`w-3.5 h-3.5 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>
            {isOpen && (
              <div className="space-y-1 pt-1.5 pl-2">
                {group.items.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`block py-1 text-xs ${
                      pathname === item.href ? 'text-[#38bdf8] font-bold' : 'text-[#a9b4c4]'
                    }`}
                  >
                    {item.label}
                  </Link>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
