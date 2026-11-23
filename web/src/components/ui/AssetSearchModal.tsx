'use client';

import { useEffect, useMemo, useState, useRef } from 'react';
import { Search, X, ArrowRight, CornerDownLeft } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { groupLabel } from '@/lib/formatters';
import { useRadarStore } from '@/store/useRadarStore';
import { loadStateMap } from '@/lib/data/client';
import type { StateMapItem } from '@/lib/data/schemas';

interface SearchItem {
  id: string;
  title: string;
  subtitle: string;
  category: 'Page' | 'Country' | 'Sector' | 'Positioning' | 'Driver' | 'Asset';
  href: string;
  badge?: string;
}

const STATIC_COMMAND_ITEMS: SearchItem[] = [
  // Navigation Pages
  { id: 'page-today', title: 'Today (Market State)', subtitle: 'Real-time multi-asset percentile snapshot', category: 'Page', href: '/' },
  { id: 'page-charts', title: 'Charts (51)', subtitle: 'Complete chart gallery', category: 'Page', href: '/charts' },
  { id: 'page-risk', title: 'Risk Lab & Regimes', subtitle: 'Volatility cones, drawdowns & EVT tails', category: 'Page', href: '/risk', badge: 'P10' },
  { id: 'page-rates', title: 'Yield Curve Lab', subtitle: 'Nelson-Siegel 3-factor curve anatomy & real yields', category: 'Page', href: '/rates', badge: 'P4' },
  { id: 'page-volatility', title: 'Implied Volatility', subtitle: 'VIX term structure, SKEW & VRP trailing proxy', category: 'Page', href: '/volatility', badge: 'P3' },
  { id: 'page-positioning', title: 'Futures Positioning', subtitle: 'CFTC speculative crowding & hysteresis ladder', category: 'Page', href: '/positioning', badge: 'P2' },
  { id: 'page-connectedness', title: 'Cross-Market Connectedness', subtitle: 'Diebold-Yilmaz GFEVD shock spillovers', category: 'Page', href: '/connectedness', badge: 'P7' },
  { id: 'page-events', title: 'Central Bank Events', subtitle: 'FOMC & RBI MPC abnormal reactions & CAR ribbon', category: 'Page', href: '/events', badge: 'P6' },
  { id: 'page-fundamentals', title: 'Energy Fundamentals', subtitle: 'EIA 5-year seasonal inventory bands', category: 'Page', href: '/fundamentals', badge: 'P9' },
  { id: 'page-global', title: 'Global Credit & Reserves', subtitle: 'BIS REER valuation, TIC foreign holdings & ECB/BoJ', category: 'Page', href: '/global', badge: 'P8' },
  { id: 'page-track-record', title: 'Track Record & Validation', subtitle: 'Pre-registered forward evaluation vs base rates', category: 'Page', href: '/track-record', badge: 'P11' },
  { id: 'page-data-health', title: 'Source Data Health', subtitle: 'Pipeline circuit breaker & schema drift status', category: 'Page', href: '/data-health', badge: 'P0' },
  { id: 'page-liquidity', title: 'Net Liquidity Monitor', subtitle: 'Fed balance sheet, TGA & reverse repo audit', category: 'Page', href: '/liquidity' },
  { id: 'page-india', title: 'India Flows', subtitle: 'FII/DII equity & debt flow monitor', category: 'Page', href: '/india' },
  { id: 'page-briefing', title: 'Daily Market Briefing', subtitle: 'Automated statistical market commentary', category: 'Page', href: '/briefing' },
  { id: 'page-methodology', title: 'Mathematical Methodology', subtitle: 'Formulas, statistical specs & estimators', category: 'Page', href: '/methodology' },
  { id: 'page-corrections', title: 'Corrections & Errata', subtitle: 'Transparent pipeline correction audit log', category: 'Page', href: '/corrections' },

  // Countries
  { id: 'cntry-us', title: 'United States', subtitle: 'US Market Profile, S&P 500, 10Y Yields', category: 'Country', href: '/countries/US/' },
  { id: 'cntry-in', title: 'India', subtitle: 'Nifty 50, RBI Policy, Rupee, FII Flows', category: 'Country', href: '/countries/IN/' },
  { id: 'cntry-eu', title: 'Eurozone', subtitle: 'Euro Stoxx 50, ECB Rates, Bund Yields', category: 'Country', href: '/countries/EU/' },
  { id: 'cntry-jp', title: 'Japan', subtitle: 'Nikkei 225, BoJ Yield Curve Control, Yen', category: 'Country', href: '/countries/JP/' },
  { id: 'cntry-gb', title: 'United Kingdom', subtitle: 'FTSE 100, BoE Bank Rate, Gilts', category: 'Country', href: '/countries/GB/' },
  { id: 'cntry-cn', title: 'China', subtitle: 'CSI 300, PBOC Liquidity, Yuan', category: 'Country', href: '/countries/CN/' },
  { id: 'cntry-br', title: 'Brazil', subtitle: 'Ibovespa, Selic Rate, Real', category: 'Country', href: '/countries/BR/' },
  { id: 'cntry-au', title: 'Australia', subtitle: 'ASX 200, RBA Cash Rate, Aussie Dollar', category: 'Country', href: '/countries/AU/' },

  // Sectors
  { id: 'sec-tech', title: 'Information Technology', subtitle: 'Semiconductors, Software & Tech Hardware', category: 'Sector', href: '/sectors/' },
  { id: 'sec-fin', title: 'Financials', subtitle: 'Banks, Insurance & Capital Markets', category: 'Sector', href: '/sectors/' },
  { id: 'sec-eng', title: 'Energy', subtitle: 'Oil & Gas Exploration, Refining & Pipelines', category: 'Sector', href: '/sectors/' },
  { id: 'sec-hc', title: 'Health Care', subtitle: 'Pharma, Biotech & Medical Devices', category: 'Sector', href: '/sectors/' },

  // Drivers
  { id: 'drv-liq', title: 'Fed Net Liquidity', subtitle: 'Fed Assets minus TGA minus Reverse Repo', category: 'Driver', href: '/drivers/' },
  { id: 'drv-real', title: 'US 10Y Real Yield', subtitle: 'TIPS-implied benchmark real borrowing cost', category: 'Driver', href: '/drivers/' },
  { id: 'drv-dxy', title: 'US Dollar Index (DXY)', subtitle: 'Trade-weighted dollar strength vs majors', category: 'Driver', href: '/drivers/' },
  { id: 'drv-oil', title: 'WTI Crude Oil', subtitle: 'Global energy benchmark and inflation driver', category: 'Driver', href: '/drivers/' },

  // Positioning (CFTC Commitments of Traders)
  { id: 'cot-sp500-emini', title: 'S&P 500 E-mini Futures (CFTC)', subtitle: 'Broad US equity speculative crowding (ES)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-nasdaq-100', title: 'Nasdaq 100 Mini Futures (CFTC)', subtitle: 'Tech growth leadership positioning (NQ)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-eur-usd', title: 'Euro FX Futures (CFTC)', subtitle: 'Largest component of US Dollar Index (6E)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-jpy-usd', title: 'Japanese Yen Futures (CFTC)', subtitle: 'Global carry trade barometer (6J)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-gbp-usd', title: 'British Pound Futures (CFTC)', subtitle: 'European cross currency positioning (6B)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-aud-usd', title: 'Australian Dollar Futures (CFTC)', subtitle: 'Commodity currency & China proxy (6A)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-cad-usd', title: 'Canadian Dollar Futures (CFTC)', subtitle: 'North American trade & energy cross (6C)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-chf-usd', title: 'Swiss Franc Futures (CFTC)', subtitle: 'European safe haven positioning (6S)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-us-2y-note', title: '2-Year Treasury Note Futures (CFTC)', subtitle: 'Fed policy rate expectation crowding (ZT)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-us-10y-note', title: '10-Year Treasury Note Futures (CFTC)', subtitle: 'Benchmark long-term duration positioning (ZN)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-us-30y-bond', title: '30-Year Treasury Bond Futures (CFTC)', subtitle: 'Long-duration term premium proxy (ZB)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-wti-crude', title: 'WTI Crude Oil Futures (CFTC)', subtitle: 'Energy demand & geopolitical risk crowding (CL)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-natural-gas', title: 'Natural Gas Futures (CFTC)', subtitle: 'Henry Hub weather-driven storage positioning (NG)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-gold', title: 'Gold Futures (CFTC)', subtitle: 'Monetary debasement & real yield hedge (GC)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-copper', title: 'Copper Futures (CFTC)', subtitle: 'Global industrial growth barometer (HG)', category: 'Positioning', href: '/positioning/' },
  { id: 'cot-silver', title: 'Silver Futures (CFTC)', subtitle: 'High-beta monetary & industrial metal (SI)', category: 'Positioning', href: '/positioning/' },
];

interface Props {
  assets?: StateMapItem[];
}

export function AssetSearchModal({ assets: propAssets }: Props) {
  const router = useRouter();
  const open = useRadarStore((s) => s.searchOpen);
  const setOpen = useRadarStore((s) => s.setSearchOpen);
  const setSelectedAsset = useRadarStore((s) => s.setSelectedAsset);
  const [internalAssets, setInternalAssets] = useState<StateMapItem[]>([]);
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (propAssets && propAssets.length > 0) {
      setInternalAssets(propAssets);
      return;
    }
    loadStateMap()
      .then((res) => {
        if (res.status !== 'error' && res.data.data?.assets) {
          setInternalAssets(res.data.data.assets);
        }
      })
      .catch(() => {});
  }, [propAssets]);

  // Global Cmd+K / Ctrl+K listener
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setOpen(!open);
      } else if (e.key === 'Escape' && open) {
        setOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, setOpen]);

  const activeAssets = propAssets && propAssets.length > 0 ? propAssets : internalAssets;

  // Combine static command items with dynamic assets
  const combinedItems: SearchItem[] = useMemo(() => {
    const assetItems: SearchItem[] = activeAssets.map((a) => ({
      id: `asset-${a.asset}`,
      title: a.asset.toUpperCase(),
      subtitle: `${a.name} • ${groupLabel(a.group)}`,
      category: 'Asset' as const,
      href: `/context/${a.asset.toLowerCase()}/`,
    }));

    return [...STATIC_COMMAND_ITEMS, ...assetItems];
  }, [activeAssets]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return combinedItems.slice(0, 16);
    return combinedItems
      .filter(
        (item) =>
          item.title.toLowerCase().includes(q) ||
          item.subtitle.toLowerCase().includes(q) ||
          item.category.toLowerCase().includes(q)
      )
      .slice(0, 24);
  }, [combinedItems, query]);

  // Keyboard navigation inside modal
  useEffect(() => {
    setSelectedIndex(0);
  }, [matches]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % Math.max(1, matches.length));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 + matches.length) % Math.max(1, matches.length));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (matches[selectedIndex]) {
        executeSelection(matches[selectedIndex]);
      }
    }
  };

  const executeSelection = (item: SearchItem) => {
    if (item.category === 'Asset') {
      setSelectedAsset(item.title);
    }
    setOpen(false);
    setQuery('');
    router.push(item.href);
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-16 px-4 bg-black/70 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label="Universal Command Palette"
    >
      <div className="w-full max-w-xl rounded-xl border border-[#1d2633] bg-[#0c1017] shadow-2xl overflow-hidden flex flex-col">
        {/* Search Input Box */}
        <div className="flex items-center gap-3 px-4 py-3.5 border-b border-[#1d2633]">
          <Search className="w-4 h-4 text-[#38bdf8] shrink-0" />
          <input
            autoFocus
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a command, page, country, driver, or symbol (e.g. Rates, Risk, US, SPX)..."
            className="flex-1 bg-transparent text-sm text-[#e8edf5] placeholder-[#6f7c90] focus:outline-none"
          />
          <kbd className="hidden sm:inline font-mono text-[10px] bg-[#111620] px-1.5 py-0.5 rounded border border-[#1d2633] text-[#6f7c90]">
            ESC
          </kbd>
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label="Close command palette"
            className="p-1 text-[#6f7c90] hover:text-[#e8edf5] cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results List */}
        <div ref={listRef} className="max-h-96 overflow-y-auto divide-y divide-[#151c27] p-1.5">
          {matches.length === 0 ? (
            <div className="p-8 text-center text-xs text-[#6f7c90]">
              No results matching "{query}".
            </div>
          ) : (
            matches.map((item, idx) => {
              const isSelected = idx === selectedIndex;
              const categoryColor =
                item.category === 'Page'
                  ? 'text-[#38bdf8] bg-[#0e2133] border-[#1d3d5e]'
                  : item.category === 'Country'
                  ? 'text-[#4ade80] bg-[#0c2419] border-[#1c4732]'
                  : item.category === 'Sector'
                  ? 'text-[#f59e0b] bg-[#291e0a] border-[#523b12]'
                  : item.category === 'Positioning'
                  ? 'text-[#c084fc] bg-[#241335] border-[#4b2670]'
                  : item.category === 'Driver'
                  ? 'text-[#ec4899] bg-[#290d1c] border-[#57193a]'
                  : 'text-[#94a3b8] bg-[#141a24] border-[#222b3b]';

              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => executeSelection(item)}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-lg text-left transition-colors cursor-pointer ${
                    isSelected ? 'bg-[#16202e] text-[#e8edf5]' : 'hover:bg-[#121823] text-[#a9b4c4]'
                  }`}
                >
                  <div className="flex items-center space-x-3 min-w-0">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${categoryColor} shrink-0`}
                    >
                      {item.category}
                    </span>
                    <div className="min-w-0">
                      <div className="flex items-center space-x-2">
                        <span className="font-semibold text-xs text-[#e8edf5]">{item.title}</span>
                        {item.badge && (
                          <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-[#16202c] border border-[#26354a] text-[#38bdf8]">
                            {item.badge}
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-[#6f7c90] truncate">{item.subtitle}</div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-1.5 shrink-0 ml-3">
                    {isSelected && (
                      <span className="flex items-center space-x-1 text-[10px] font-mono text-[#38bdf8]">
                        <span>Go</span>
                        <CornerDownLeft className="w-3 h-3" />
                      </span>
                    )}
                  </div>
                </button>
              );
            })
          )}
        </div>

        {/* Footer shortcuts */}
        <div className="px-4 py-2.5 border-t border-[#1d2633] bg-[#07090d] flex items-center justify-between text-[11px] text-[#6f7c90] font-mono">
          <div className="flex items-center space-x-3">
            <span>↑↓ Navigate</span>
            <span>↵ Select</span>
            <span>ESC Close</span>
          </div>
          <span>MacroMonitor</span>
        </div>
      </div>
    </div>
  );
}
