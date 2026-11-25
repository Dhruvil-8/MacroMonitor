import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';
import { GlobalModals } from '@/components/ui/GlobalModals';
import { HeaderActions } from '@/components/ui/HeaderActions';
import { SyncStatusBadge } from '@/components/ui/SyncStatusBadge';
import { HeaderNav, MobileNav } from '@/components/layout/HeaderNav';

export const metadata: Metadata = {
  title: 'MacroMonitor — Macro Statistics',
  description: 'Open-data macro statistics and analytics. Traceable distributions, liquidity monitors, and cross-asset flow analytics.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-theme="dark">
      <body className="min-h-screen flex flex-col bg-[#07090d] text-[#e8edf5]">
        {/* Top Navigation Bar (Frosted Glass) */}
        <header className="border-b border-white/[0.08] bg-[#0a0d14]/80 backdrop-blur-xl sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-14">
              {/* Brand & Symbol */}
              <div className="flex items-center space-x-3 shrink-0">
                <Link href="/" className="flex items-center space-x-2.5 group">
                  <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-sky-400 to-blue-600 flex items-center justify-center font-semibold text-xs text-white shadow-sm shadow-sky-500/20 group-hover:scale-105 transition-transform">
                    M
                  </div>
                  <span className="font-semibold text-[15px] tracking-tight text-[#f3f6fb]">
                    MacroMonitor
                  </span>
                </Link>
                <span className="text-[11px] font-mono text-[#6f7c90] hidden xl:inline">
                  / Global Macro Intelligence
                </span>
              </div>

              {/* Grouped 5-Dropdown Navigation (P12) */}
              <nav className="hidden lg:flex items-center">
                <HeaderNav />
              </nav>

              {/* Right Side Tools: Search Modal, Market Clock, Saved Views, Data Sync */}
              <div className="flex items-center space-x-1 sm:space-x-2 text-xs font-mono shrink-0">
                <Link
                  href="/market-clock"
                  className="hidden md:flex items-center space-x-1.5 px-2.5 py-1 rounded-[6px] border border-[#1d2633] bg-[#121823] text-xs font-mono text-[#8c9bb0] hover:text-[#e8edf5] hover:border-[#38bdf8] transition-colors"
                  title="Global Financial Market Clock & Sessions"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span>Market Clock</span>
                </Link>
                <HeaderActions />
                <SyncStatusBadge />
              </div>
            </div>

            {/* Mobile Nav Accordion */}
            <div className="lg:hidden border-t border-[#1d2633]">
              <MobileNav />
            </div>
          </div>
        </header>

        {/* Content Body */}
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>

        {/* Global Modals for Cmd+K search & Share card */}
        <GlobalModals />

        {/* Footer with Non-negotiable Compliance Disclaimer */}
        <footer className="border-t border-[#1d2633] bg-[#0c1017] py-6 text-xs text-[#6f7c90]">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-3">
            <div className="flex flex-col sm:flex-row justify-between items-center gap-2">
              <span className="font-mono">MacroMonitor &bull; Macro Statistics</span>
              <div className="flex flex-wrap items-center justify-center sm:justify-end gap-x-4 gap-y-2">
                <Link href="/analogues" className="hover:text-[#a9b4c4] transition">
                  Historical Analogues
                </Link>
                <Link href="/positioning" className="hover:text-[#a9b4c4] transition">
                  COT Positioning
                </Link>
                <Link href="/data-health" className="hover:text-[#a9b4c4] transition">
                  Data Health
                </Link>
                <Link href="/methodology" className="hover:text-[#a9b4c4] transition">
                  Methodology &amp; Specs
                </Link>
                <a
                  href="https://github.com/Dhruvil-8/MacroMonitor"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-[#a9b4c4] text-[#38bdf8] transition flex items-center gap-1 font-mono"
                >
                  GitHub &rarr;
                </a>
              </div>
            </div>
            <p className="text-[11px] text-[#6f7c90] leading-relaxed border-t border-[#1d2633]/60 pt-3">
              <strong>Disclaimer:</strong> For information and research only. Descriptive statistics, not investment advice or a recommendation. Past patterns do not predict future results. Every number is traceable back to its underlying sources (FRED, NSE, US Treasury, NY Fed, CFTC).
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
