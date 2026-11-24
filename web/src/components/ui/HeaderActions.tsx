"use client";

import { Search } from "lucide-react";
import { useRadarStore } from "@/store/useRadarStore";
import { SavedViewsMenu } from "./SavedViewsMenu";

export function HeaderActions() {
  const setSearchOpen = useRadarStore((s) => s.setSearchOpen);

  return (
    <div className="flex items-center space-x-1 sm:space-x-2">
      <button
        type="button"
        onClick={() => setSearchOpen(true)}
        className="flex items-center space-x-1 sm:space-x-1.5 px-2 sm:px-2.5 py-1 rounded-[6px] border border-[#1d2633] bg-[#121823] text-xs text-[#8c9bb0] hover:text-[#e8edf5] cursor-pointer transition-colors"
        aria-label="Search assets (Cmd+K)"
      >
        <Search className="w-3.5 h-3.5 text-[#38bdf8]" />
        <span className="hidden sm:inline">Search</span>
        <kbd className="hidden sm:inline font-mono text-[10px] bg-[#07090d] px-1.5 py-0.5 rounded border border-[#1d2633] text-[#6f7c90]">
          ⌘K
        </kbd>
      </button>
      <SavedViewsMenu />
      <a
        href="https://github.com/Dhruvil-8/MacroMonitor"
        target="_blank"
        rel="noopener noreferrer"
        className="hidden sm:flex items-center space-x-1 px-2.5 py-1 rounded-[6px] border border-[#1d2633] bg-[#121823] text-xs text-[#8c9bb0] hover:text-[#e8edf5] hover:border-[#38bdf8] transition-colors"
        title="GitHub Repository"
        aria-label="GitHub Repository"
      >
        <svg className="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
          <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
        </svg>
        <span className="hidden sm:inline font-mono">GitHub</span>
      </a>
    </div>
  );
}
