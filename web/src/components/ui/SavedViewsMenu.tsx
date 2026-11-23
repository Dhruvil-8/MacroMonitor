"use client";

import { ChangeEvent, useRef, useState } from "react";
import { Bookmark, Check, Copy, Download, Trash2, Upload } from "lucide-react";
import {
  loadSavedViews,
  migrateViewState,
  NamedSavedView,
  persistSavedViews,
} from "@/lib/views/schema";
import { useRadarStore } from "@/store/useRadarStore";

export function SavedViewsMenu() {
  const [open, setOpen] = useState(false);
  const [views, setViews] = useState<NamedSavedView[]>(() => loadSavedViews());
  const [newName, setNewName] = useState("");
  const [copiedUrl, setCopiedUrl] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const exportViewState = useRadarStore((s) => s.exportViewState);
  const applyViewState = useRadarStore((s) => s.applyViewState);
  const buildShareableViewUrl = useRadarStore((s) => s.buildShareableViewUrl);

  const handleSave = () => {
    const trimmed = newName.trim();
    if (!trimmed) return;
    const next: NamedSavedView[] = [
      {
        id: `view-${Date.now()}`,
        name: trimmed,
        createdAt: new Date().toISOString().slice(0, 10),
        state: exportViewState(),
      },
      ...views,
    ].slice(0, 25);
    persistSavedViews(next);
    setViews(next);
    setNewName("");
  };

  const handleDelete = (id: string) => {
    const next = views.filter((v) => v.id !== id);
    persistSavedViews(next);
    setViews(next);
  };

  const handleCopyUrl = async () => {
    const url = buildShareableViewUrl();
    try {
      await navigator.clipboard.writeText(url);
      setCopiedUrl(true);
      setTimeout(() => setCopiedUrl(false), 1800);
    } catch {
      setCopiedUrl(false);
    }
  };

  const handleExportJson = () => {
    const blob = new Blob([JSON.stringify(views, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "global-market-radar-views.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImportJson = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImportError(null);
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result));
        if (!Array.isArray(parsed)) throw new Error("Expected JSON array of saved views.");
        const valid: NamedSavedView[] = [];
        for (const item of parsed) {
          const st = migrateViewState(item?.state);
          if (st && typeof item?.name === "string") {
            valid.push({
              id: String(item.id || `imp-${Date.now()}-${valid.length}`),
              name: item.name,
              createdAt: String(item.createdAt || new Date().toISOString().slice(0, 10)),
              state: st,
            });
          }
        }
        const merged = [...valid, ...views].slice(0, 25);
        persistSavedViews(merged);
        setViews(merged);
      } catch (err) {
        setImportError(err instanceof Error ? err.message : "Invalid view JSON");
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-[6px] border border-[#1d2633] bg-[#121823] text-[#8c9bb0] hover:text-[#e8edf5] cursor-pointer whitespace-nowrap"
      >
        <Bookmark className="w-3.5 h-3.5 text-[#38bdf8]" />
        <span>Saved Views ({views.length})</span>
      </button>

      {open && (
        <div className="absolute right-0 mt-1.5 w-80 sm:w-96 z-40 rounded-[8px] border border-[#1d2633] bg-[#0c1017] shadow-xl p-3.5 flex flex-col gap-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-semibold text-[#e8edf5]">
              Workspace Views (Zero-PII Local)
            </span>
            <button
              type="button"
              onClick={handleCopyUrl}
              className="inline-flex items-center gap-1 text-[11px] font-mono text-[#38bdf8] hover:underline cursor-pointer"
            >
              {copiedUrl ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
              <span>{copiedUrl ? "Copied ?view=" : "Copy ?view= URL"}</span>
            </button>
          </div>

          <div className="flex items-center gap-1.5">
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSave()}
              placeholder="Save current layout as..."
              className="flex-1 px-2.5 py-1.5 text-xs rounded-[6px] border border-[#1d2633] bg-[#07090d] text-[#e8edf5] placeholder-[#6f7c90] focus:outline-none focus:border-[#38bdf8]"
            />
            <button
              type="button"
              onClick={handleSave}
              className="px-2.5 py-1.5 text-xs font-medium rounded-[6px] bg-[#38bdf8] text-[#07090d] font-semibold hover:opacity-90 cursor-pointer shrink-0"
            >
              Save
            </button>
          </div>

          <div className="max-h-48 overflow-y-auto divide-y divide-[#1d2633] border-t border-b border-[#1d2633]">
            {views.length === 0 ? (
              <p className="py-3 text-[11px] text-[#6f7c90]">
                No saved views in browser storage yet. Save any tab, asset, or preset.
              </p>
            ) : (
              views.map((v) => (
                <div key={v.id} className="py-2 flex items-center justify-between gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      applyViewState(v.state);
                      setOpen(false);
                    }}
                    className="text-left min-w-0 flex-1 cursor-pointer group"
                  >
                    <div className="text-xs font-medium text-[#e8edf5] group-hover:text-[#38bdf8] truncate">
                      {v.name}
                    </div>
                    <div className="text-[10px] font-mono text-[#6f7c90] truncate">
                      {v.state.page} · {v.state.selectedAsset} · {v.createdAt}
                    </div>
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDelete(v.id)}
                    aria-label={`Delete view ${v.name}`}
                    className="p-1 text-[#6f7c90] hover:text-[#f87171] cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))
            )}
          </div>

          <div className="flex items-center justify-between gap-2 text-[11px]">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleExportJson}
                className="inline-flex items-center gap-1 text-[#8c9bb0] hover:text-[#e8edf5] cursor-pointer"
              >
                <Download className="w-3 h-3" />
                <span>Export JSON</span>
              </button>
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                className="inline-flex items-center gap-1 text-[#8c9bb0] hover:text-[#e8edf5] cursor-pointer"
              >
                <Upload className="w-3 h-3" />
                <span>Import JSON</span>
              </button>
              <input ref={fileRef} type="file" accept=".json,application/json" onChange={handleImportJson} className="hidden" />
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="text-[#6f7c90] hover:text-[#e8edf5] cursor-pointer"
            >
              Close
            </button>
          </div>
          {importError && <div className="text-[11px] text-[#f87171]">{importError}</div>}
        </div>
      )}
    </div>
  );
}
