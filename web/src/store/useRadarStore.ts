import { create } from "zustand";
import { decodeViewParam, encodeViewParam, SavedViewStateV2 } from "../lib/views/schema";

export type ContextMetricKey = "vol20" | "ret_20d" | "drawdown_252d" | "trend_tstat";
export type RotationUniverseKey = "equity_global" | "equity_us_sector" | "equity_india";

export interface AuditPayload {
  title: string;
  formula: string;
  parameters: Record<string, string | number>;
  inputs: { id: string; lastUpdate: string | null }[];
  percentileSummary: string;
  n: number | string;
  nEff: number | string;
  asOf: string;
  codeVersion: string;
  paramsHash: string;
  specAnchor: string;
}

export interface ShareTargetPayload {
  chartKey: string;
  title: string;
  subtitle: string;
  headlineMetric: string;
  percentileSummary: string;
  asOf: string;
  n: number | string;
  nEff: number | string;
  queryParams?: Record<string, string>;
}

interface RadarState {
  page: string;
  theme: "dark" | "light";
  selectedAsset: string;
  selectedMetric: ContextMetricKey;
  selectedCorrPair: string;
  selectedDivPair: string;
  rotationUniverse: RotationUniverseKey;
  stateGroupFilter: string;
  macroCountry: string;
  macroVintageMode: "pit" | "revised";
  analoguePreset: string;
  attentionEntity: string;
  auditPayload: AuditPayload | null;
  shareTarget: ShareTargetPayload | null;
  searchOpen: boolean;
  activeSpecId: string;
  setPage: (page: string) => void;
  toggleTheme: () => void;
  setSelectedAsset: (assetId: string) => void;
  setSelectedMetric: (metric: ContextMetricKey) => void;
  setSelectedCorrPair: (pairId: string) => void;
  setSelectedDivPair: (pairId: string) => void;
  setRotationUniverse: (u: RotationUniverseKey) => void;
  setStateGroupFilter: (g: string) => void;
  setMacroCountry: (country: string) => void;
  setMacroVintageMode: (mode: "pit" | "revised") => void;
  setAnaloguePreset: (presetId: string) => void;
  setAttentionEntity: (entityId: string) => void;
  openAudit: (payload: AuditPayload) => void;
  closeAudit: () => void;
  openShare: (payload: ShareTargetPayload) => void;
  closeShare: () => void;
  setSearchOpen: (open: boolean) => void;
  openSpecInMethodology: (specId: string) => void;
  exportViewState: () => SavedViewStateV2;
  applyViewState: (view: SavedViewStateV2) => void;
  buildShareableViewUrl: () => string;
}

export const useRadarStore = create<RadarState>((set, get) => ({
  page: "today",
  theme: "dark",
  selectedAsset: "SPX",
  selectedMetric: "vol20",
  selectedCorrPair: "SPX_US10Y",
  selectedDivPair: "NDX_RSP",
  rotationUniverse: "equity_global",
  stateGroupFilter: "all",
  macroCountry: "US",
  macroVintageMode: "pit",
  analoguePreset: "rates_dollar_pressure",
  attentionEntity: "SPX",
  auditPayload: null,
  shareTarget: null,
  searchOpen: false,
  activeSpecId: "returns_volatility",

  setPage: (page) => set({ page }),
  toggleTheme: () => {
    const next = get().theme === "dark" ? "light" : "dark";
    if (typeof document !== "undefined") {
      document.documentElement.classList.toggle("dark", next === "dark");
    }
    set({ theme: next });
  },
  setSelectedAsset: (selectedAsset) => set({ selectedAsset }),
  setSelectedMetric: (selectedMetric) => set({ selectedMetric }),
  setSelectedCorrPair: (selectedCorrPair) => set({ selectedCorrPair }),
  setSelectedDivPair: (selectedDivPair) => set({ selectedDivPair }),
  setRotationUniverse: (rotationUniverse) => set({ rotationUniverse }),
  setStateGroupFilter: (stateGroupFilter) => set({ stateGroupFilter }),
  setMacroCountry: (macroCountry) => set({ macroCountry }),
  setMacroVintageMode: (macroVintageMode) => set({ macroVintageMode }),
  setAnaloguePreset: (analoguePreset) => set({ analoguePreset }),
  setAttentionEntity: (attentionEntity) => set({ attentionEntity }),
  openAudit: (auditPayload) => set({ auditPayload }),
  closeAudit: () => set({ auditPayload: null }),
  openShare: (shareTarget) => set({ shareTarget }),
  closeShare: () => set({ shareTarget: null }),
  setSearchOpen: (searchOpen) => set({ searchOpen }),
  openSpecInMethodology: (specId) => set({ auditPayload: null, activeSpecId: specId, page: "methodology" }),
  exportViewState: () => {
    const s = get();
    return {
      v: 2,
      page: s.page,
      selectedAsset: s.selectedAsset,
      selectedMetric: s.selectedMetric,
      selectedCorrPair: s.selectedCorrPair,
      selectedDivPair: s.selectedDivPair,
      rotationUniverse: s.rotationUniverse,
      stateGroupFilter: s.stateGroupFilter,
      macroCountry: s.macroCountry,
      macroVintageMode: s.macroVintageMode,
      analoguePreset: s.analoguePreset,
      attentionEntity: s.attentionEntity,
    };
  },
  applyViewState: (view) => {
    set({
      page: view.page,
      selectedAsset: view.selectedAsset,
      selectedMetric: (view.selectedMetric as ContextMetricKey) || "vol20",
      selectedCorrPair: view.selectedCorrPair,
      selectedDivPair: view.selectedDivPair,
      rotationUniverse: (view.rotationUniverse as RotationUniverseKey) || "equity_global",
      stateGroupFilter: view.stateGroupFilter,
      macroCountry: view.macroCountry,
      macroVintageMode: view.macroVintageMode,
      analoguePreset: view.analoguePreset,
      attentionEntity: view.attentionEntity,
    });
  },
  buildShareableViewUrl: () => {
    if (typeof window === "undefined") return "";
    const token = encodeViewParam(get().exportViewState());
    return `${window.location.origin}${window.location.pathname}?view=${token}`;
  },
}));
