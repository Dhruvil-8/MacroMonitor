import { z } from "zod";

export const VIEW_SCHEMA_VERSION = 2;

export const SavedViewStateV2Schema = z.object({
  v: z.literal(2),
  page: z.string(),
  selectedAsset: z.string().default("SPX"),
  selectedMetric: z.string().default("vol20"),
  selectedCorrPair: z.string().default("SPX_US10Y"),
  selectedDivPair: z.string().default("NDX_RSP"),
  rotationUniverse: z.string().default("equity_global"),
  stateGroupFilter: z.string().default("all"),
  macroCountry: z.string().default("US"),
  macroVintageMode: z.enum(["pit", "revised"]).default("pit"),
  analoguePreset: z.string().default("rates_dollar_pressure"),
  attentionEntity: z.string().default("SPX"),
});

export type SavedViewStateV2 = z.infer<typeof SavedViewStateV2Schema>;

export interface NamedSavedView {
  id: string;
  name: string;
  createdAt: string;
  state: SavedViewStateV2;
}

const LegacyViewStateV1Schema = z.object({
  v: z.literal(1),
  page: z.string(),
  asset: z.string().optional(),
  metric: z.string().optional(),
  corr: z.string().optional(),
  div: z.string().optional(),
  universe: z.string().optional(),
  group: z.string().optional(),
});

export function migrateViewState(raw: unknown): SavedViewStateV2 | null {
  const v2 = SavedViewStateV2Schema.safeParse(raw);
  if (v2.success) return v2.data;
  const v1 = LegacyViewStateV1Schema.safeParse(raw);
  if (v1.success) {
    return {
      v: 2,
      page: v1.data.page,
      selectedAsset: v1.data.asset ?? "SPX",
      selectedMetric: v1.data.metric ?? "vol20",
      selectedCorrPair: v1.data.corr ?? "SPX_US10Y",
      selectedDivPair: v1.data.div ?? "NDX_RSP",
      rotationUniverse: v1.data.universe ?? "equity_global",
      stateGroupFilter: v1.data.group ?? "all",
      macroCountry: "US",
      macroVintageMode: "pit",
      analoguePreset: "rates_dollar_pressure",
      attentionEntity: "SPX",
    };
  }
  return null;
}

function toBase64Url(str: string): string {
  const bytes = new TextEncoder().encode(str);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(b64: string): string {
  const norm = b64.replace(/-/g, "+").replace(/_/g, "/");
  const pad = norm.length % 4 === 0 ? "" : "=".repeat(4 - (norm.length % 4));
  const bin = atob(norm + pad);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

export function encodeViewParam(state: SavedViewStateV2): string {
  return toBase64Url(JSON.stringify(state));
}

export function decodeViewParam(param: string): SavedViewStateV2 | null {
  try {
    const json = fromBase64Url(param);
    return migrateViewState(JSON.parse(json));
  } catch {
    return null;
  }
}

const STORAGE_KEY = "gmr_saved_views_v2";

export function loadSavedViews(): NamedSavedView[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    const out: NamedSavedView[] = [];
    for (const item of arr) {
      const migrated = migrateViewState(item?.state);
      if (migrated && typeof item?.id === "string" && typeof item?.name === "string") {
        out.push({
          id: item.id,
          name: item.name,
          createdAt: String(item.createdAt ?? new Date().toISOString()),
          state: migrated,
        });
      }
    }
    return out;
  } catch {
    return [];
  }
}

export function persistSavedViews(views: NamedSavedView[]): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(views));
}
