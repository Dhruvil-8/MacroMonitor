// web/src/lib/data/client.ts
// Typed data loading client with runtime Zod validation and staleness checks (Section 11.3)

import {
  CompactSeriesEnvelope,
  CompactSeriesEnvelopeSchema,
  ContextEnvelope,
  ContextEnvelopeSchema,
  CorrelationsEnvelope,
  CorrelationsEnvelopeSchema,
  DivergencesEnvelope,
  DivergencesEnvelopeSchema,
  ManifestEnvelope,
  ManifestEnvelopeSchema,
  ObservationsEnvelope,
  ObservationsEnvelopeSchema,
  PulseEnvelope,
  PulseEnvelopeSchema,
  RotationEnvelope,
  RotationEnvelopeSchema,
  StateMapEnvelope,
  StateMapEnvelopeSchema,
  VolLandscapeEnvelope,
  VolLandscapeEnvelopeSchema,
  DecompositionEnvelope,
  DecompositionEnvelopeSchema,
  CountriesEnvelope,
  CountriesEnvelopeSchema,
  SectorsEnvelope,
  SectorsEnvelopeSchema,
  DriversEnvelope,
  DriversEnvelopeSchema,
  RiskEnvelope,
  RiskEnvelopeSchema,
  PositioningEnvelope,
  PositioningEnvelopeSchema,
  StressEnvelope,
  StressEnvelopeSchema,
  AnaloguesPresetsEnvelope,
  AnaloguesPresetsEnvelopeSchema,
  AnaloguePresetDetailEnvelope,
  AnaloguePresetDetailEnvelopeSchema,
  AnaloguesEvaluationEnvelope,
  AnaloguesEvaluationEnvelopeSchema,
  SourceStatusEnvelope,
  SourceStatusEnvelopeSchema,
  BenchmarkEnvelope,
  BenchmarkEnvelopeSchema,
  ImpliedVolEnvelope,
  ImpliedVolEnvelopeSchema,
  YieldCurveEnvelope,
  YieldCurveEnvelopeSchema,
  EventStudyEnvelope,
  EventStudyEnvelopeSchema,
  ConnectednessEnvelope,
  ConnectednessEnvelopeSchema,
  FundamentalsEnvelope,
  FundamentalsEnvelopeSchema,
  GlobalSlowEnvelope,
  GlobalSlowEnvelopeSchema,
  TailsRegimesEnvelope,
  TailsRegimesEnvelopeSchema,
  TrackRecordEnvelope,
  TrackRecordEnvelopeSchema,
} from "./schemas";
import { z } from "zod";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";
const DATA_BASE_URL = process.env.NEXT_PUBLIC_DATA_BASE_URL || `${basePath}/data/v1`;
const STALE_AFTER_HOURS = 36;

export type LoadResult<T> =
  | { status: "ok"; data: T; asOf: string }
  | { status: "stale"; data: T; asOf: string; ageHours: number }
  | { status: "error"; error: string };

let manifestCache: ManifestEnvelope | null = null;

export async function fetchManifest(): Promise<LoadResult<ManifestEnvelope>> {
  if (manifestCache) {
    return { status: "ok", data: manifestCache, asOf: manifestCache.as_of };
  }

  try {
    const res = await fetch(`${DATA_BASE_URL}/manifest.json`);
    if (!res.ok) {
      return { status: "error", error: `Failed to fetch manifest: HTTP ${res.status}` };
    }
    const json = await res.json();
    const parsed = ManifestEnvelopeSchema.safeParse(json);
    if (!parsed.success) {
      return { status: "error", error: `Manifest schema validation failed: ${parsed.error.message}` };
    }
    manifestCache = parsed.data;
    return checkStaleness(parsed.data, parsed.data.as_of);
  } catch (err) {
    return { status: "error", error: err instanceof Error ? err.message : String(err) };
  }
}

async function loadDataset<T>(
  endpoint: string,
  schema: z.ZodType<T, any, any>
): Promise<LoadResult<T>> {
  try {
    const res = await fetch(`${DATA_BASE_URL}${endpoint}`);
    if (!res.ok) {
      return { status: "error", error: `HTTP ${res.status} fetching ${endpoint}` };
    }
    const json = await res.json();
    const parsed = schema.safeParse(json);
    if (!parsed.success) {
      return { status: "error", error: `Validation failed for ${endpoint}: ${parsed.error.message}` };
    }

    const asOf = (json as { as_of?: string }).as_of || new Date().toISOString().split("T")[0];
    return checkStaleness(parsed.data, asOf);
  } catch (err) {
    return { status: "error", error: err instanceof Error ? err.message : String(err) };
  }
}

function checkStaleness<T>(data: T, asOf: string): LoadResult<T> {
  const asOfDate = new Date(asOf);
  const now = new Date();
  const diffHours = (now.getTime() - asOfDate.getTime()) / (1000 * 60 * 60);

  if (diffHours > STALE_AFTER_HOURS) {
    return { status: "stale", data, asOf, ageHours: Math.round(diffHours) };
  }
  return { status: "ok", data, asOf };
}

export async function loadPulse(): Promise<LoadResult<PulseEnvelope>> {
  return loadDataset("/pulse.json", PulseEnvelopeSchema);
}

export async function loadStateMap(): Promise<LoadResult<StateMapEnvelope>> {
  return loadDataset("/state_map.json", StateMapEnvelopeSchema);
}

export async function loadVolLandscape(): Promise<LoadResult<VolLandscapeEnvelope>> {
  return loadDataset("/vol_landscape.json", VolLandscapeEnvelopeSchema);
}

export async function loadCompactSeries(seriesId: string): Promise<LoadResult<CompactSeriesEnvelope>> {
  return loadDataset(`/series/${seriesId}.json`, CompactSeriesEnvelopeSchema);
}

export async function loadObservations(): Promise<LoadResult<ObservationsEnvelope>> {
  return loadDataset("/observations.json", ObservationsEnvelopeSchema);
}

export async function loadDivergences(): Promise<LoadResult<DivergencesEnvelope>> {
  return loadDataset("/divergences.json", DivergencesEnvelopeSchema);
}

export async function loadCorrelations(): Promise<LoadResult<CorrelationsEnvelope>> {
  return loadDataset("/correlations.json", CorrelationsEnvelopeSchema);
}

export async function loadRotation(): Promise<LoadResult<RotationEnvelope>> {
  return loadDataset("/rotation_global.json", RotationEnvelopeSchema);
}

export async function loadContext(seriesId: string): Promise<LoadResult<ContextEnvelope>> {
  return loadDataset(`/context/${seriesId}.json`, ContextEnvelopeSchema);
}

export async function loadDecomposition(targetId: string = "featured"): Promise<LoadResult<DecompositionEnvelope>> {
  return loadDataset(`/decomposition/${targetId}.json`, DecompositionEnvelopeSchema);
}

export async function loadCountries(): Promise<LoadResult<CountriesEnvelope>> {
  return loadDataset("/countries.json", CountriesEnvelopeSchema);
}

export async function loadSectors(): Promise<LoadResult<SectorsEnvelope>> {
  return loadDataset("/sectors.json", SectorsEnvelopeSchema);
}

export async function loadDrivers(): Promise<LoadResult<DriversEnvelope>> {
  return loadDataset("/drivers.json", DriversEnvelopeSchema);
}

export async function loadRisk(): Promise<LoadResult<RiskEnvelope>> {
  return loadDataset("/risk.json", RiskEnvelopeSchema);
}

export async function loadPositioning(): Promise<LoadResult<PositioningEnvelope>> {
  return loadDataset("/positioning.json", PositioningEnvelopeSchema);
}

export async function loadStress(): Promise<LoadResult<StressEnvelope>> {
  return loadDataset("/stress.json", StressEnvelopeSchema);
}

export async function loadAnaloguesIndex(): Promise<LoadResult<AnaloguesPresetsEnvelope>> {
  return loadDataset("/analogues/presets.json", AnaloguesPresetsEnvelopeSchema);
}

export async function loadAnaloguePreset(presetId: string): Promise<LoadResult<AnaloguePresetDetailEnvelope>> {
  return loadDataset(`/analogues/${presetId}.json`, AnaloguePresetDetailEnvelopeSchema);
}

export async function loadAnaloguesEvaluation(): Promise<LoadResult<AnaloguesEvaluationEnvelope>> {
  return loadDataset("/analogues/evaluation.json", AnaloguesEvaluationEnvelopeSchema);
}

export async function loadSourceStatus(): Promise<LoadResult<SourceStatusEnvelope>> {
  return loadDataset("/source_status.json", SourceStatusEnvelopeSchema);
}

export async function loadBenchmarks(): Promise<LoadResult<BenchmarkEnvelope>> {
  return loadDataset("/benchmarks.json", BenchmarkEnvelopeSchema);
}

export async function loadImpliedVol(): Promise<LoadResult<ImpliedVolEnvelope>> {
  return loadDataset("/implied_vol.json", ImpliedVolEnvelopeSchema);
}

export async function loadYieldCurve(): Promise<LoadResult<YieldCurveEnvelope>> {
  return loadDataset("/yield_curve.json", YieldCurveEnvelopeSchema);
}

export async function loadEvents(eventType: string = "fomc"): Promise<LoadResult<EventStudyEnvelope>> {
  return loadDataset(`/events_${eventType}.json`, EventStudyEnvelopeSchema);
}

export async function loadConnectedness(): Promise<LoadResult<ConnectednessEnvelope>> {
  return loadDataset("/connectedness.json", ConnectednessEnvelopeSchema);
}

export async function loadInventories(): Promise<LoadResult<FundamentalsEnvelope>> {
  return loadDataset("/inventories.json", FundamentalsEnvelopeSchema);
}

export async function loadGlobalSlow(): Promise<LoadResult<GlobalSlowEnvelope>> {
  return loadDataset("/global_slow.json", GlobalSlowEnvelopeSchema);
}

export async function loadTailsRegimes(): Promise<LoadResult<TailsRegimesEnvelope>> {
  return loadDataset("/tails_regimes.json", TailsRegimesEnvelopeSchema);
}

export async function loadTrackRecord(): Promise<LoadResult<TrackRecordEnvelope>> {
  return loadDataset("/track_record.json", TrackRecordEnvelopeSchema);
}






