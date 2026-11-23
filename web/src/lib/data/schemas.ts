// web/src/lib/data/schemas.ts
// Zod mirror of pipeline/src/radar/export/schemas.py (Section 10)
import { z } from "zod";

export const BaseEnvelopeSchema = <T extends z.ZodTypeAny>(dataSchema: T) =>
  z.object({
    schema_version: z.string(),
    as_of: z.string(),
    code_version: z.string(),
    params_hash: z.string(),
    data: dataSchema,
  });

// 1. Manifest
export const ManifestFileEntrySchema = z.object({
  path: z.string(),
  schema_version: z.string(),
  size_bytes: z.number(),
  as_of: z.string(),
});

export const ManifestEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  manifest: z.object({
    datasets: z.array(ManifestFileEntrySchema),
  }),
});

export type ManifestEnvelope = z.infer<typeof ManifestEnvelopeSchema>;

// 2. Compact Series
export const CompactSeriesDataSchema = z.object({
  id: z.string(),
  name: z.string(),
  unit: z.string(),
  frequency: z.string(),
  d: z.array(z.string()),
  v: z.array(z.number().nullable()),
});

export const CompactSeriesEnvelopeSchema = BaseEnvelopeSchema(CompactSeriesDataSchema);
export type CompactSeriesData = z.infer<typeof CompactSeriesDataSchema>;
export type CompactSeriesEnvelope = z.infer<typeof CompactSeriesEnvelopeSchema>;

// 3. Market State Map
export const StateMapItemSchema = z.object({
  asset: z.string(),
  name: z.string(),
  group: z.string(),
  region: z.string(),
  trend_t: z.number(),
  vol_percentile: z.number(),
  quadrant: z.enum(["Orderly trend", "Violent trend", "Quiet range", "Choppy"]),
  momentum_20d: z.number(),
});

export type StateMapItem = z.infer<typeof StateMapItemSchema>;

export const StateMapDataSchema = z.object({
  assets: z.array(StateMapItemSchema),
});

export const StateMapEnvelopeSchema = BaseEnvelopeSchema(StateMapDataSchema);
export type StateMapEnvelope = z.infer<typeof StateMapEnvelopeSchema>;

// 4. Volatility Landscape
export const VolLandscapeItemSchema = z.object({
  asset: z.string(),
  name: z.string(),
  group: z.string(),
  region: z.string(),
  vol10: z.number(),
  vol20: z.number(),
  vol60: z.number(),
  vol_trend: z.number(),
  vol20_percentile: z.number(),
  n: z.number(),
  n_eff: z.number(),
});

export type VolLandscapeItem = z.infer<typeof VolLandscapeItemSchema>;

export const VolLandscapeDataSchema = z.object({
  landscape: z.array(VolLandscapeItemSchema),
});

export const VolLandscapeEnvelopeSchema = BaseEnvelopeSchema(VolLandscapeDataSchema);
export type VolLandscapeEnvelope = z.infer<typeof VolLandscapeEnvelopeSchema>;

// 5. Pulse Summary
export const PulseStatSchema = z.object({
  id: z.string(),
  name: z.string(),
  value: z.number(),
  change_1d: z.number().nullable(),
  unit: z.string(),
  percentile: z.number().nullable(),
  sparkline: z.array(z.number()),
});

export type PulseStat = z.infer<typeof PulseStatSchema>;

export const PulseDataSchema = z.object({
  headline: z.string(),
  risk_regime: z.string(),
  stats: z.array(PulseStatSchema),
  observations_count: z.number(),
});

export const PulseEnvelopeSchema = BaseEnvelopeSchema(PulseDataSchema);
export type PulseEnvelope = z.infer<typeof PulseEnvelopeSchema>;

// 6. Observations (Change Feed)
export const ObservationItemSchema = z.object({
  id: z.string(),
  type: z.string(),
  asset: z.string(),
  severity: z.enum(["low", "medium", "high"]),
  headline: z.string(),
  body: z.string(),
  metrics: z.record(z.any()),
  window: z.string(),
  inputs: z.array(z.string()),
  as_of: z.string(),
  chart_ref: z.string(),
  rule_id: z.string(),
  code_version: z.string(),
});

export type ObservationItem = z.infer<typeof ObservationItemSchema>;

export const ObservationsDataSchema = z.object({
  observations: z.array(ObservationItemSchema),
});

export const ObservationsEnvelopeSchema = BaseEnvelopeSchema(ObservationsDataSchema);
export type ObservationsEnvelope = z.infer<typeof ObservationsEnvelopeSchema>;

// 7. Divergences
export const CusumShiftSchema = z.object({
  date: z.string(),
  direction: z.string(),
  value: z.number(),
});
export type CusumShift = z.infer<typeof CusumShiftSchema>;

export const CusumDataSchema = z.object({
  dates: z.array(z.string()),
  cusum_pos: z.array(z.number()),
  cusum_neg: z.array(z.number()),
  shifts: z.array(CusumShiftSchema),
  threshold: z.number(),
});
export type CusumData = z.infer<typeof CusumDataSchema>;

export const DivergenceItemSchema = z.object({
  id: z.string(),
  asset_a: z.string(),
  asset_b: z.string(),
  rationale: z.string(),
  spread_z: z.number(),
  percentile: z.number(),
  consecutive_days: z.number(),
  active: z.boolean(),
  history_dates: z.array(z.string()),
  history_spread: z.array(z.number()),
  cusum: CusumDataSchema.optional(),
});

export type DivergenceItem = z.infer<typeof DivergenceItemSchema>;

export const DivergencesDataSchema = z.object({
  tested_pairs_count: z.number(),
  pairs: z.array(DivergenceItemSchema),
});

export const DivergencesEnvelopeSchema = BaseEnvelopeSchema(DivergencesDataSchema);
export type DivergencesEnvelope = z.infer<typeof DivergencesEnvelopeSchema>;

// 8. Correlations & Heatmap Trio
export const CorrelationPairItemSchema = z.object({
  id: z.string(),
  asset_a: z.string(),
  asset_b: z.string(),
  rho60: z.number(),
  rho250: z.number(),
  shift: z.number(),
  percentile: z.number(),
  history_dates: z.array(z.string()),
  history_rho60: z.array(z.number()),
});

export type CorrelationPairItem = z.infer<typeof CorrelationPairItemSchema>;

export const CorrelationsDataSchema = z.object({
  benchmark_assets: z.array(z.string()),
  heatmap_now: z.record(z.record(z.number())),
  heatmap_3m: z.record(z.record(z.number())),
  heatmap_1y: z.record(z.record(z.number())),
  pairs: z.array(CorrelationPairItemSchema),
});

export const CorrelationsEnvelopeSchema = BaseEnvelopeSchema(CorrelationsDataSchema);
export type CorrelationsEnvelope = z.infer<typeof CorrelationsEnvelopeSchema>;

// 9. Relative Rotation Map
export const RotationItemSchema = z.object({
  asset: z.string(),
  name: z.string(),
  group: z.string(),
  rs_ratio: z.number(),
  rs_mom: z.number(),
  phase: z.enum(["Leading", "Fading", "Lagging", "Recovering"]),
  tail_dates: z.array(z.string()),
  tail_x: z.array(z.number()),
  tail_y: z.array(z.number()),
});

export type RotationItem = z.infer<typeof RotationItemSchema>;

export const RotationDataSchema = z.object({
  benchmark: z.string(),
  assets: z.array(RotationItemSchema),
});

export const RotationEnvelopeSchema = BaseEnvelopeSchema(RotationDataSchema);
export type RotationEnvelope = z.infer<typeof RotationEnvelopeSchema>;

// 10. Historical Context
export const AnalogItemSchema = z.object({
  date: z.string(),
  value: z.number(),
  distance: z.number(),
  fwd_5d: z.number().nullable().optional(),
  fwd_20d: z.number().nullable().optional(),
});

export type AnalogItem = z.infer<typeof AnalogItemSchema>;

export const ContextDataSchema = z.object({
  id: z.string(),
  name: z.string(),
  group: z.string(),
  region: z.string(),
  unit: z.string(),
  current_value: z.number(),
  percentile: z.number(),
  z_score: z.number(),
  n: z.number(),
  n_eff: z.number(),
  histogram_bins: z.array(z.number()),
  histogram_counts: z.array(z.number()),
  kde_grid: z.array(z.number()),
  kde_density: z.array(z.number()),
  normal_density: z.array(z.number()),
  qq_theoretical: z.array(z.number()),
  qq_empirical: z.array(z.number()),
  ecdf_x: z.array(z.number()),
  ecdf_y: z.array(z.number()),
  calendar_heatmap: z.array(z.object({
    year: z.number(),
    month: z.number(),
    return: z.number(),
  })),
  analogs: z.array(AnalogItemSchema),
  peer_rank: z.number(),
  peer_count: z.number(),
  description: z.string(),
  source_note: z.string(),
});

export type ContextData = z.infer<typeof ContextDataSchema>;

export const ContextEnvelopeSchema = BaseEnvelopeSchema(ContextDataSchema);
export type ContextEnvelope = z.infer<typeof ContextEnvelopeSchema>;


// 11. Global Liquidity
export const LiquidityAuditSchema = z.object({
  formula: z.string(),
  parameters: z.record(z.string(), z.any()),
  series_used: z.array(z.string()),
  n: z.number(),
  n_eff: z.number(),
});

export const LiquidityDataSchema = z.object({
  current_net_liquidity: z.number(),
  change_4w: z.number(),
  change_4w_pct: z.number(),
  change_13w: z.number(),
  change_13w_pct: z.number(),
  stablecoins_total_bn: z.number(),
  stablecoins_30d_pct: z.number(),
  dates: z.array(z.string()),
  net_liquidity: z.array(z.number()),
  fed_assets: z.array(z.number()),
  tga: z.array(z.number()),
  rrp: z.array(z.number()),
  spx_overlay: z.array(z.number()),
  rolling_corr_60d: z.array(z.number()),
  audit: LiquidityAuditSchema,
});

export type LiquidityData = z.infer<typeof LiquidityDataSchema>;

export const LiquidityEnvelopeSchema = BaseEnvelopeSchema(LiquidityDataSchema);
export type LiquidityEnvelope = z.infer<typeof LiquidityEnvelopeSchema>;


// 12. India Flows & Positioning
export const OiBuildupItemSchema = z.object({
  symbol: z.string(),
  price_change_pct: z.number(),
  oi_change_pct: z.number(),
  buildup: z.enum(["Long Buildup", "Short Buildup", "Short Covering", "Long Unwinding", "Neutral"]),
  pcr: z.number(),
});

export type OiBuildupItem = z.infer<typeof OiBuildupItemSchema>;

export const IndiaFlowsAuditSchema = z.object({
  formula: z.string(),
  parameters: z.record(z.string(), z.any()),
  series_used: z.array(z.string()),
  n: z.number(),
  n_eff: z.number(),
});

export const IndiaFlowsDataSchema = z.object({
  india_vix: z.number(),
  india_vix_percentile: z.number(),
  pcr: z.number(),
  fii_net_1d: z.number(),
  dii_net_1d: z.number(),
  fii_net_5d: z.number(),
  dii_net_5d: z.number(),
  fii_net_20d: z.number(),
  dii_net_20d: z.number(),
  fii_20d_percentile: z.number(),
  dii_20d_percentile: z.number(),
  delivery_pct_index: z.number(),
  delivery_pct_percentile: z.number(),
  flow_dates: z.array(z.string()),
  fii_daily: z.array(z.number()),
  dii_daily: z.array(z.number()),
  fii_cum_20d: z.array(z.number()),
  dii_cum_20d: z.array(z.number()),
  oi_buildup: z.array(OiBuildupItemSchema),
  audit: IndiaFlowsAuditSchema,
});

export type IndiaFlowsData = z.infer<typeof IndiaFlowsDataSchema>;

export const IndiaFlowsEnvelopeSchema = BaseEnvelopeSchema(IndiaFlowsDataSchema);
export type IndiaFlowsEnvelope = z.infer<typeof IndiaFlowsEnvelopeSchema>;

// 14. Daily Briefing
export const BriefingSectionSchema = z.object({
  title: z.string(),
  content: z.string(),
});

export type BriefingSection = z.infer<typeof BriefingSectionSchema>;

export const BriefingDataSchema = z.object({
  as_of: z.string(),
  headline: z.string(),
  executive_summary: z.string(),
  stress_score: z.number(),
  stress_regime: z.string(),
  sections: z.array(BriefingSectionSchema),
});

export type BriefingData = z.infer<typeof BriefingDataSchema>;

export const BriefingEnvelopeSchema = BaseEnvelopeSchema(BriefingDataSchema);
export type BriefingEnvelope = z.infer<typeof BriefingEnvelopeSchema>;

// 15. Move Decomposition
export const DecompositionHistoryPointSchema = z.object({
  date: z.string(),
  total_return: z.number(),
  contributions: z.record(z.string(), z.number()),
  residual: z.number(),
});

export type DecompositionHistoryPoint = z.infer<typeof DecompositionHistoryPointSchema>;

export const DecompositionDataSchema = z.object({
  target: z.string(),
  target_id: z.string(),
  horizon_days: z.number(),
  lens: z.enum(["local", "usd"]),
  total_return: z.number(),
  contributions: z.record(z.string(), z.number()),
  intervals: z.record(z.string(), z.tuple([z.number(), z.number()])),
  residual: z.number(),
  betas: z.record(z.string(), z.number()),
  r2_in_sample: z.number(),
  r2_out_sample: z.number(),
  mostly_unexplained: z.boolean(),
  regularized: z.boolean(),
  n_obs: z.number(),
  window_dates: z.object({
    prior_start: z.string(),
    prior_end: z.string(),
    horizon_start: z.string(),
    horizon_end: z.string(),
  }),
  narrative: z.string(),
  history: z.array(DecompositionHistoryPointSchema).optional(),
});

export type DecompositionData = z.infer<typeof DecompositionDataSchema>;

export const DecompositionEnvelopeSchema = BaseEnvelopeSchema(DecompositionDataSchema);
export type DecompositionEnvelope = z.infer<typeof DecompositionEnvelopeSchema>;

// 16. Country Layer (M5)
export const CountryMatrixRowSchema = z.object({
  code: z.string(),
  name: z.string(),
  index_id: z.string(),
  currency: z.string(),
  returns_local: z.record(z.string(), z.number()),
  returns_usd: z.record(z.string(), z.number()),
  returns_fx: z.record(z.string(), z.number()),
});

export type CountryMatrixRow = z.infer<typeof CountryMatrixRowSchema>;

export const CountriesDataSchema = z.object({
  countries: z.array(CountryMatrixRowSchema),
  breadth_200d: z.number(),
  avg_corr_60d: z.number(),
  avg_corr_250d: z.number(),
  tracked_count: z.number(),
});

export type CountriesData = z.infer<typeof CountriesDataSchema>;

export const CountriesEnvelopeSchema = BaseEnvelopeSchema(CountriesDataSchema);
export type CountriesEnvelope = z.infer<typeof CountriesEnvelopeSchema>;

// 17. Sector Layer (M7)
export const SectorRotationPointSchema = z.object({
  date: z.string(),
  rs_ratio: z.number(),
  rs_mom: z.number(),
});

export type SectorRotationPoint = z.infer<typeof SectorRotationPointSchema>;

export const SectorItemSchema = z.object({
  id: z.string(),
  name: z.string(),
  weight: z.number(),
  returns: z.record(z.string(), z.number()),
  percentiles: z.record(z.string(), z.number()),
  contributions: z.record(z.string(), z.number()),
  rs_ratio: z.number(),
  rs_mom: z.number(),
  phase: z.string(),
  trail: z.array(SectorRotationPointSchema),
});

export type SectorItem = z.infer<typeof SectorItemSchema>;

export const SectorRegionDataSchema = z.object({
  region_id: z.string(),
  country_name: z.string(),
  benchmark_name: z.string(),
  index_id: z.string(),
  as_of: z.string(),
  weights_source: z.string(),
  weights_source_url: z.string(),
  index_returns: z.record(z.string(), z.number()),
  sector_breadth: z.record(z.string(), z.number()),
  residuals: z.record(z.string(), z.number()),
  sectors: z.array(SectorItemSchema),
});

export type SectorRegionData = z.infer<typeof SectorRegionDataSchema>;

export const SectorsDataSchema = z.object({
  regions: z.record(z.string(), SectorRegionDataSchema),
});

export type SectorsData = z.infer<typeof SectorsDataSchema>;

export const SectorsEnvelopeSchema = BaseEnvelopeSchema(SectorsDataSchema);
export type SectorsEnvelope = z.infer<typeof SectorsEnvelopeSchema>;

// 18. Driver Sensitivity Layer (M8)
export const DriverSensitivityItemSchema = z.object({
  driver_id: z.string(),
  beta: z.number(),
  ci_lower: z.number(),
  ci_upper: z.number(),
  p_value: z.number(),
  t_stat: z.number(),
  stability: z.number(),
  prior: z.string(),
  prior_aligned: z.boolean(),
  significant: z.boolean(),
  history: z.array(z.object({
    date: z.string(),
    beta: z.number(),
    ci_lower: z.number(),
    ci_upper: z.number(),
  })),
});

export type DriverSensitivityItem = z.infer<typeof DriverSensitivityItemSchema>;

export const DriverTargetMatrixRowSchema = z.object({
  target_id: z.string(),
  target_name: z.string(),
  target_code: z.string(),
  window: z.number(),
  r2: z.number(),
  regularized: z.boolean(),
  n_obs: z.number(),
  drivers: z.record(z.string(), DriverSensitivityItemSchema),
});

export type DriverTargetMatrixRow = z.infer<typeof DriverTargetMatrixRowSchema>;

export const DriverScatterDataSchema = z.object({
  scatter: z.array(z.array(z.number())),
  line: z.array(z.array(z.number())),
  band_upper: z.array(z.array(z.number())),
  band_lower: z.array(z.array(z.number())),
  beta: z.number(),
  r2: z.number(),
  t_stat: z.number(),
  n_obs: z.number(),
});

export type DriverScatterData = z.infer<typeof DriverScatterDataSchema>;

export const DriversDataSchema = z.object({
  matrix_250d: z.array(DriverTargetMatrixRowSchema),
  matrix_120d: z.array(DriverTargetMatrixRowSchema),
  pairs_tested: z.number(),
  scatter_examples: z.record(z.string(), DriverScatterDataSchema),
});

export type DriversData = z.infer<typeof DriversDataSchema>;

export const DriversEnvelopeSchema = BaseEnvelopeSchema(DriversDataSchema);
export type DriversEnvelope = z.infer<typeof DriversEnvelopeSchema>;

// 19. Risk Lab Layer (M9)
export const VolatilityConePointSchema = z.object({
  window: z.number(),
  min: z.number(),
  p10: z.number(),
  p25: z.number(),
  p50: z.number(),
  p75: z.number(),
  p90: z.number(),
  max: z.number(),
  current: z.number(),
  percentile: z.number(),
});
export type VolatilityConePoint = z.infer<typeof VolatilityConePointSchema>;

export const RealizedEwmaPointSchema = z.object({
  date: z.string(),
  realized_vol: z.number(),
  ewma_vol: z.number(),
});
export type RealizedEwmaPoint = z.infer<typeof RealizedEwmaPointSchema>;

export const DistributionShapeDataSchema = z.object({
  skewness: z.number(),
  excess_kurtosis: z.number(),
  jb_stat: z.number(),
  jb_pvalue: z.number(),
  qq_theoretical: z.array(z.number()),
  qq_empirical: z.array(z.number()),
  mean: z.number(),
  std: z.number(),
});
export type DistributionShapeData = z.infer<typeof DistributionShapeDataSchema>;

export const TailRiskMetricSchema = z.object({
  confidence: z.number(),
  hist_var: z.number(),
  hist_es: z.number(),
  gauss_var: z.number(),
  gauss_es: z.number(),
  tail_ratio: z.number(),
});
export type TailRiskMetric = z.infer<typeof TailRiskMetricSchema>;

export const AutocorrelationRiskDataSchema = z.object({
  lags: z.array(z.number()),
  acf_returns: z.array(z.number()),
  acf_squared: z.array(z.number()),
  band: z.number(),
  n_eff: z.number(),
});
export type AutocorrelationRiskData = z.infer<typeof AutocorrelationRiskDataSchema>;

export const DrawdownEpisodeItemSchema = z.object({
  peak_date: z.string(),
  trough_date: z.string(),
  recovery_date: z.string().nullable().optional(),
  depth: z.number(),
  duration_days: z.number(),
  recovery_days: z.number().nullable().optional(),
  status: z.string(),
});
export type DrawdownEpisodeItem = z.infer<typeof DrawdownEpisodeItemSchema>;

export const DrawdownSeriesPointSchema = z.object({
  date: z.string(),
  drawdown: z.number(),
});
export type DrawdownSeriesPoint = z.infer<typeof DrawdownSeriesPointSchema>;

export const PcaFactorDataSchema = z.object({
  scree: z.array(z.number()),
  loadings_pc1: z.record(z.string(), z.number()),
  loadings_pc2: z.record(z.string(), z.number()),
  pc1_history: z.array(z.object({
    date: z.string(),
    pc1_share: z.number(),
  })),
});
export type PcaFactorData = z.infer<typeof PcaFactorDataSchema>;

export const DownsideCorrelationItemSchema = z.object({
  target_id: z.string(),
  target_name: z.string(),
  all_day_corr: z.number(),
  downside_corr: z.number(),
  stress_premium: z.number(),
  stress_threshold: z.number(),
  n_all: z.number(),
  n_stress: z.number(),
});
export type DownsideCorrelationItem = z.infer<typeof DownsideCorrelationItemSchema>;

export const AssetRiskProfileSchema = z.object({
  asset_id: z.string(),
  asset_name: z.string(),
  asset_group: z.string(),
  volatility_cone: z.array(VolatilityConePointSchema),
  vol_history: z.array(RealizedEwmaPointSchema),
  distribution: DistributionShapeDataSchema,
  tail_risk: z.array(TailRiskMetricSchema),
  autocorrelation: AutocorrelationRiskDataSchema,
  drawdown_series: z.array(DrawdownSeriesPointSchema),
  drawdown_episodes: z.array(DrawdownEpisodeItemSchema),
});
export type AssetRiskProfile = z.infer<typeof AssetRiskProfileSchema>;

export const RiskDataSchema = z.object({
  selected_asset: z.string(),
  assets_available: z.array(z.object({
    id: z.string(),
    name: z.string(),
  })),
  profiles: z.record(z.string(), AssetRiskProfileSchema),
  pca: PcaFactorDataSchema,
  downside_correlations: z.array(DownsideCorrelationItemSchema),
  market_synchronization_latest: z.number(),
});
export type RiskData = z.infer<typeof RiskDataSchema>;

export const RiskEnvelopeSchema = BaseEnvelopeSchema(RiskDataSchema);
export type RiskEnvelope = z.infer<typeof RiskEnvelopeSchema>;

// 20. CFTC Positioning (M14)
export const CotPointSchema = z.object({
  date: z.string(),
  open_interest: z.number(),
  noncomm_long: z.number(),
  noncomm_short: z.number(),
  net_speculative: z.number(),
  net_pct_oi: z.number().nullable().optional(),
  percentile: z.number().nullable().optional(),
  z_score: z.number().nullable().optional(),
});
export type CotPoint = z.infer<typeof CotPointSchema>;

export const CotContractSchema = z.object({
  id: z.string(),
  name: z.string(),
  cftc_code: z.string(),
  asset_class: z.string().optional(),
  category: z.string().optional(),
  n: z.number(),
  n_eff: z.number(),
  latest_date: z.string(),
  open_interest: z.number(),
  net_speculative: z.number(),
  net_pct_oi: z.number().nullable().optional(),
  chg_1w_pct_oi: z.number().nullable().optional(),
  chg_4w_pct_oi: z.number().nullable().optional(),
  chg_13w_pct_oi: z.number().nullable().optional(),
  percentile: z.number().nullable().optional(),
  percentile_4w_ago: z.number().nullable().optional(),
  percentile_13w_ago: z.number().nullable().optional(),
  z_score: z.number().nullable().optional(),
  is_extreme: z.boolean().optional(),
  extreme_type: z.string().optional(),
  lag_label: z.string().optional(),
  series: z.array(CotPointSchema),
});
export type CotContract = z.infer<typeof CotContractSchema>;

export const PositioningEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string().optional(),
  status: z.string().optional(),
  error: z.string().nullable().optional(),
  contracts: z.array(CotContractSchema),
});
export type PositioningEnvelope = z.infer<typeof PositioningEnvelopeSchema>;

// 21. Multi-Asset Stress Dashboard
export const CrisisEpisodeSchema = z.object({
  id: z.string().optional(),
  name: z.string().optional(),
  label: z.string().optional(),
  date: z.string().optional(),
  start: z.string().optional(),
  end: z.string().optional(),
  peak_date: z.string().optional(),
  peak_composite: z.number().optional(),
  stress: z.number().optional(),
});
export type CrisisEpisode = z.infer<typeof CrisisEpisodeSchema>;

export const StressEpisodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  start: z.string(),
  end: z.string(),
  peak_date: z.string(),
  peak_composite: z.number(),
});
export type StressEpisode = z.infer<typeof StressEpisodeSchema>;

export const StressSeriesPointSchema = z.object({
  date: z.string(),
  composite: z.number(),
  components: z.record(z.string(), z.number().nullable().optional()),
});
export type StressSeriesPoint = z.infer<typeof StressSeriesPointSchema>;

export const StressEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string().optional(),
  latest: z.object({
    date: z.string(),
    composite: z.number(),
    tercile: z.string(),
    components: z.record(z.string(), z.number().nullable().optional()),
    missing_components: z.array(z.string()).optional(),
  }),
  component_keys: z.array(z.string()),
  episodes: z.array(StressEpisodeSchema),
  series: z.array(StressSeriesPointSchema),
});
export type StressEnvelope = z.infer<typeof StressEnvelopeSchema>;

// 22. Historical Analogue Engine
export const AnalogueMatchItemSchema = z.object({
  rank: z.number(),
  date: z.string(),
  distance: z.number(),
  similarity_score: z.number(),
  features: z.array(z.number()),
  spx_fwd_3m: z.number().nullable().optional(),
  us10y_fwd_3m: z.number().nullable().optional(),
  spx_mdd_3m: z.number().nullable().optional(),
});
export type AnalogueMatchItem = z.infer<typeof AnalogueMatchItemSchema>;

export const AnalogueOutcomeFanSchema = z.object({
  asset: z.string(),
  unit: z.string(),
  horizon: z.string(),
  horizon_days: z.number(),
  n_matches: z.number(),
  n_base: z.number(),
  cond_p10: z.number(),
  cond_p25: z.number(),
  cond_p50: z.number(),
  cond_p75: z.number(),
  cond_p90: z.number(),
  base_p25: z.number(),
  base_p50: z.number(),
  base_p75: z.number(),
  diff_median: z.number(),
  diff_ci90_low: z.number(),
  diff_ci90_high: z.number(),
  cond_mdd_p50: z.number().nullable().optional(),
  cond_mdd_p90: z.number().nullable().optional(),
});
export type AnalogueOutcomeFan = z.infer<typeof AnalogueOutcomeFanSchema>;

export const AnalogueEvalMetricsSchema = z.object({
  n_eval: z.number(),
  spearman_rho: z.number(),
  iqr_coverage: z.number(),
  mae_cond: z.number().nullable(),
  mae_base: z.number().nullable(),
  mae_skill: z.number(),
  verdict_class: z.string(),
  verdict_text: z.string(),
  subperiods: z.array(z.object({
    label: z.string(),
    n_eval: z.number(),
    spearman_rho: z.number(),
    iqr_coverage: z.number(),
    mae_skill: z.number(),
  })).optional(),
});
export type AnalogueEvalMetrics = z.infer<typeof AnalogueEvalMetricsSchema>;

export const AnaloguePresetSummarySchema = z.object({
  id: z.string(),
  name: z.string(),
  rationale: z.string(),
  query_date: z.string(),
  n_features: z.number(),
  n_indep: z.number(),
  top_match_date: z.string().nullable().optional(),
  top_match_distance: z.number().nullable().optional(),
  verdict_class: z.string(),
  mae_skill: z.number(),
});
export type AnaloguePresetSummary = z.infer<typeof AnaloguePresetSummarySchema>;

export const AnaloguesPresetsEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  status: z.string().optional(),
  presets: z.array(AnaloguePresetSummarySchema),
});
export type AnaloguesPresetsEnvelope = z.infer<typeof AnaloguesPresetsEnvelopeSchema>;
export type AnaloguesPresetsData = AnaloguesPresetsEnvelope;

export const AnaloguePresetDetailEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  status: z.string().optional(),
  params_hash: z.string().optional(),
  preset_id: z.string(),
  name: z.string(),
  rationale: z.string(),
  query_date: z.string(),
  n_features: z.number(),
  n_indep: z.number(),
  shrinkage_delta: z.number(),
  decluster_gap_days: z.number(),
  feature_labels: z.array(z.string()),
  today_vector: z.array(z.number()),
  matches: z.array(AnalogueMatchItemSchema),
  outcome_fans: z.array(AnalogueOutcomeFanSchema),
  evaluation: AnalogueEvalMetricsSchema.extend({
    target_asset: z.string().optional(),
  }),
});
export type AnaloguePresetDetailEnvelope = z.infer<typeof AnaloguePresetDetailEnvelopeSchema>;
export type AnaloguePresetDetailData = AnaloguePresetDetailEnvelope;

export const AnaloguesEvaluationEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  status: z.string().optional(),
  skill_threshold: z.number(),
  min_eval_points: z.number(),
  presets: z.array(
    AnalogueEvalMetricsSchema.extend({
      preset_id: z.string(),
      name: z.string(),
      target_asset: z.string(),
    })
  ),
});
export type AnaloguesEvaluationEnvelope = z.infer<typeof AnaloguesEvaluationEnvelopeSchema>;
export type AnaloguesEvaluationData = AnaloguesEvaluationEnvelope;

// --- P0: Source Status & Resilience Schemas ---
export const SourceStatusItemSchema = z.object({
  id: z.string(),
  name: z.string(),
  provider: z.string(),
  documentation_url: z.string(),
  terms_url: z.string(),
  rate_limit: z.string(),
  schedule: z.string(),
  status: z.enum(["healthy", "degraded", "stale", "rate_limited"]).or(z.string()),
  last_success: z.string(),
  last_verified: z.string(),
  tracked_series_count: z.number(),
  stale_series_count: z.number(),
  circuit_state: z.string(),
  owner_notes: z.string(),
});
export type SourceStatusItem = z.infer<typeof SourceStatusItemSchema>;

export const SourceStatusDataSchema = z.object({
  sources: z.array(SourceStatusItemSchema),
  overall_status: z.string(),
  total_tracked_series: z.number(),
  total_healthy_series: z.number(),
});
export type SourceStatusData = z.infer<typeof SourceStatusDataSchema>;

export const SourceStatusEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: SourceStatusDataSchema,
});
export type SourceStatusEnvelope = z.infer<typeof SourceStatusEnvelopeSchema>;

// --- P1: Benchmarks and Episodes Schemas ---
export const CrossCorrelationItemSchema = z.object({
  lag_weeks: z.number(),
  correlation: z.number(),
  interpretation: z.string(),
});
export type CrossCorrelationItem = z.infer<typeof CrossCorrelationItemSchema>;

export const BenchmarkAgreementItemSchema = z.object({
  benchmark_name: z.string(),
  n_obs: z.number(),
  effective_n: z.number(),
  pearson_corr: z.number(),
  spearman_rank_corr: z.number(),
  best_lead_lag: z.number(),
  cross_correlation: z.array(CrossCorrelationItemSchema).default([]),
});
export type BenchmarkAgreementItem = z.infer<typeof BenchmarkAgreementItemSchema>;

export const HistoricalEpisodeItemSchema = z.object({
  id: z.string(),
  name: z.string(),
  start_date: z.string(),
  end_date: z.string(),
  category: z.string(),
  peak_stress_score: z.number(),
  peak_stress_date: z.string(),
  mean_stress_score: z.number(),
  lead_lag_to_official_peak_days: z.number(),
  elevated_ratio: z.number(),
  description: z.string().default(""),
});
export type HistoricalEpisodeItem = z.infer<typeof HistoricalEpisodeItemSchema>;

export const BenchmarkDataSchema = z.object({
  stress_cross_checks: z.array(BenchmarkAgreementItemSchema),
  episodes: z.array(HistoricalEpisodeItemSchema),
  policy_uncertainty_level: z.number(),
  policy_uncertainty_percentile: z.number(),
});
export type BenchmarkData = z.infer<typeof BenchmarkDataSchema>;

export const BenchmarkEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: BenchmarkDataSchema,
});
export type BenchmarkEnvelope = z.infer<typeof BenchmarkEnvelopeSchema>;

// --- P3: Options-Implied Layer Schemas ---
export const TermStructurePointSchema = z.object({
  tenor: z.string(),
  days: z.number(),
  current: z.number(),
  one_month_ago: z.number(),
  one_year_median: z.number(),
  p10: z.number(),
  p90: z.number(),
});
export type TermStructurePoint = z.infer<typeof TermStructurePointSchema>;

export const FearTailItemSchema = z.object({
  id: z.string(),
  name: z.string(),
  current: z.number(),
  percentile: z.number(),
  z_score: z.number(),
  chg_1w: z.number(),
  chg_1m: z.number(),
  sparkline: z.array(z.number()).default([]),
});
export type FearTailItem = z.infer<typeof FearTailItemSchema>;

export const ImpliedVsRealizedItemSchema = z.object({
  market: z.string(),
  implied_vol: z.number(),
  realized_vol_21d: z.number(),
  gap: z.number(),
  gap_percentile: z.number(),
  vrp_proxy: z.number().nullable().optional(),
  vrp_label: z.string().default("Trailing proxy"),
});
export type ImpliedVsRealizedItem = z.infer<typeof ImpliedVsRealizedItemSchema>;

export const SlopeHistoryItemSchema = z.object({
  date: z.string(),
  vix: z.number(),
  vix3m: z.number(),
  slope_3m: z.number(),
  slope_9d: z.number(),
  state: z.string(),
  percentile: z.number(),
});
export type SlopeHistoryItem = z.infer<typeof SlopeHistoryItemSchema>;

export const ImpliedVolDataSchema = z.object({
  as_of: z.string(),
  current_state: z.string(),
  slope_3m: z.number(),
  slope_9d: z.number(),
  slope_percentile: z.number(),
  curve: z.array(TermStructurePointSchema),
  fear_tail_indices: z.array(FearTailItemSchema),
  implied_vs_realized: z.array(ImpliedVsRealizedItemSchema),
  slope_history: z.array(SlopeHistoryItemSchema),
  interpretation: z.string(),
});
export type ImpliedVolData = z.infer<typeof ImpliedVolDataSchema>;

export const ImpliedVolEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: ImpliedVolDataSchema,
});
export type ImpliedVolEnvelope = z.infer<typeof ImpliedVolEnvelopeSchema>;

// --- P4: Yield Curve Lab Schemas ---
export const YieldCurvePointItemSchema = z.object({
  maturity: z.string(),
  months: z.number(),
  yield_pct: z.number(),
  fitted_pct: z.number(),
});
export type YieldCurvePointItem = z.infer<typeof YieldCurvePointItemSchema>;

export const SmoothPointSchema = z.object({
  months: z.number(),
  yield_pct: z.number(),
});
export type SmoothPoint = z.infer<typeof SmoothPointSchema>;

export const CurveSnapshotItemSchema = z.object({
  label: z.string(),
  date: z.string(),
  points: z.array(YieldCurvePointItemSchema),
  smooth_fitted: z.array(SmoothPointSchema).default([]),
});
export type CurveSnapshotItem = z.infer<typeof CurveSnapshotItemSchema>;

export const NelsonSiegelFactorsItemSchema = z.object({
  level: z.number(),
  slope: z.number(),
  curvature: z.number(),
  rmse: z.number(),
  level_percentile: z.number(),
  slope_percentile: z.number(),
  curvature_percentile: z.number(),
});
export type NelsonSiegelFactorsItem = z.infer<typeof NelsonSiegelFactorsItemSchema>;

export const YieldSpreadItemSchema = z.object({
  name: z.string(),
  current: z.number(),
  chg_1m: z.number(),
  percentile: z.number(),
  inverted: z.boolean(),
});
export type YieldSpreadItem = z.infer<typeof YieldSpreadItemSchema>;

export const TenYearAnatomyItemSchema = z.object({
  date: z.string(),
  nominal_10y: z.number(),
  real_10y: z.number(),
  breakeven_10y: z.number(),
  term_premium: z.number(),
});
export type TenYearAnatomyItem = z.infer<typeof TenYearAnatomyItemSchema>;

export const HeatmapRowItemSchema = z.object({
  date: z.string(),
  yields: z.record(z.string(), z.number()),
});
export type HeatmapRowItem = z.infer<typeof HeatmapRowItemSchema>;

export const YieldCurveDataSchema = z.object({
  as_of: z.string(),
  factors: NelsonSiegelFactorsItemSchema,
  spreads: z.array(YieldSpreadItemSchema),
  curves: z.array(CurveSnapshotItemSchema),
  anatomy: z.array(TenYearAnatomyItemSchema),
  heatmap: z.array(HeatmapRowItemSchema),
  maturities: z.array(z.string()),
  interpretation: z.string(),
});
export type YieldCurveData = z.infer<typeof YieldCurveDataSchema>;

export const YieldCurveEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: YieldCurveDataSchema,
});
export type YieldCurveEnvelope = z.infer<typeof YieldCurveEnvelopeSchema>;

// --- P6: Event Studies Schemas ---
export const ReactionRibbonPointSchema = z.object({
  day: z.number(),
  mean_car: z.number(),
  median_car: z.number(),
  ci_lower_90: z.number(),
  ci_lower_50: z.number(),
  ci_upper_50: z.number(),
  ci_upper_90: z.number(),
});
export type ReactionRibbonPoint = z.infer<typeof ReactionRibbonPointSchema>;

export const DecisionStatsItemSchema = z.object({
  count: z.number(),
  mean_day_0_ar: z.number(),
  median_day_0_ar: z.number(),
  pct_positive: z.number(),
});
export type DecisionStatsItem = z.infer<typeof DecisionStatsItemSchema>;

export const EventDetailItemSchema = z.object({
  id: z.string(),
  date: z.string(),
  decision: z.string(),
  rate_before: z.number(),
  rate_after: z.number(),
  day_0_ar: z.number(),
  car_10d: z.number(),
  description: z.string().default(""),
});
export type EventDetailItem = z.infer<typeof EventDetailItemSchema>;

export const EventStudyDataSchema = z.object({
  event_type: z.string(),
  target_name: z.string(),
  n_events: z.number(),
  ribbon: z.array(ReactionRibbonPointSchema),
  by_decision: z.record(z.string(), DecisionStatsItemSchema),
  magnitude_ratio: z.number(),
  events: z.array(EventDetailItemSchema),
  interpretation: z.string(),
});
export type EventStudyData = z.infer<typeof EventStudyDataSchema>;

export const EventStudyEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: EventStudyDataSchema,
});
export type EventStudyEnvelope = z.infer<typeof EventStudyEnvelopeSchema>;

// --- P7: Connectedness Schemas ---
export const ConnectednessCountryItemSchema = z.object({
  id: z.string(),
  name: z.string(),
  own: z.number(),
  from_others: z.number(),
  to_others: z.number(),
  net: z.number(),
  role: z.string(),
});
export type ConnectednessCountryItem = z.infer<typeof ConnectednessCountryItemSchema>;

export const TciPointSchema = z.object({
  date: z.string(),
  tci: z.number(),
});
export type TciPoint = z.infer<typeof TciPointSchema>;

export const ConnectednessDataSchema = z.object({
  as_of: z.string(),
  horizon_weeks: z.number(),
  tci: z.number(),
  countries: z.array(ConnectednessCountryItemSchema),
  matrix: z.array(z.record(z.string(), z.any())),
  labels: z.array(z.string()),
  tci_history: z.array(TciPointSchema),
  interpretation: z.string(),
});
export type ConnectednessData = z.infer<typeof ConnectednessDataSchema>;

export const ConnectednessEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: ConnectednessDataSchema,
});
export type ConnectednessEnvelope = z.infer<typeof ConnectednessEnvelopeSchema>;

// --- P9: Fundamentals Schemas ---
export const SeasonalWeekPointSchema = z.object({
  week: z.number(),
  min_5y: z.number(),
  max_5y: z.number(),
  mean_5y: z.number(),
  current_year: z.number().nullable().optional(),
  prev_year: z.number().nullable().optional(),
});
export type SeasonalWeekPoint = z.infer<typeof SeasonalWeekPointSchema>;

export const InventoryItemSchema = z.object({
  id: z.string(),
  name: z.string(),
  unit: z.string(),
  as_of: z.string(),
  current_level: z.number(),
  mean_5y: z.number(),
  min_5y: z.number(),
  max_5y: z.number(),
  deviation: z.number(),
  deviation_pct: z.number(),
  deviation_percentile: z.number(),
  build_draw_1w: z.number(),
  band_years_label: z.string(),
  weeks: z.array(SeasonalWeekPointSchema),
});
export type InventoryItem = z.infer<typeof InventoryItemSchema>;

export const FundamentalsDataSchema = z.object({
  as_of: z.string(),
  inventories: z.array(InventoryItemSchema),
  lag_label: z.string(),
  interpretation: z.string(),
});
export type FundamentalsData = z.infer<typeof FundamentalsDataSchema>;

export const FundamentalsEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: FundamentalsDataSchema,
});
export type FundamentalsEnvelope = z.infer<typeof FundamentalsEnvelopeSchema>;

// --- P8: Slow Global Layer Schemas ---
export const ReerValuationItemSchema = z.object({
  id: z.string(),
  name: z.string(),
  current_index: z.number(),
  percentile: z.number(),
  chg_1y: z.number(),
  valuation: z.string(),
  as_of: z.string(),
});
export type ReerValuationItem = z.infer<typeof ReerValuationItemSchema>;

export const ForeignTreasuryHoldingItemSchema = z.object({
  country: z.string(),
  holdings_bn: z.number(),
  share_pct: z.number(),
  chg_1y_bn: z.number(),
  as_of: z.string(),
});
export type ForeignTreasuryHoldingItem = z.infer<typeof ForeignTreasuryHoldingItemSchema>;

export const CentralBankAssetItemSchema = z.object({
  central_bank: z.string(),
  assets_trillion_usd: z.number(),
  chg_1y_pct: z.number(),
  as_of: z.string(),
});
export type CentralBankAssetItem = z.infer<typeof CentralBankAssetItemSchema>;

export const IndiaMacroProfileSchema = z.object({
  fx_reserves_bn_usd: z.number(),
  fx_reserves_as_of: z.string(),
  import_cover_months: z.number(),
  fpi_equity_12m_bn: z.number(),
  fpi_debt_12m_bn: z.number(),
  fpi_net_total_12m_bn: z.number(),
  rupee_reer_percentile: z.number(),
});
export type IndiaMacroProfile = z.infer<typeof IndiaMacroProfileSchema>;

export const GlobalSlowDataSchema = z.object({
  as_of: z.string(),
  reer: z.array(ReerValuationItemSchema),
  foreign_treasury_holdings: z.array(ForeignTreasuryHoldingItemSchema),
  central_bank_assets: z.array(CentralBankAssetItemSchema),
  india_macro: IndiaMacroProfileSchema,
  lag_label: z.string(),
  interpretation: z.string(),
});
export type GlobalSlowData = z.infer<typeof GlobalSlowDataSchema>;

export const GlobalSlowEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: GlobalSlowDataSchema,
});
export type GlobalSlowEnvelope = z.infer<typeof GlobalSlowEnvelopeSchema>;

// --- P10: Advanced Tails and Regimes Schemas ---
export const MeanExcessPointSchema = z.object({
  threshold: z.number(),
  quantile: z.number(),
  mean_excess: z.number(),
  count: z.number(),
});
export type MeanExcessPoint = z.infer<typeof MeanExcessPointSchema>;

export const EvtGpdDataSchema = z.object({
  shape_xi: z.number(),
  scale_beta: z.number(),
  threshold_u: z.number(),
  tail_fatness: z.string(),
  var_99_pct: z.number(),
  es_99_pct: z.number(),
  var_995_pct: z.number(),
  es_995_pct: z.number(),
  gaussian_var_99: z.number(),
  gaussian_var_995: z.number(),
  empirical_var_99: z.number(),
  empirical_var_995: z.number(),
  excess_ratio_over_gaussian: z.number(),
  mean_excess_curve: z.array(MeanExcessPointSchema).default([]),
});
export type EvtGpdData = z.infer<typeof EvtGpdDataSchema>;

export const RegimeRibbonPointSchema = z.object({
  date: z.string(),
  filtered_prob_stress: z.number(),
  smoothed_prob_stress: z.number(),
  regime: z.string(),
});
export type RegimeRibbonPoint = z.infer<typeof RegimeRibbonPointSchema>;

export const MarkovRegimesDataSchema = z.object({
  current_regime: z.string(),
  current_prob_stress: z.number(),
  transition_matrix: z.record(z.string(), z.number()),
  expected_duration_days: z.record(z.string(), z.number()),
  ribbon: z.array(RegimeRibbonPointSchema).default([]),
});
export type MarkovRegimesData = z.infer<typeof MarkovRegimesDataSchema>;

export const TailsRegimesDataSchema = z.object({
  as_of: z.string(),
  target_name: z.string(),
  evt: EvtGpdDataSchema,
  regimes: MarkovRegimesDataSchema,
  interpretation: z.string(),
});
export type TailsRegimesData = z.infer<typeof TailsRegimesDataSchema>;

export const TailsRegimesEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: TailsRegimesDataSchema,
});
export type TailsRegimesEnvelope = z.infer<typeof TailsRegimesEnvelopeSchema>;

// --- P11: Track Record & Validation Schemas ---
export const TrackRecordRuleItemSchema = z.object({
  rule_id: z.string(),
  rule_name: z.string(),
  hypothesis: z.string(),
  horizon_days: z.number(),
  n_alerts: z.number(),
  n_episodes: z.number(),
  hit_rate: z.number(),
  base_rate: z.number(),
  edge_pct: z.number(),
  ci_lower: z.number(),
  ci_upper: z.number(),
  median_lead_days: z.number(),
  status: z.string(),
  sample_period: z.string(),
});
export type TrackRecordRuleItem = z.infer<typeof TrackRecordRuleItemSchema>;

export const CalibrationBandItemSchema = z.object({
  severity: z.string(),
  n_episodes: z.number(),
  observed_rate: z.number(),
  expected_rate: z.number(),
});
export type CalibrationBandItem = z.infer<typeof CalibrationBandItemSchema>;

export const RuleChangelogItemSchema = z.object({
  date: z.string(),
  version: z.string(),
  rule_id: z.string(),
  change_summary: z.string(),
  commit_hash: z.string(),
});
export type RuleChangelogItem = z.infer<typeof RuleChangelogItemSchema>;

export const TrackRecordDataSchema = z.object({
  as_of: z.string(),
  overall_hit_rate: z.number(),
  overall_base_rate: z.number(),
  total_episodes: z.number(),
  rules: z.array(TrackRecordRuleItemSchema).default([]),
  calibration: z.array(CalibrationBandItemSchema).default([]),
  changelog: z.array(RuleChangelogItemSchema).default([]),
  statement: z.string(),
});
export type TrackRecordData = z.infer<typeof TrackRecordDataSchema>;

export const TrackRecordEnvelopeSchema = z.object({
  schema_version: z.string(),
  as_of: z.string(),
  code_version: z.string(),
  params_hash: z.string(),
  data: TrackRecordDataSchema,
});
export type TrackRecordEnvelope = z.infer<typeof TrackRecordEnvelopeSchema>;














