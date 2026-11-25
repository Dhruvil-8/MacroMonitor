# pipeline/src/radar/export/schemas.py
"""Data contract definitions using Pydantic v2.

Implements Section 10 and docs/spec/s04-export-contracts.md.
Single source of truth for exported JSON data contracts.
"""

from datetime import datetime, timezone
from typing import Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class BaseEnvelope(BaseModel, Generic[T]):
    schema_version: str = Field(default="1.0", description="Contract schema version")
    as_of: str = Field(description="ISO date string YYYY-MM-DD")
    fetched_at: Optional[str] = Field(default_factory=_utc_now_iso, description="UTC timestamp of pipeline fetch/export")
    code_version: str = Field(default="v1.0.0", description="Git commit hash or release tag")
    params_hash: str = Field(description="SHA256 hash of params.yaml")
    data: T


# 1. Manifest
class ManifestFileEntry(BaseModel):
    path: str
    schema_version: str = "1.0"
    size_bytes: int
    as_of: str


class ManifestData(BaseModel):
    datasets: List[ManifestFileEntry]


class ManifestEnvelope(BaseModel):
    schema_version: str = "1.0"
    as_of: str
    fetched_at: Optional[str] = Field(default_factory=_utc_now_iso, description="UTC timestamp of pipeline fetch/export")
    code_version: str = "v1.0.0"
    params_hash: str
    manifest: ManifestData


# 2. Compact Series
class CompactSeriesData(BaseModel):
    id: str
    name: str
    unit: str
    frequency: str
    d: List[str]  # Dates YYYY-MM-DD
    v: List[Optional[float]]  # Values


class CompactSeriesEnvelope(BaseEnvelope[CompactSeriesData]):
    pass


# 3. Market State Map
class StateMapItem(BaseModel):
    asset: str
    name: str
    group: str
    region: str
    trend_t: float
    vol_percentile: float
    quadrant: str  # 'Orderly trend' | 'Violent trend' | 'Quiet range' | 'Choppy'
    momentum_20d: float


class StateMapData(BaseModel):
    assets: List[StateMapItem]


class StateMapEnvelope(BaseEnvelope[StateMapData]):
    pass


# 4. Volatility Landscape
class VolLandscapeItem(BaseModel):
    asset: str
    name: str
    group: str
    region: str
    vol10: float
    vol20: float
    vol60: float
    vol_trend: float
    vol20_percentile: float
    n: int
    n_eff: float


class VolLandscapeData(BaseModel):
    landscape: List[VolLandscapeItem]


class VolLandscapeEnvelope(BaseEnvelope[VolLandscapeData]):
    pass


# 5. Pulse Summary
class PulseStat(BaseModel):
    id: str
    name: str
    value: float
    change_1d: Optional[float]
    unit: str
    percentile: Optional[float]
    sparkline: List[float]


class PulseData(BaseModel):
    headline: str
    risk_regime: str
    stats: List[PulseStat]
    observations_count: int


class PulseEnvelope(BaseEnvelope[PulseData]):
    pass


# 6. Observations (Change Feed)
class ObservationItem(BaseModel):
    id: str
    type: str  # 'volatility_shift' | 'divergence' | 'unusual_return' | ...
    asset: str
    severity: str  # 'low' | 'medium' | 'high'
    headline: str
    body: str
    metrics: dict[str, Any]
    window: str
    inputs: List[str]
    as_of: str
    chart_ref: str
    rule_id: str
    code_version: str


class ObservationsData(BaseModel):
    observations: List[ObservationItem]


class ObservationsEnvelope(BaseEnvelope[ObservationsData]):
    pass


# 7. Divergences
class CusumShift(BaseModel):
    date: str
    direction: str  # 'upward' | 'downward'
    value: float


class CusumData(BaseModel):
    dates: List[str]
    cusum_pos: List[float]
    cusum_neg: List[float]
    shifts: List[CusumShift]
    threshold: float


class DivergenceItem(BaseModel):
    id: str
    asset_a: str
    asset_b: str
    rationale: str
    spread_z: float
    percentile: float
    consecutive_days: int
    active: bool
    history_dates: List[str]
    history_spread: List[float]
    cusum: Optional[CusumData] = None


class DivergencesData(BaseModel):
    tested_pairs_count: int
    pairs: List[DivergenceItem]


class DivergencesEnvelope(BaseEnvelope[DivergencesData]):
    pass


# 8. Correlations & Heatmap Trio
class CorrelationPairItem(BaseModel):
    id: str
    asset_a: str
    asset_b: str
    rho60: float
    rho250: float
    shift: float
    percentile: float
    history_dates: List[str]
    history_rho60: List[float]


class CorrelationsData(BaseModel):
    benchmark_assets: List[str]
    heatmap_now: dict[str, dict[str, float]]
    heatmap_3m: dict[str, dict[str, float]]
    heatmap_1y: dict[str, dict[str, float]]
    pairs: List[CorrelationPairItem]


class CorrelationsEnvelope(BaseEnvelope[CorrelationsData]):
    pass


# 9. Relative Rotation Map
class RotationItem(BaseModel):
    asset: str
    name: str
    group: str
    rs_ratio: float
    rs_mom: float
    phase: str  # 'Leading' | 'Fading' | 'Lagging' | 'Recovering'
    tail_dates: List[str]
    tail_x: List[float]
    tail_y: List[float]


class RotationData(BaseModel):
    benchmark: str
    assets: List[RotationItem]


class RotationEnvelope(BaseEnvelope[RotationData]):
    pass


# 10. Historical Context
class AnalogItem(BaseModel):
    date: str
    value: float
    distance: float
    fwd_5d: Optional[float] = None
    fwd_20d: Optional[float] = None


class ContextData(BaseModel):
    id: str
    name: str
    group: str
    region: str
    unit: str
    current_value: float
    percentile: float
    z_score: float
    n: int
    n_eff: float
    histogram_bins: List[float]
    histogram_counts: List[int]
    kde_grid: List[float] = []
    kde_density: List[float] = []
    normal_density: List[float] = []
    qq_theoretical: List[float] = []
    qq_empirical: List[float] = []
    ecdf_x: List[float] = []
    ecdf_y: List[float] = []
    calendar_heatmap: List[dict[str, Any]] = []
    analogs: List[AnalogItem]
    peer_rank: int
    peer_count: int
    description: str
    source_note: str
    latest_return: Optional[float] = None
    return_percentile: Optional[float] = None


class ContextEnvelope(BaseEnvelope[ContextData]):
    pass


# 11. Global Liquidity
class LiquidityAudit(BaseModel):
    formula: str
    parameters: dict[str, Any]
    series_used: List[str]
    n: int
    n_eff: float


class LiquidityData(BaseModel):
    current_net_liquidity: float
    change_4w: float
    change_4w_pct: float
    change_13w: float
    change_13w_pct: float
    stablecoins_total_bn: float
    stablecoins_30d_pct: float
    dates: List[str]
    net_liquidity: List[float]
    fed_assets: List[float]
    tga: List[float]
    rrp: List[float]
    spx_overlay: List[float]
    rolling_corr_60d: List[float]
    audit: LiquidityAudit


class LiquidityEnvelope(BaseEnvelope[LiquidityData]):
    pass


# 12. India Flows & Positioning
class OiBuildupItem(BaseModel):
    symbol: str
    price_change_pct: float
    oi_change_pct: float
    buildup: str  # 'Long Buildup' | 'Short Buildup' | 'Short Covering' | 'Long Unwinding' | 'Neutral'
    pcr: float


class IndiaFlowsAudit(BaseModel):
    formula: str
    parameters: dict[str, Any]
    series_used: List[str]
    n: int
    n_eff: float


class IndiaFlowsData(BaseModel):
    india_vix: float
    india_vix_percentile: float
    pcr: float
    fii_net_1d: float
    dii_net_1d: float
    fii_net_5d: float
    dii_net_5d: float
    fii_net_20d: float
    dii_net_20d: float
    fii_20d_percentile: float
    dii_20d_percentile: float
    delivery_pct_index: float
    delivery_pct_percentile: float
    flow_dates: List[str]
    fii_daily: List[float]
    dii_daily: List[float]
    fii_cum_20d: List[float]
    dii_cum_20d: List[float]
    oi_buildup: List[OiBuildupItem]
    audit: IndiaFlowsAudit


class IndiaFlowsEnvelope(BaseEnvelope[IndiaFlowsData]):
    pass


# 13. Financial Stress Composite
class CrisisEpisode(BaseModel):
    name: str
    date: str
    stress: float


class StressAudit(BaseModel):
    formula: str
    parameters: dict[str, Any]
    series_used: List[str]
    n: int
    n_eff: float


class StressData(BaseModel):
    current_stress: float
    regime: str
    components: dict[str, float]
    missing_components: List[str]
    dates: List[str]
    history: List[float]
    episodes: List[CrisisEpisode]
    audit: StressAudit


class StressEnvelope(BaseEnvelope[StressData]):
    pass


# 14. Daily Briefing
class BriefingSection(BaseModel):
    title: str
    content: str


class BriefingData(BaseModel):
    as_of: str
    headline: str
    executive_summary: str
    stress_score: float
    stress_regime: str
    sections: List[BriefingSection]


class BriefingEnvelope(BaseEnvelope[BriefingData]):
    pass


# 15. Move Decomposition
class DecompositionHistoryPoint(BaseModel):
    date: str
    total_return: float
    contributions: dict[str, float]
    residual: float


class DecompositionData(BaseModel):
    target: str
    target_id: str
    horizon_days: int
    lens: str  # "local" | "usd"
    total_return: float
    contributions: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    residual: float
    betas: dict[str, float]
    r2_in_sample: float
    r2_out_sample: float
    mostly_unexplained: bool
    regularized: bool
    n_obs: int
    window_dates: dict[str, str]
    narrative: str
    history: List[DecompositionHistoryPoint] = []


class DecompositionEnvelope(BaseEnvelope[DecompositionData]):
    pass


# 16. Country Layer (M5)
class CountryMatrixRow(BaseModel):
    code: str
    name: str
    index_id: str
    currency: str
    returns_local: dict[str, float]
    returns_usd: dict[str, float]
    returns_fx: dict[str, float]


class CountriesData(BaseModel):
    countries: List[CountryMatrixRow]
    breadth_200d: float
    avg_corr_60d: float
    avg_corr_250d: float
    tracked_count: int


class CountriesEnvelope(BaseEnvelope[CountriesData]):
    pass


# 17. Sector Layer (M7)
class SectorRotationPoint(BaseModel):
    date: str
    rs_ratio: float
    rs_mom: float


class SectorItem(BaseModel):
    id: str
    name: str
    weight: float
    returns: dict[str, float]
    percentiles: dict[str, float]
    contributions: dict[str, float]
    rs_ratio: float
    rs_mom: float
    phase: str
    trail: List[SectorRotationPoint] = []


class SectorRegionData(BaseModel):
    region_id: str
    country_name: str
    benchmark_name: str
    index_id: str
    as_of: str
    weights_source: str
    weights_source_url: str
    index_returns: dict[str, float]
    sector_breadth: dict[str, float]
    residuals: dict[str, float]
    sectors: List[SectorItem]


class SectorsData(BaseModel):
    regions: dict[str, SectorRegionData]


class SectorsEnvelope(BaseEnvelope[SectorsData]):
    pass


# 18. Driver Sensitivity Layer (M8)
class DriverSensitivityItem(BaseModel):
    driver_id: str
    beta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    t_stat: float
    stability: float
    prior: str
    prior_aligned: bool
    significant: bool
    history: List[dict[str, Any]] = []


class DriverTargetMatrixRow(BaseModel):
    target_id: str
    target_name: str
    target_code: str
    window: int
    r2: float
    regularized: bool
    n_obs: int
    drivers: dict[str, DriverSensitivityItem]


class DriverScatterData(BaseModel):
    scatter: List[List[float]]
    line: List[List[float]]
    band_upper: List[List[float]]
    band_lower: List[List[float]]
    beta: float
    r2: float
    t_stat: float
    n_obs: int


class DriversData(BaseModel):
    matrix_250d: List[DriverTargetMatrixRow]
    matrix_120d: List[DriverTargetMatrixRow]
    pairs_tested: int
    scatter_examples: dict[str, DriverScatterData] = {}


class DriversEnvelope(BaseEnvelope[DriversData]):
    pass


# 19. Risk Lab Layer (M9)
class VolatilityConePoint(BaseModel):
    window: int
    min: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    max: float
    current: float
    percentile: float


class RealizedEwmaPoint(BaseModel):
    date: str
    realized_vol: float
    ewma_vol: float


class DistributionShapeData(BaseModel):
    skewness: float
    excess_kurtosis: float
    jb_stat: float
    jb_pvalue: float
    qq_theoretical: List[float]
    qq_empirical: List[float]
    mean: float
    std: float


class TailRiskMetric(BaseModel):
    confidence: float
    hist_var: float
    hist_es: float
    gauss_var: float
    gauss_es: float
    tail_ratio: float


class AutocorrelationRiskData(BaseModel):
    lags: List[int]
    acf_returns: List[float]
    acf_squared: List[float]
    band: float
    n_eff: float


class DrawdownEpisodeItem(BaseModel):
    peak_date: str
    trough_date: str
    recovery_date: Optional[str] = None
    depth: float
    duration_days: int
    recovery_days: Optional[int] = None
    status: str


class DrawdownSeriesPoint(BaseModel):
    date: str
    drawdown: float


class PcaFactorData(BaseModel):
    scree: List[float]
    loadings_pc1: dict[str, float]
    loadings_pc2: dict[str, float]
    pc1_history: List[dict[str, Any]]


class DownsideCorrelationItem(BaseModel):
    target_id: str
    target_name: str
    all_day_corr: float
    downside_corr: float
    stress_premium: float
    stress_threshold: float
    n_all: int
    n_stress: int


class AssetRiskProfile(BaseModel):
    asset_id: str
    asset_name: str
    asset_group: str
    volatility_cone: List[VolatilityConePoint]
    vol_history: List[RealizedEwmaPoint]
    distribution: DistributionShapeData
    tail_risk: List[TailRiskMetric]
    autocorrelation: AutocorrelationRiskData
    drawdown_series: List[DrawdownSeriesPoint]
    drawdown_episodes: List[DrawdownEpisodeItem]


class RiskData(BaseModel):
    selected_asset: str
    assets_available: List[dict[str, str]]
    profiles: dict[str, AssetRiskProfile]
    pca: PcaFactorData
    downside_correlations: List[DownsideCorrelationItem]
    market_synchronization_latest: float


class RiskEnvelope(BaseEnvelope[RiskData]):
    pass


# --- P0: Source Status & Resilience Schemas ---

class SourceStatusItem(BaseModel):
    id: str
    name: str
    provider: str
    documentation_url: str
    terms_url: str
    rate_limit: str
    schedule: str
    status: str = "healthy"
    last_success: str
    last_verified: str
    tracked_series_count: int
    stale_series_count: int = 0
    circuit_state: str = "CLOSED"
    owner_notes: str = ""


class SourceStatusData(BaseModel):
    sources: List[SourceStatusItem]
    overall_status: str = "healthy"
    total_tracked_series: int = 0
    total_healthy_series: int = 0


class SourceStatusEnvelope(BaseEnvelope[SourceStatusData]):
    pass


# --- P1: Benchmarks and Episodes Schemas ---

class CrossCorrelationItem(BaseModel):
    lag_weeks: int
    correlation: float
    interpretation: str


class BenchmarkAgreementItem(BaseModel):
    benchmark_name: str
    n_obs: int
    effective_n: float
    pearson_corr: float
    spearman_rank_corr: float
    best_lead_lag: int
    cross_correlation: List[CrossCorrelationItem] = Field(default_factory=list)


class HistoricalEpisodeItem(BaseModel):
    id: str
    name: str
    start_date: str
    end_date: str
    category: str
    peak_stress_score: float
    peak_stress_date: str
    mean_stress_score: float
    lead_lag_to_official_peak_days: int
    elevated_ratio: float
    description: str = ""


class BenchmarkData(BaseModel):
    stress_cross_checks: List[BenchmarkAgreementItem]
    episodes: List[HistoricalEpisodeItem]
    policy_uncertainty_level: float = 120.5
    policy_uncertainty_percentile: float = 58.0


class BenchmarkEnvelope(BaseEnvelope[BenchmarkData]):
    pass


# --- P3: Options-Implied Layer Schemas ---

class TermStructurePoint(BaseModel):
    tenor: str
    days: int
    current: float
    one_month_ago: float
    one_year_median: float
    p10: float
    p90: float


class FearTailItem(BaseModel):
    id: str
    name: str
    current: float
    percentile: float
    z_score: float
    chg_1w: float
    chg_1m: float
    sparkline: List[float] = Field(default_factory=list)


class ImpliedVsRealizedItem(BaseModel):
    market: str
    implied_vol: float
    realized_vol_21d: float
    gap: float
    gap_percentile: float
    vrp_proxy: Optional[float] = None
    vrp_label: str = "Trailing proxy"


class SlopeHistoryItem(BaseModel):
    date: str
    vix: float
    vix3m: float
    slope_3m: float
    slope_9d: float
    state: str
    percentile: float


class ImpliedVolData(BaseModel):
    as_of: str
    current_state: str
    slope_3m: float
    slope_9d: float
    slope_percentile: float
    curve: List[TermStructurePoint]
    fear_tail_indices: List[FearTailItem]
    implied_vs_realized: List[ImpliedVsRealizedItem]
    slope_history: List[SlopeHistoryItem]
    interpretation: str


class ImpliedVolEnvelope(BaseEnvelope[ImpliedVolData]):
    pass


# --- P4: Yield Curve Lab Schemas ---

class YieldCurvePointItem(BaseModel):
    maturity: str
    months: int
    yield_pct: float
    fitted_pct: float


class SmoothPoint(BaseModel):
    months: float
    yield_pct: float


class CurveSnapshotItem(BaseModel):
    label: str
    date: str
    points: List[YieldCurvePointItem]
    smooth_fitted: List[SmoothPoint] = Field(default_factory=list)


class NelsonSiegelFactorsItem(BaseModel):
    level: float
    slope: float
    curvature: float
    rmse: float
    level_percentile: float
    slope_percentile: float
    curvature_percentile: float


class YieldSpreadItem(BaseModel):
    name: str
    current: float
    chg_1m: float
    percentile: float
    inverted: bool


class TenYearAnatomyItem(BaseModel):
    date: str
    nominal_10y: float
    real_10y: float
    breakeven_10y: float
    term_premium: float


class HeatmapRowItem(BaseModel):
    date: str
    yields: dict[str, float]


class YieldCurveData(BaseModel):
    as_of: str
    factors: NelsonSiegelFactorsItem
    spreads: List[YieldSpreadItem]
    curves: List[CurveSnapshotItem]
    anatomy: List[TenYearAnatomyItem]
    heatmap: List[HeatmapRowItem]
    maturities: List[str]
    interpretation: str


class YieldCurveEnvelope(BaseEnvelope[YieldCurveData]):
    pass


# --- P6: Event Studies Schemas ---

class ReactionRibbonPoint(BaseModel):
    day: int
    mean_car: float
    median_car: float
    ci_lower_90: float
    ci_lower_50: float
    ci_upper_50: float
    ci_upper_90: float


class DecisionStatsItem(BaseModel):
    count: int
    mean_day_0_ar: float
    median_day_0_ar: float
    pct_positive: float


class EventDetailItem(BaseModel):
    id: str
    date: str
    decision: str
    rate_before: float
    rate_after: float
    day_0_ar: float
    car_10d: float
    description: str = ""


class EventStudyData(BaseModel):
    event_type: str
    target_name: str
    n_events: int
    ribbon: List[ReactionRibbonPoint]
    by_decision: dict[str, DecisionStatsItem]
    magnitude_ratio: float
    events: List[EventDetailItem]
    interpretation: str


class EventStudyEnvelope(BaseEnvelope[EventStudyData]):
    pass


# --- P7: Connectedness Schemas ---

class ConnectednessCountryItem(BaseModel):
    id: str
    name: str
    own: float
    from_others: float
    to_others: float
    net: float
    role: str


class TciPoint(BaseModel):
    date: str
    tci: float


class ConnectednessData(BaseModel):
    as_of: str
    horizon_weeks: int
    tci: float
    countries: List[ConnectednessCountryItem]
    matrix: List[dict[str, Any]]
    labels: List[str]
    tci_history: List[TciPoint]
    interpretation: str


class ConnectednessEnvelope(BaseEnvelope[ConnectednessData]):
    pass


# --- P9: Fundamentals (Energy & Agriculture) Schemas ---

class SeasonalWeekPoint(BaseModel):
    week: int
    min_5y: float
    max_5y: float
    mean_5y: float
    current_year: Optional[float] = None
    prev_year: Optional[float] = None


class InventoryItem(BaseModel):
    id: str
    name: str
    unit: str
    as_of: str
    current_level: float
    mean_5y: float
    min_5y: float
    max_5y: float
    deviation: float
    deviation_pct: float
    deviation_percentile: float
    build_draw_1w: float
    band_years_label: str
    weeks: List[SeasonalWeekPoint]


class FundamentalsData(BaseModel):
    as_of: str
    inventories: List[InventoryItem]
    lag_label: str
    interpretation: str


class FundamentalsEnvelope(BaseEnvelope[FundamentalsData]):
    pass


# --- P8: Slow Global Layer Schemas ---

class ReerValuationItem(BaseModel):
    id: str
    name: str
    current_index: float
    percentile: float
    chg_1y: float
    valuation: str
    as_of: str


class ForeignTreasuryHoldingItem(BaseModel):
    country: str
    holdings_bn: float
    share_pct: float
    chg_1y_bn: float
    as_of: str


class CentralBankAssetItem(BaseModel):
    central_bank: str
    assets_trillion_usd: float
    chg_1y_pct: float
    as_of: str


class IndiaMacroProfile(BaseModel):
    fx_reserves_bn_usd: float
    fx_reserves_as_of: str
    import_cover_months: float
    fpi_equity_12m_bn: float
    fpi_debt_12m_bn: float
    fpi_net_total_12m_bn: float
    rupee_reer_percentile: float


class GlobalSlowData(BaseModel):
    as_of: str
    reer: List[ReerValuationItem]
    foreign_treasury_holdings: List[ForeignTreasuryHoldingItem]
    central_bank_assets: List[CentralBankAssetItem]
    india_macro: IndiaMacroProfile
    lag_label: str
    interpretation: str


class GlobalSlowEnvelope(BaseEnvelope[GlobalSlowData]):
    pass


# --- P10: Advanced Tails and Regimes Schemas ---

class MeanExcessPoint(BaseModel):
    threshold: float
    quantile: float
    mean_excess: float
    count: int


class EvtGpdData(BaseModel):
    shape_xi: float
    scale_beta: float
    threshold_u: float
    tail_fatness: str
    var_99_pct: float
    es_99_pct: float
    var_995_pct: float
    es_995_pct: float
    gaussian_var_99: float
    gaussian_var_995: float
    empirical_var_99: float
    empirical_var_995: float
    excess_ratio_over_gaussian: float
    mean_excess_curve: List[MeanExcessPoint] = Field(default_factory=list)


class RegimeRibbonPoint(BaseModel):
    date: str
    filtered_prob_stress: float
    smoothed_prob_stress: float
    regime: str


class MarkovRegimesData(BaseModel):
    current_regime: str
    current_prob_stress: float
    transition_matrix: dict[str, float]
    expected_duration_days: dict[str, float]
    ribbon: List[RegimeRibbonPoint] = Field(default_factory=list)


class TailsRegimesData(BaseModel):
    as_of: str
    target_name: str
    evt: EvtGpdData
    regimes: MarkovRegimesData
    interpretation: str


class TailsRegimesEnvelope(BaseEnvelope[TailsRegimesData]):
    pass


# --- P11: Track Record & Validation Schemas ---

class TrackRecordRuleItem(BaseModel):
    rule_id: str
    rule_name: str
    hypothesis: str
    horizon_days: int
    n_alerts: int
    n_episodes: int
    hit_rate: float
    base_rate: float
    edge_pct: float
    ci_lower: float
    ci_upper: float
    median_lead_days: float
    status: str
    sample_period: str


class CalibrationBandItem(BaseModel):
    severity: str
    n_episodes: int
    observed_rate: float
    expected_rate: float


class RuleChangelogItem(BaseModel):
    date: str
    version: str
    rule_id: str
    change_summary: str
    commit_hash: str


class TrackRecordData(BaseModel):
    as_of: str
    overall_hit_rate: float
    overall_base_rate: float
    total_episodes: int
    rules: List[TrackRecordRuleItem] = Field(default_factory=list)
    calibration: List[CalibrationBandItem] = Field(default_factory=list)
    changelog: List[RuleChangelogItem] = Field(default_factory=list)
    statement: str


class TrackRecordEnvelope(BaseEnvelope[TrackRecordData]):
    pass


# 22. Cross-Asset Ratios & Fair-Value Models
class RatioItem(BaseModel):
    id: str
    name: str
    category: str
    op: str
    current: float
    chg_1d: float
    chg_20d: float
    z_score: float
    percentile: float
    interpretation: str
    history_dates: List[str] = Field(default_factory=list)
    history_values: List[float] = Field(default_factory=list)


class RatiosData(BaseModel):
    ratios: List[RatioItem] = Field(default_factory=list)
    gold_valuation: dict[str, Any] = Field(default_factory=dict)
    cu_au_vs_10y: dict[str, Any] = Field(default_factory=dict)
    credit_vs_vol: dict[str, Any] = Field(default_factory=dict)


class RatiosEnvelope(BaseEnvelope[RatiosData]):
    pass


# 23. 4-Quadrant Growth-Inflation Macro Cycles
class CycleTrailPoint(BaseModel):
    date: str
    growth_z: float
    inflation_z: float
    quadrant: str


class CycleQuadrantReturn(BaseModel):
    quadrant: str
    days: int
    share_pct: float
    returns: dict[str, Optional[float]] = Field(default_factory=dict)


class CycleGaugeItem(BaseModel):
    id: str
    name: str
    value: float
    unit: str
    percentile: float
    note: str


class CyclesData(BaseModel):
    current_quadrant: str
    current_growth_z: float
    current_inflation_z: float
    trail: List[CycleTrailPoint] = Field(default_factory=list)
    quadrant_returns: List[CycleQuadrantReturn] = Field(default_factory=list)
    gauges: List[CycleGaugeItem] = Field(default_factory=list)


class CyclesEnvelope(BaseEnvelope[CyclesData]):
    pass












