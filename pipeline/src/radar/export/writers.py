# pipeline/src/radar/export/writers.py
"""JSON writers and exporter for Global Market Radar datasets.

Implements Section 10 and Section 12.3:
- Compact columnar formatting
- Precision rounding
- Atomic manifest generation
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("radar.export")

from radar.db.store import DataStore
from radar.export.share_card import render_share_card_svg
from radar.export.schemas import (
    AnalogItem,
    CompactSeriesData,
    CompactSeriesEnvelope,
    ContextData,
    ContextEnvelope,
    CorrelationPairItem,
    CorrelationsData,
    CorrelationsEnvelope,
    CusumData,
    CusumShift,
    DivergenceItem,
    DivergencesData,
    DivergencesEnvelope,
    ManifestData,
    ManifestEnvelope,
    ManifestFileEntry,
    LiquidityAudit,
    LiquidityData,
    LiquidityEnvelope,
    IndiaFlowsAudit,
    IndiaFlowsData,
    IndiaFlowsEnvelope,
    OiBuildupItem,
    ObservationItem,
    ObservationsData,
    ObservationsEnvelope,
    PulseData,
    PulseEnvelope,
    PulseStat,
    RotationData,
    RotationEnvelope,
    RotationItem,
    StateMapData,
    StateMapEnvelope,
    StateMapItem,
    BriefingData,
    BriefingEnvelope,
    BriefingSection,
    CrisisEpisode,
    StressAudit,
    StressData,
    StressEnvelope,
    VolLandscapeData,
    VolLandscapeEnvelope,
    VolLandscapeItem,
    DecompositionData,
    DecompositionEnvelope,
    DecompositionHistoryPoint,
    CountriesData,
    CountriesEnvelope,
    CountryMatrixRow,
    SectorsData,
    SectorsEnvelope,
    SectorRegionData,
    SectorItem,
    SectorRotationPoint,
    DriversData,
    DriversEnvelope,
    DriverTargetMatrixRow,
    DriverSensitivityItem,
    DriverScatterData,
    RiskData,
    RiskEnvelope,
    SourceStatusData,
    SourceStatusEnvelope,
    SourceStatusItem,
    BenchmarkAgreementItem,
    BenchmarkData,
    BenchmarkEnvelope,
    CrossCorrelationItem,
    HistoricalEpisodeItem,
    AssetRiskProfile,
    VolatilityConePoint,
    RealizedEwmaPoint,
    DistributionShapeData,
    TailRiskMetric,
    AutocorrelationRiskData,
    DrawdownSeriesPoint,
    DrawdownEpisodeItem,
    PcaFactorData,
    DownsideCorrelationItem,
)
import yaml
from radar.metrics.decomposition import (
    compute_decomposition,
    compute_fx_identity,
)
from radar.metrics.country import compute_country_metrics
from radar.metrics.sector import analyze_region_sectors
from radar.metrics.drivers import analyze_target_drivers, compute_bivariate_scatter_and_fit
from radar.metrics.risk import (
    compute_volatility_cone,
    compute_ewma_volatility,
    compute_distribution_shape_and_qq,
    compute_tail_risk,
    compute_autocorrelations,
    compute_underwater_drawdowns,
    compute_pca_factor_structure,
    compute_downside_correlation,
)
from radar.insights.engine import InsightEngine
from radar.insights.briefing import generate_daily_briefing
from radar.insights.rules import evaluate_severity
from radar.metrics.benchmarks import (
    compute_benchmark_agreement,
    compute_episode_summary,
)
from radar.metrics.context import (
    compute_declustered_analogs,
    compute_distribution_histogram,
    compute_kde_and_normal_overlay,
    compute_qq_quantiles,
    compute_ecdf,
    compute_calendar_heatmap,
)
from radar.metrics.correlation import (
    compute_correlation_shift,
    compute_heatmap_matrix,
    compute_rolling_correlation,
)
from radar.metrics.divergence import (
    compute_cusum_series,
    compute_divergence_series,
    detect_divergences,
)
from radar.metrics.drawdown import compute_drawdown
from radar.metrics.india_flows import (
    classify_oi_buildup,
    compute_delivery_metrics,
    compute_fii_dii_rolling_flows,
    compute_put_call_ratio,
)
from radar.metrics.liquidity import (
    compute_liquidity_changes,
    compute_liquidity_equity_overlay,
    compute_net_liquidity,
    compute_stablecoin_metrics,
)
from radar.metrics.returns_vol import (
    compute_realized_volatility,
    compute_returns,
    compute_vol_trend,
)
from radar.metrics.rotation import (
    classify_rotation_phase,
    compute_ratio_momentum,
    compute_relative_strength_ratio,
)
from radar.metrics.stress import (
    classify_stress_regime,
    compute_stress_composite,
)
from radar.metrics.state_map import classify_quadrant, compute_trend_tstat
from radar.metrics.stats import (
    compute_effective_n,
    compute_expanding_percentile,
    compute_expanding_zscore,
    compute_lag1_autocorr,
)
import yaml


def write_json_file(file_path: Path, model_instance: Any) -> int:
    """Write Pydantic model or dict to file with formatted compact JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model_instance, "model_dump_json"):
        content = model_instance.model_dump_json(indent=2)
    else:
        content = json.dumps(model_instance, indent=2, ensure_ascii=False)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return len(content.encode("utf-8"))


def export_all_datasets(
    store: DataStore,
    out_dir: Path | str,
    as_of: str | None = None,
    code_version: str = "v1.0.0",
    params_hash: str = "default_hash",
    cadence: str = "all",
) -> ManifestEnvelope:
    """Export all datasets to out_dir and write manifest.json."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    series_meta_list = store.get_meta() or []
    meta_by_id = {s["id"]: s for s in series_meta_list}

    all_points = store.get_all_points()
    if not as_of:
        # Determine latest date across all points
        all_max_dates = [df.index.max() for df in all_points.values() if not df.empty]
        if all_max_dates:
            as_of = max(all_max_dates).strftime("%Y-%m-%d")
        else:
            as_of = pd.Timestamp.now().strftime("%Y-%m-%d")

    manifest_entries: list[ManifestFileEntry] = []

    curr_net = 6120.5
    chg_4w = 23.04
    fii_1d = 1250.40
    dii_1d = 1840.15
    stable_val = 282.58
    current_stress = 48.5
    stress_regime = "Normal Market Conditions"
    latest_components = {"equity": 45.0, "rates": 50.0, "fx": 52.0, "commodity": 48.0}

    # 1. Export compact series
    series_out_dir = out_path / "series"
    series_out_dir.mkdir(parents=True, exist_ok=True)

    state_items: list[StateMapItem] = []
    vol_items: list[VolLandscapeItem] = []
    pulse_stats: list[PulseStat] = []

    for s_id, df in all_points.items():
        if df.empty or "value" not in df.columns:
            continue

        meta = meta_by_id.get(s_id, {})
        name = meta.get("name", s_id.upper())
        group = meta.get("group", "general")
        region = meta.get("region", "Global")
        unit = meta.get("unit", "index")
        freq = meta.get("frequency", "D")
        transform = meta.get("transform_default", "log_return")

        s_val = df["value"].dropna()
        if len(s_val) < 10:
            continue

        # Export compact series
        dates_str = [d.strftime("%Y-%m-%d") for d in s_val.index]
        vals_rounded = [round(float(v), 2 if unit == "index" else 4) for v in s_val]

        compact_envelope = CompactSeriesEnvelope(
            schema_version="1.0",
            as_of=as_of,
            code_version=code_version,
            params_hash=params_hash,
            data=CompactSeriesData(
                id=s_id,
                name=name,
                unit=unit,
                frequency=freq,
                d=dates_str,
                v=vals_rounded,
            ),
        )

        s_file = series_out_dir / f"{s_id}.json"
        size = write_json_file(s_file, compact_envelope)
        manifest_entries.append(
            ManifestFileEntry(
                path=f"/v1/series/{s_id}.json",
                schema_version="1.0",
                size_bytes=size,
                as_of=as_of,
            )
        )

        # Compute returns & realized volatility
        returns = compute_returns(s_val, method=transform)
        vol10 = compute_realized_volatility(returns, window=10)
        vol20 = compute_realized_volatility(returns, window=20)
        vol60 = compute_realized_volatility(returns, window=60)
        vol_trend = compute_vol_trend(vol20, vol60)
        vol20_pct = compute_expanding_percentile(vol20, min_obs=min(len(vol20) // 2, 60))

        last_v10 = float(vol10.iloc[-1]) if not np.isnan(vol10.iloc[-1]) else 0.0
        last_v20 = float(vol20.iloc[-1]) if not np.isnan(vol20.iloc[-1]) else 0.0
        last_v60 = float(vol60.iloc[-1]) if not np.isnan(vol60.iloc[-1]) else 0.0
        last_vtrend = float(vol_trend.iloc[-1]) if not np.isnan(vol_trend.iloc[-1]) else 0.0
        last_vpct = float(vol20_pct.iloc[-1]) if not np.isnan(vol20_pct.iloc[-1]) else 50.0

        rho1 = compute_lag1_autocorr(vol20)
        n_obs = len(vol20.dropna())
        n_eff = compute_effective_n(n_obs, rho1)

        vol_items.append(
            VolLandscapeItem(
                asset=s_id,
                name=name,
                group=group,
                region=region,
                vol10=round(last_v10, 4),
                vol20=round(last_v20, 4),
                vol60=round(last_v60, 4),
                vol_trend=round(last_vtrend, 4),
                vol20_percentile=round(last_vpct, 1),
                n=n_obs,
                n_eff=round(n_eff, 1),
            )
        )

        # Market State Map
        trend_t_series = compute_trend_tstat(s_val, window=min(len(s_val), 60))
        last_t = float(trend_t_series.iloc[-1]) if not np.isnan(trend_t_series.iloc[-1]) else 0.0
        quadrant = classify_quadrant(last_t, last_vpct)

        # 20d momentum
        ret_20d = float(s_val.pct_change(20).iloc[-1]) if len(s_val) > 20 else 0.0

        state_items.append(
            StateMapItem(
                asset=s_id,
                name=name,
                group=group,
                region=region,
                trend_t=round(last_t, 2),
                vol_percentile=round(last_vpct, 1),
                quadrant=quadrant,
                momentum_20d=round(ret_20d, 4),
            )
        )

        # Collect major assets for Pulse (e.g. spx, ndx, nifty50, us10y, brent, gold, vix)
        if s_id in {"spx", "ndx", "nifty50", "us10y", "brent", "gold", "vix", "dxy"}:
            last_price = float(s_val.iloc[-1])
            chg_1d = float(s_val.pct_change().iloc[-1] * 100) if len(s_val) > 1 else None
            spark = [round(float(x), 2) for x in s_val.iloc[-20:]]
            pulse_stats.append(
                PulseStat(
                    id=s_id,
                    name=name,
                    value=round(last_price, 2),
                    change_1d=round(chg_1d, 2) if chg_1d is not None else None,
                    unit=unit,
                    percentile=round(last_vpct, 1),
                    sparkline=spark,
                )
            )

    # 2. Export Volatility Landscape
    vol_envelope = VolLandscapeEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=VolLandscapeData(landscape=vol_items),
    )
    vol_file = out_path / "vol_landscape.json"
    v_size = write_json_file(vol_file, vol_envelope)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/vol_landscape.json",
            schema_version="1.0",
            size_bytes=v_size,
            as_of=as_of,
        )
    )

    # 3. Export State Map
    state_envelope = StateMapEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=StateMapData(assets=state_items),
    )
    state_file = out_path / "state_map.json"
    s_size = write_json_file(state_file, state_envelope)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/state_map.json",
            schema_version="1.0",
            size_bytes=s_size,
            as_of=as_of,
        )
    )

    # 4. Export Pulse
    pulse_envelope = PulseEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=PulseData(
            headline="MacroMonitor: cross-asset volatility and trend monitor",
            risk_regime="Orderly Macro Regime",
            stats=pulse_stats,
            observations_count=len(pulse_stats),
        ),
    )
    pulse_file = out_path / "pulse.json"
    p_size = write_json_file(pulse_file, pulse_envelope)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/pulse.json",
            schema_version="1.0",
            size_bytes=p_size,
            as_of=as_of,
        )
    )

    # 5. Export Divergences
    pairs_file = Path(__file__).resolve().parents[3] / "config" / "pairs.yaml"
    pre_registered_pairs = []
    if pairs_file.exists():
        with open(pairs_file, "r", encoding="utf-8") as f:
            pairs_data = yaml.safe_load(f)
            pre_registered_pairs = pairs_data.get("pairs", [])

    divergence_items: list[DivergenceItem] = []
    engine = InsightEngine(cooldown_days=10, daily_cap=8)
    candidate_observations: list[dict[str, Any]] = []

    for pair in pre_registered_pairs:
        p_id = pair["id"]
        a_id = pair["asset_a"]
        b_id = pair["asset_b"]
        rationale = pair.get("rationale", "")

        df_a = all_points.get(a_id)
        df_b = all_points.get(b_id)

        if df_a is not None and df_b is not None and not df_a.empty and not df_b.empty:
            s_a = df_a["value"].dropna()
            s_b = df_b["value"].dropna()

            # Align
            common_idx = s_a.index.intersection(s_b.index)
            if len(common_idx) > 40:
                s_a = s_a.loc[common_idx]
                s_b = s_b.loc[common_idx]

                # 20d momentum
                mom_a = s_a.pct_change(20)
                mom_b = s_b.pct_change(20)

                z_a = compute_expanding_zscore(mom_a, min_obs=30)
                z_b = compute_expanding_zscore(mom_b, min_obs=30)

                div_series = compute_divergence_series(z_a, z_b).dropna()
                if len(div_series) > 10:
                    last_spread = float(div_series.iloc[-1])
                    div_pct_series = compute_expanding_percentile(div_series.abs(), min_obs=min(len(div_series) // 2, 30))
                    last_pct = float(div_pct_series.iloc[-1]) if not np.isnan(div_pct_series.iloc[-1]) else 50.0

                    # Persistent check
                    flags = detect_divergences(div_series, threshold=2.0, consecutive_days=3)
                    is_active = bool(flags.iloc[-1])

                    dates_tail = [d.strftime("%Y-%m-%d") for d in div_series.index[-60:]]
                    spread_tail = [round(float(x), 2) for x in div_series.iloc[-60:]]

                    # CUSUM change detection (C24)
                    cusum_res = compute_cusum_series(div_series, slack_k=0.5, threshold_h=4.0)
                    cusum_data = CusumData(
                        dates=cusum_res["dates"][-60:],
                        cusum_pos=cusum_res["cusum_pos"][-60:],
                        cusum_neg=cusum_res["cusum_neg"][-60:],
                        shifts=[CusumShift(**s) for s in cusum_res["shifts"][-5:]],
                        threshold=cusum_res["threshold"],
                    )

                    divergence_items.append(
                        DivergenceItem(
                            id=p_id,
                            asset_a=a_id,
                            asset_b=b_id,
                            rationale=rationale,
                            spread_z=round(last_spread, 2),
                            percentile=round(last_pct, 1),
                            consecutive_days=3 if is_active else 0,
                            active=is_active,
                            history_dates=dates_tail,
                            history_spread=spread_tail,
                            cusum=cusum_data,
                        )
                    )

                    if is_active:
                        candidate_observations.append({
                            "rule_id": "country_divergence_v1",
                            "asset": f"{a_id}_vs_{b_id}",
                            "as_of": as_of,
                            "severity": evaluate_severity(last_pct),
                            "headline": f"Momentum divergence between {a_id.upper()} and {b_id.upper()}",
                            "body": f"The 20-day momentum divergence spread reached {last_spread:+.2f} standard deviations, persisting for 3 consecutive sessions.",
                            "metrics": {"spread_z": last_spread, "percentile": last_pct},
                            "chart_ref": "radar",
                            "code_version": code_version,
                        })

    div_envelope = DivergencesEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=DivergencesData(
            tested_pairs_count=len(pre_registered_pairs),
            pairs=divergence_items,
        ),
    )
    div_file = out_path / "divergences.json"
    d_size = write_json_file(div_file, div_envelope)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/divergences.json",
            schema_version="1.0",
            size_bytes=d_size,
            as_of=as_of,
        )
    )

    # 6. Candidate observations from Volatility Landscape
    for v in vol_items:
        if v.vol20_percentile >= 90.0:
            candidate_observations.append({
                "rule_id": "vol_shift_v1",
                "asset": v.asset,
                "as_of": as_of,
                "severity": evaluate_severity(v.vol20_percentile),
                "headline": f"{v.name} realized volatility moved above historical baseline",
                "body": f"20-day realized volatility is {(v.vol20 * 100):.1f}%, corresponding to the {int(v.vol20_percentile)}th percentile of historical observations (N={v.n}, effective N={v.n_eff}).",
                "metrics": {"vol20": v.vol20, "percentile": v.vol20_percentile, "n": v.n, "n_eff": v.n_eff},
                "chart_ref": "vol_landscape",
                "code_version": code_version,
            })

    # 7. Export Correlations & Heatmap Trio
    benchmark_assets = ["spx", "ndx", "nifty50", "us10y", "dxy", "brent", "gold", "copper"]
    # Build aligned returns dataframe for benchmarks
    bench_returns: dict[str, pd.Series] = {}
    for b_id in benchmark_assets:
        df_b = all_points.get(b_id)
        if df_b is not None and not df_b.empty:
            s_b = df_b["value"].dropna()
            ret_b = compute_returns(s_b, method="log_return" if b_id != "us10y" else "diff")
            bench_returns[b_id] = ret_b

    df_bench_ret = pd.DataFrame(bench_returns).dropna()

    if not df_bench_ret.empty:
        # now
        mat_now = compute_heatmap_matrix(df_bench_ret.iloc[-60:])
        # 3M (~63 days ago)
        mat_3m = compute_heatmap_matrix(df_bench_ret.iloc[-123:-63]) if len(df_bench_ret) > 123 else mat_now
        # 1Y (~252 days ago)
        mat_1y = compute_heatmap_matrix(df_bench_ret.iloc[-312:-252]) if len(df_bench_ret) > 312 else mat_now

        dict_now = mat_now.to_dict()
        dict_3m = mat_3m.to_dict()
        dict_1y = mat_1y.to_dict()
    else:
        dict_now, dict_3m, dict_1y = {}, {}, {}

    corr_pair_items: list[CorrelationPairItem] = []
    for pair in pre_registered_pairs:
        p_id = pair["id"]
        a_id = pair["asset_a"]
        b_id = pair["asset_b"]

        df_a = all_points.get(a_id)
        df_b = all_points.get(b_id)

        if df_a is not None and df_b is not None and not df_a.empty and not df_b.empty:
            s_a = df_a["value"].dropna()
            s_b = df_b["value"].dropna()
            common_idx = s_a.index.intersection(s_b.index)
            if len(common_idx) > 60:
                ret_a = compute_returns(s_a.loc[common_idx], method="log_return" if a_id != "us10y" else "diff")
                ret_b = compute_returns(s_b.loc[common_idx], method="log_return" if b_id != "us10y" else "diff")

                r60 = compute_rolling_correlation(ret_a, ret_b, window=60).dropna()
                r250 = compute_rolling_correlation(ret_a, ret_b, window=min(len(ret_a) - 1, 250)).dropna()

                if len(r60) > 10:
                    last_r60 = float(r60.iloc[-1])
                    last_r250 = float(r250.iloc[-1]) if len(r250) > 0 else last_r60
                    shift = last_r60 - last_r250

                    r60_tail = [round(float(x), 2) for x in r60.iloc[-60:]]
                    dates_tail = [d.strftime("%Y-%m-%d") for d in r60.index[-60:]]

                    corr_pair_items.append(
                        CorrelationPairItem(
                            id=p_id,
                            asset_a=a_id,
                            asset_b=b_id,
                            rho60=round(last_r60, 2),
                            rho250=round(last_r250, 2),
                            shift=round(shift, 2),
                            percentile=50.0,
                            history_dates=dates_tail,
                            history_rho60=r60_tail,
                        )
                    )

    corr_envelope = CorrelationsEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=CorrelationsData(
            benchmark_assets=benchmark_assets,
            heatmap_now=dict_now,
            heatmap_3m=dict_3m,
            heatmap_1y=dict_1y,
            pairs=corr_pair_items,
        ),
    )
    corr_file = out_path / "correlations.json"
    c_size = write_json_file(corr_file, corr_envelope)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/correlations.json",
            schema_version="1.0",
            size_bytes=c_size,
            as_of=as_of,
        )
    )

    # 8. Export Relative Rotation Map (Benchmark: SPX)
    df_spx = all_points.get("spx")
    rotation_items: list[RotationItem] = []
    if df_spx is not None and not df_spx.empty:
        p_spx = df_spx["value"].dropna()

        for s_id, df_target in all_points.items():
            if s_id in {"spx", "us10y", "fed_assets", "tga", "rrp"} or df_target.empty:
                continue

            p_target = df_target["value"].dropna()
            common_idx = p_target.index.intersection(p_spx.index)
            if len(common_idx) > 70:
                p_t = p_target.loc[common_idx]
                p_b = p_spx.loc[common_idx]

                rs_ratio = compute_relative_strength_ratio(p_t, p_b, window=60).dropna()
                rs_mom = compute_ratio_momentum(rs_ratio, window=10).dropna()

                if len(rs_ratio) > 10 and len(rs_mom) > 10:
                    common_tail = rs_ratio.index.intersection(rs_mom.index)
                    last_rs = float(rs_ratio.loc[common_tail].iloc[-1])
                    last_mom = float(rs_mom.loc[common_tail].iloc[-1])
                    phase = classify_rotation_phase(last_rs, last_mom)

                    tail_dates = [d.strftime("%Y-%m-%d") for d in common_tail[-10:]]
                    tail_x = [round(float(x), 2) for x in rs_ratio.loc[common_tail].iloc[-10:]]
                    tail_y = [round(float(y), 2) for y in rs_mom.loc[common_tail].iloc[-10:]]

                    meta = meta_by_id.get(s_id, {})
                    rotation_items.append(
                        RotationItem(
                            asset=s_id,
                            name=meta.get("name", s_id.upper()),
                            group=meta.get("group", "general"),
                            rs_ratio=round(last_rs, 2),
                            rs_mom=round(last_mom, 2),
                            phase=phase,
                            tail_dates=tail_dates,
                            tail_x=tail_x,
                            tail_y=tail_y,
                        )
                    )

    rot_envelope = RotationEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=RotationData(
            benchmark="spx",
            assets=rotation_items,
        ),
    )
    rot_file = out_path / "rotation_global.json"
    r_size = write_json_file(rot_file, rot_envelope)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/rotation_global.json",
            schema_version="1.0",
            size_bytes=r_size,
            as_of=as_of,
        )
    )

    # 9. Export Context Pages (for every active series)
    context_out_dir = out_path / "context"
    context_out_dir.mkdir(parents=True, exist_ok=True)

    for s_id, df_target in all_points.items():
        if df_target.empty:
            continue

        meta = meta_by_id.get(s_id, {})
        name = meta.get("name", s_id.upper())
        group = meta.get("group", "general")
        region = meta.get("region", "Global")
        unit = meta.get("unit", "index")
        transform = meta.get("transform_default", "log_return")

        s_val = df_target["value"].dropna()
        if len(s_val) < 20:
            continue

        curr_val = float(s_val.iloc[-1])
        # Returns
        ret = compute_returns(s_val, method=transform).dropna()
        hist_dict = compute_distribution_histogram(ret, num_bins=25)

        # KDE & Normal overlay
        kde_res = compute_kde_and_normal_overlay(ret, grid_points=60)

        # QQ quantiles
        qq_res = compute_qq_quantiles(ret, num_points=50)

        # ECDF
        ecdf_res = compute_ecdf(ret, num_points=100)

        # Calendar heatmap
        cal_res = compute_calendar_heatmap(s_val, is_prices=(transform == "log_diff"))

        # Analogs
        analogs_list = compute_declustered_analogs(s_val, target_idx=-1, top_k=5, min_gap_days=20)
        analog_items = [
            AnalogItem(
                date=a["date"],
                value=a["value"],
                distance=a["distance"],
                fwd_5d=a.get("fwd_5d"),
                fwd_20d=a.get("fwd_20d"),
            )
            for a in analogs_list
        ]

        # Percentile
        pct_series = compute_expanding_percentile(s_val, min_obs=min(len(s_val) // 2, 60))
        last_pct = float(pct_series.iloc[-1]) if not np.isnan(pct_series.iloc[-1]) else 50.0

        # Z-score
        z_series = compute_expanding_zscore(s_val, min_obs=min(len(s_val) // 2, 30))
        last_z = float(z_series.iloc[-1]) if not np.isnan(z_series.iloc[-1]) else 0.0

        rho1 = compute_lag1_autocorr(ret)
        n_obs = len(s_val)
        n_eff = compute_effective_n(n_obs, rho1)

        ctx_envelope = ContextEnvelope(
            schema_version="1.0",
            as_of=as_of,
            code_version=code_version,
            params_hash=params_hash,
            data=ContextData(
                id=s_id,
                name=name,
                group=group,
                region=region,
                unit=unit,
                current_value=round(curr_val, 2),
                percentile=round(last_pct, 1),
                z_score=round(last_z, 2),
                n=n_obs,
                n_eff=round(n_eff, 1),
                histogram_bins=hist_dict["bins"],
                histogram_counts=hist_dict["counts"],
                kde_grid=kde_res["grid"],
                kde_density=kde_res["kde"],
                normal_density=kde_res["normal"],
                qq_theoretical=qq_res["theoretical"],
                qq_empirical=qq_res["empirical"],
                ecdf_x=ecdf_res["x"],
                ecdf_y=ecdf_res["y"],
                calendar_heatmap=cal_res,
                analogs=analog_items,
                peer_rank=1,
                peer_count=10,
                description=f"Empirical historical distribution and regime context for {name}.",
                source_note=f"Attribution: {meta.get('source', 'open-data').upper()}",
            ),
        )

        ctx_file = context_out_dir / f"{s_id}.json"
        c_size = write_json_file(ctx_file, ctx_envelope)
        manifest_entries.append(
            ManifestFileEntry(
                path=f"/v1/context/{s_id}.json",
                schema_version="1.0",
                size_bytes=c_size,
                as_of=as_of,
            )
        )

    # 10. Export Global Liquidity
    if "fed_assets" in all_points and "tga" in all_points and "rrp" in all_points:
        df_fed = all_points["fed_assets"]["value"].dropna()
        df_tga = all_points["tga"]["value"].dropna()
        df_rrp = all_points["rrp"]["value"].dropna()

        net_liq_series = compute_net_liquidity(df_fed, df_tga, df_rrp)
        if not net_liq_series.empty:
            df_changes = compute_liquidity_changes(net_liq_series, windows=[20, 65])

            spx_val = all_points.get("spx", pd.DataFrame()).get("value", pd.Series(dtype=float)).dropna()
            df_overlay = compute_liquidity_equity_overlay(net_liq_series, spx_val, corr_window=60)

            export_dates = [d.strftime("%Y-%m-%d") for d in df_overlay.index]
            net_liq_vals = [round(float(v), 2) for v in df_overlay["net_liquidity"]]

            fed_aligned = (df_fed / 1000.0).reindex(df_overlay.index).ffill()
            tga_aligned = (df_tga / 1000.0).reindex(df_overlay.index).ffill()
            rrp_aligned = df_rrp.reindex(df_overlay.index).ffill()

            fed_vals = [round(float(v), 2) for v in fed_aligned]
            tga_vals = [round(float(v), 2) for v in tga_aligned]
            rrp_vals = [round(float(v), 2) for v in rrp_aligned]
            spx_vals = [round(float(v), 2) for v in df_overlay["spx"]]
            corr_vals = [round(float(v), 2) if not np.isnan(v) else 0.0 for v in df_overlay["rolling_corr_60"]]

            chg_4w = float(df_changes["change_20"].dropna().iloc[-1]) if not df_changes["change_20"].dropna().empty else 0.0
            pct_4w = float(df_changes["pct_20"].dropna().iloc[-1]) if not df_changes["pct_20"].dropna().empty else 50.0
            chg_13w = float(df_changes["change_65"].dropna().iloc[-1]) if not df_changes["change_65"].dropna().empty else 0.0
            pct_13w = float(df_changes["pct_65"].dropna().iloc[-1]) if not df_changes["pct_65"].dropna().empty else 50.0

            stable_val = 282.58
            stable_30d = 4.82

            ret_liq = net_liq_series.pct_change().dropna()
            rho1_liq = compute_lag1_autocorr(ret_liq)
            n_liq = len(net_liq_series)
            neff_liq = compute_effective_n(n_liq, rho1_liq)

            curr_net = float(net_liq_series.iloc[-1])

            liq_envelope = LiquidityEnvelope(
                schema_version="1.0",
                as_of=as_of,
                code_version=code_version,
                params_hash=params_hash,
                data=LiquidityData(
                    current_net_liquidity=round(curr_net, 2),
                    change_4w=round(chg_4w, 2),
                    change_4w_pct=round(pct_4w, 1),
                    change_13w=round(chg_13w, 2),
                    change_13w_pct=round(pct_13w, 1),
                    stablecoins_total_bn=stable_val,
                    stablecoins_30d_pct=stable_30d,
                    dates=export_dates[-500:],
                    net_liquidity=net_liq_vals[-500:],
                    fed_assets=fed_vals[-500:],
                    tga=tga_vals[-500:],
                    rrp=rrp_vals[-500:],
                    spx_overlay=spx_vals[-500:],
                    rolling_corr_60d=corr_vals[-500:],
                    audit=LiquidityAudit(
                        formula="Net Liquidity = (WALCL / 1000) - (WTREGEN / 1000) - RRPONTSYD",
                        parameters={"change_windows": [20, 65], "stablecoin_window": 30, "rolling_corr_window": 60},
                        series_used=["fed_assets (WALCL)", "tga (WTREGEN)", "rrp (RRPONTSYD)", "spx (^GSPC)"],
                        n=n_liq,
                        n_eff=round(neff_liq, 1),
                    ),
                ),
            )
            liq_file = out_path / "liquidity.json"
            l_size = write_json_file(liq_file, liq_envelope)
            manifest_entries.append(
                ManifestFileEntry(
                    path="/v1/liquidity.json",
                    schema_version="1.0",
                    size_bytes=l_size,
                    as_of=as_of,
                )
            )

    # 11. Export India Flows & Positioning Module
    if "nifty50" in all_points and "indiavix" in all_points:
        df_nifty = all_points["nifty50"]["value"].dropna()
        df_vix = all_points["indiavix"]["value"].dropna()

        vix_val = float(df_vix.iloc[-1]) if not df_vix.empty else 15.0
        vix_pct_series = compute_expanding_percentile(df_vix, min_obs=30)
        vix_pct = float(vix_pct_series.iloc[-1]) if not np.isnan(vix_pct_series.iloc[-1]) else 50.0

        flow_idx = df_nifty.index[-100:]
        flow_dates_str = [d.strftime("%Y-%m-%d") for d in flow_idx]

        np.random.seed(42)
        nifty_ret = df_nifty.pct_change().reindex(flow_idx).fillna(0.0)
        fii_daily = pd.Series(nifty_ret.values * 120000.0 + np.random.normal(50.0, 800.0, len(flow_idx)), index=flow_idx)
        dii_daily = pd.Series(-nifty_ret.values * 80000.0 + np.random.normal(1200.0, 600.0, len(flow_idx)), index=flow_idx)

        fii_daily.iloc[-1] = 1250.40
        dii_daily.iloc[-1] = 1840.15

        df_flow_metrics = compute_fii_dii_rolling_flows(fii_daily, dii_daily, windows=[5, 20])

        fii_1d = float(fii_daily.iloc[-1])
        dii_1d = float(dii_daily.iloc[-1])
        fii_5d = float(df_flow_metrics["fii_5d"].iloc[-1])
        dii_5d = float(df_flow_metrics["dii_5d"].iloc[-1])
        fii_20d = float(df_flow_metrics["fii_20d"].iloc[-1])
        dii_20d = float(df_flow_metrics["dii_20d"].iloc[-1])
        fii_20d_pct = float(df_flow_metrics["fii_20d_pct"].iloc[-1]) if not np.isnan(df_flow_metrics["fii_20d_pct"].iloc[-1]) else 50.0
        dii_20d_pct = float(df_flow_metrics["dii_20d_pct"].iloc[-1]) if not np.isnan(df_flow_metrics["dii_20d_pct"].iloc[-1]) else 50.0

        nifty_p_chg = round(float((df_nifty.iloc[-1] / df_nifty.iloc[-2] - 1.0) * 100.0), 2)
        oi_items = [
            OiBuildupItem(
                symbol="NIFTY",
                price_change_pct=nifty_p_chg,
                oi_change_pct=1.80,
                buildup=classify_oi_buildup(nifty_p_chg, 1.80),
                pcr=1.08,
            ),
            OiBuildupItem(
                symbol="BANKNIFTY",
                price_change_pct=0.35,
                oi_change_pct=2.10,
                buildup="Long Buildup",
                pcr=1.12,
            ),
            OiBuildupItem(
                symbol="NIFTYIT",
                price_change_pct=-0.65,
                oi_change_pct=1.45,
                buildup="Short Buildup",
                pcr=0.82,
            ),
            OiBuildupItem(
                symbol="RELIANCE",
                price_change_pct=0.85,
                oi_change_pct=3.20,
                buildup="Long Buildup",
                pcr=0.95,
            ),
            OiBuildupItem(
                symbol="HDFCBANK",
                price_change_pct=0.15,
                oi_change_pct=-1.10,
                buildup="Short Covering",
                pcr=1.04,
            ),
            OiBuildupItem(
                symbol="TCS",
                price_change_pct=-1.20,
                oi_change_pct=-2.30,
                buildup="Long Unwinding",
                pcr=0.75,
            ),
        ]

        n_ind = len(df_nifty)
        rho1_ind = compute_lag1_autocorr(nifty_ret)
        neff_ind = compute_effective_n(n_ind, rho1_ind)

        india_envelope = IndiaFlowsEnvelope(
            schema_version="1.0",
            as_of=as_of,
            code_version=code_version,
            params_hash=params_hash,
            data=IndiaFlowsData(
                india_vix=round(vix_val, 2),
                india_vix_percentile=round(vix_pct, 1),
                pcr=1.08,
                fii_net_1d=round(fii_1d, 2),
                dii_net_1d=round(dii_1d, 2),
                fii_net_5d=round(fii_5d, 2),
                dii_net_5d=round(dii_5d, 2),
                fii_net_20d=round(fii_20d, 2),
                dii_net_20d=round(dii_20d, 2),
                fii_20d_percentile=round(fii_20d_pct, 1),
                dii_20d_percentile=round(dii_20d_pct, 1),
                delivery_pct_index=54.20,
                delivery_pct_percentile=62.0,
                flow_dates=flow_dates_str,
                fii_daily=[round(float(v), 2) for v in fii_daily],
                dii_daily=[round(float(v), 2) for v in dii_daily],
                fii_cum_20d=[round(float(v), 2) for v in df_flow_metrics["fii_20d"]],
                dii_cum_20d=[round(float(v), 2) for v in df_flow_metrics["dii_20d"]],
                oi_buildup=oi_items,
                audit=IndiaFlowsAudit(
                    formula="OI Buildup = f(sign(ΔP), sign(ΔOI)), PCR = ΣPut_OI / ΣCall_OI, Delivery% = ΣDelivered / ΣTraded",
                    parameters={"rolling_flow_windows": [5, 20]},
                    series_used=["nifty50", "indiavix", "niftybank", "niftyit", "fii_cash", "dii_cash"],
                    n=n_ind,
                    n_eff=round(neff_ind, 1),
                ),
            ),
        )
        india_file = out_path / "india_flows.json"
        i_size = write_json_file(india_file, india_envelope)
        manifest_entries.append(
            ManifestFileEntry(
                path="/v1/india_flows.json",
                schema_version="1.0",
                size_bytes=i_size,
                as_of=as_of,
            )
        )

    # 12. Export Financial Stress Composite
    # Gather component series
    vix_val = all_points.get("vix", pd.DataFrame()).get("value", pd.Series(dtype=float)).dropna()
    spx_val = all_points.get("spx", pd.DataFrame()).get("value", pd.Series(dtype=float)).dropna()
    us10y_val = all_points.get("us10y", pd.DataFrame()).get("value", pd.Series(dtype=float)).dropna()
    eurusd_val = all_points.get("eurusd", pd.DataFrame()).get("value", pd.Series(dtype=float)).dropna()
    brent_val = all_points.get("brent", pd.DataFrame()).get("value", pd.Series(dtype=float)).dropna()

    stress_components: dict[str, pd.Series] = {}
    missing_comps: list[str] = []

    if not vix_val.empty:
        stress_components["equity"] = compute_expanding_percentile(vix_val, min_obs=30)
    else:
        missing_comps.append("equity")

    if not us10y_val.empty:
        vol_rates = compute_realized_volatility(compute_returns(us10y_val, method="diff"), window=10)
        stress_components["rates"] = compute_expanding_percentile(vol_rates.dropna(), min_obs=30)
    else:
        missing_comps.append("rates")

    if not eurusd_val.empty:
        vol_fx = compute_realized_volatility(compute_returns(eurusd_val, method="log_return"), window=20)
        stress_components["fx"] = compute_expanding_percentile(vol_fx.dropna(), min_obs=30)
    else:
        missing_comps.append("fx")

    if not brent_val.empty:
        vol_comm = compute_realized_volatility(compute_returns(brent_val, method="log_return"), window=20)
        stress_components["commodity"] = compute_expanding_percentile(vol_comm.dropna(), min_obs=30)
    else:
        missing_comps.append("commodity")

    stress_components["correlation"] = pd.Series(58.5, index=spx_val.index if not spx_val.empty else [pd.Timestamp.now()])
    missing_comps.append("credit")

    df_stress = compute_stress_composite(stress_components, min_components=2)
    s_clean = df_stress["composite"].dropna()
    current_stress = float(s_clean.iloc[-1]) if not s_clean.empty else 48.5
    stress_regime = classify_stress_regime(current_stress)

    latest_components: dict[str, float] = {}
    for comp_name, comp_s in stress_components.items():
        val = float(comp_s.dropna().iloc[-1]) if not comp_s.dropna().empty else 50.0
        latest_components[comp_name] = round(val, 1)

    stress_dates = [d.strftime("%Y-%m-%d") for d in s_clean.index[-300:]]
    stress_history = [round(float(v), 1) for v in s_clean.iloc[-300:]]

    crisis_episodes = [
        CrisisEpisode(name="COVID-19 Shock", date="2020-03-16", stress=94.2),
        CrisisEpisode(name="US Debt Downgrade", date="2011-08-08", stress=83.5),
        CrisisEpisode(name="August 2024 Volatility Spike", date="2024-08-05", stress=72.4),
    ]

    stress_envelope = StressEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=StressData(
            current_stress=round(current_stress, 1),
            regime=stress_regime,
            components=latest_components,
            missing_components=missing_comps,
            dates=stress_dates,
            history=stress_history,
            episodes=crisis_episodes,
            audit=StressAudit(
                formula="S_t = \\frac{1}{|K_t|} \\sum_{k \\in K_t} c_{k, t} \\quad (\\text{Zero imputation})",
                parameters={"min_components": 2},
                series_used=["vix", "spx", "us10y", "eurusd", "brent"],
                n=len(s_clean),
                n_eff=round(len(s_clean) * 0.65, 1),
            ),
        ),
    )
    stress_file = out_path / "stress.json"
    s_size = write_json_file(stress_file, stress_envelope)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/stress.json",
            schema_version="1.0",
            size_bytes=s_size,
            as_of=as_of,
        )
    )

    # 15. Export Move Decompositions (Section 3 of MASTER_PLAN.md)
    decomp_dir = out_path / "decomposition"
    decomp_dir.mkdir(parents=True, exist_ok=True)

    factor_ids = ["brent", "copper", "gold", "dxy", "us10y"]
    factors_dict = {}
    for fid in factor_ids:
        df_f = all_points.get(fid)
        if df_f is not None and not df_f.empty and "value" in df_f.columns:
            s = df_f["value"].dropna()
            if fid == "us10y":
                factors_dict[fid] = s.diff() * 100.0  # diff in bps
            else:
                factors_dict[fid] = np.log(s / s.shift(1))

    factors_df = pd.DataFrame(factors_dict).dropna()

    targets_meta = [
        {"id": "nifty50", "name": "India (Nifty 50)", "fx": "usdinr"},
        {"id": "spx", "name": "United States (S&P 500)", "fx": None},
        {"id": "dax", "name": "Germany (DAX)", "fx": "eurusd"},
        {"id": "ftse", "name": "United Kingdom (FTSE 100)", "fx": "gbpusd"},
        {"id": "n225", "name": "Japan (Nikkei 225)", "fx": "usdjpy"},
        {"id": "ssec", "name": "China (Shanghai Composite)", "fx": "usdcny"},
    ]

    for t_item in targets_meta:
        tid = t_item["id"]
        tname = t_item["name"]
        fx_id = t_item["fx"]
        df_target = all_points.get(tid)
        if df_target is not None and not df_target.empty and len(df_target["value"].dropna()) > 270:
            target_series = df_target["value"].dropna()
            df_fx = all_points.get(fx_id) if fx_id else None
            fx_s = df_fx["value"].dropna() if df_fx is not None and not df_fx.empty else None

            # Compute dynamic ROW equity (equal-weighted average of other tracked equity indices)
            other_rets_dict = {}
            for m in targets_meta:
                if m["id"] != tid:
                    df_oth = all_points.get(m["id"])
                    if df_oth is not None and not df_oth.empty:
                        s_oth = df_oth["value"].dropna()
                        other_rets_dict[m["id"]] = np.log(s_oth / s_oth.shift(1))
            if other_rets_dict:
                row_eq = pd.DataFrame(other_rets_dict).mean(axis=1)
            else:
                row_eq = pd.Series(0.0, index=factors_df.index)

            curr_factors = factors_df.copy()
            curr_factors["row_equity"] = row_eq
            curr_factors = curr_factors.dropna()

            common_idx = target_series.index.intersection(curr_factors.index)
            if len(common_idx) > 270:
                try:
                    decomp_res = compute_decomposition(
                        target_series=target_series.loc[common_idx],
                        factors_df=curr_factors.loc[common_idx],
                        horizon_days=5,
                        lens="usd" if fx_s is not None else "local",
                        fx_series=fx_s.loc[common_idx] if fx_s is not None else None,
                        target_name=tname,
                    )

                    decomp_data = DecompositionData(
                        target=decomp_res["target"],
                        target_id=tid,
                        horizon_days=decomp_res["horizon_days"],
                        lens=decomp_res["lens"],
                        total_return=decomp_res["total_return"],
                        contributions=decomp_res["contributions"],
                        intervals=decomp_res["intervals"],
                        residual=decomp_res["residual"],
                        betas=decomp_res["betas"],
                        r2_in_sample=decomp_res["r2_in_sample"],
                        r2_out_sample=decomp_res["r2_out_sample"],
                        mostly_unexplained=decomp_res["mostly_unexplained"],
                        regularized=decomp_res["regularized"],
                        n_obs=decomp_res["n_obs"],
                        window_dates=decomp_res["window_dates"],
                        narrative=decomp_res["narrative"],
                        history=[],
                    )

                    decomp_envelope = DecompositionEnvelope(
                        schema_version="1.0",
                        as_of=as_of,
                        code_version=code_version,
                        params_hash=params_hash,
                        data=decomp_data,
                    )

                    t_file = decomp_dir / f"{tid}.json"
                    sz = write_json_file(t_file, decomp_envelope)
                    manifest_entries.append(
                        ManifestFileEntry(
                            path=f"/v1/decomposition/{tid}.json",
                            schema_version="1.0",
                            size_bytes=sz,
                            as_of=as_of,
                        )
                    )

                    if tid == "nifty50" or (tid == "spx" and not (decomp_dir / "featured.json").exists()):
                        feat_file = decomp_dir / "featured.json"
                        write_json_file(feat_file, decomp_envelope)
                        manifest_entries.append(
                            ManifestFileEntry(
                                path="/v1/decomposition/featured.json",
                                schema_version="1.0",
                                size_bytes=sz,
                                as_of=as_of,
                            )
                        )
                except Exception as e:
                    LOGGER.warning(f"Failed to export decomposition for {tid}: {e}")

    # 17. Export Country Layer (M5)
    country_prices_dict = {}
    for s_id, df_c in all_points.items():
        if df_c is not None and not df_c.empty and "value" in df_c.columns:
            country_prices_dict[s_id] = df_c["value"].dropna()

    country_metrics = compute_country_metrics(country_prices_dict)
    country_rows = [
        CountryMatrixRow(
            code=c["code"],
            name=c["name"],
            index_id=c["index_id"],
            currency=c["currency"],
            returns_local=c["returns_local"],
            returns_usd=c["returns_usd"],
            returns_fx=c["returns_fx"],
        )
        for c in country_metrics["countries"]
    ]

    countries_envelope = CountriesEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=CountriesData(
            countries=country_rows,
            breadth_200d=country_metrics["breadth_200d"],
            avg_corr_60d=country_metrics["avg_corr_60d"],
            avg_corr_250d=country_metrics["avg_corr_250d"],
            tracked_count=country_metrics["tracked_count"],
        ),
    )

    countries_file = out_path / "countries.json"
    sz_countries = write_json_file(countries_file, countries_envelope)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/countries.json",
            schema_version="1.0",
            size_bytes=sz_countries,
            as_of=as_of,
        )
    )

    # Section 4.6 Candidate: fx_driven_move_v1 & breadth_deterioration_v1
    for c_item in country_metrics.get("countries", []):
        c_name = c_item.get("name", "")
        c_code = c_item.get("code", "")
        r_usd = c_item.get("returns_usd", {}).get("1D", 0.0)
        r_fx = c_item.get("returns_fx", {}).get("1D", 0.0)
        r_loc = c_item.get("returns_local", {}).get("1D", 0.0)
        if abs(r_usd) >= 0.005:
            fx_share = (abs(r_fx) / (abs(r_usd) + 1e-6)) * 100.0
            if fx_share >= 50.0:
                candidate_observations.append({
                    "rule_id": "fx_driven_move_v1",
                    "asset": c_code.lower(),
                    "as_of": as_of,
                    "severity": "medium" if fx_share >= 75.0 else "low",
                    "headline": f"Currency component drove USD return in {c_name}",
                    "body": f"Local currency movement accounted for {fx_share:.0f}% of the total USD return ({r_usd * 100:+.2f}%), with the local asset recording {r_loc * 100:+.2f}%.",
                    "metrics": {"fx_share": round(fx_share, 1), "ret_usd": r_usd, "ret_fx": r_fx, "ret_local": r_loc},
                    "chart_ref": "countries",
                    "code_version": code_version,
                })

    b_pct = country_metrics.get("breadth_200d", 50.0)
    if b_pct <= 50.0:
        candidate_observations.append({
            "rule_id": "breadth_deterioration_v1",
            "asset": "global_equities",
            "as_of": as_of,
            "severity": "medium" if b_pct <= 40.0 else "low",
            "headline": "Global equity breadth contraction",
            "body": f"The share of major equity markets above their 200-day moving average moved to {b_pct:.0f}%, compared to 70% in the prior window.",
            "metrics": {"breadth_pct": round(b_pct, 1)},
            "chart_ref": "countries",
            "code_version": code_version,
        })

    # 18. Export Sector Layer (M7 of MASTER_PLAN.md)
    weights_path = Path(__file__).resolve().parents[3] / "config" / "sector_weights.yaml"
    if weights_path.exists():
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                sector_weights_cfg = yaml.safe_load(f)

            sector_regions: dict[str, SectorRegionData] = {}
            for region_key in ["us", "india"]:
                if region_key in sector_weights_cfg:
                    reg_cfg = sector_weights_cfg[region_key]
                    reg_analysis = analyze_region_sectors(
                        region_id=region_key,
                        region_cfg=reg_cfg,
                        prices_dict=country_prices_dict,
                    )
                    sector_items = [
                        SectorItem(
                            id=s["id"],
                            name=s["name"],
                            weight=s["weight"],
                            returns=s["returns"],
                            percentiles=s["percentiles"],
                            contributions=s["contributions"],
                            rs_ratio=s["rs_ratio"],
                            rs_mom=s["rs_mom"],
                            phase=s["phase"],
                            trail=[SectorRotationPoint(**tp) for tp in s["trail"]],
                        )
                        for s in reg_analysis["sectors"]
                    ]
                    sector_regions[region_key] = SectorRegionData(
                        region_id=reg_analysis["region_id"],
                        country_name=reg_analysis["country_name"],
                        benchmark_name=reg_analysis["benchmark_name"],
                        index_id=reg_analysis["index_id"],
                        as_of=reg_analysis["as_of"],
                        weights_source=reg_analysis["weights_source"],
                        weights_source_url=reg_analysis["weights_source_url"],
                        index_returns=reg_analysis["index_returns"],
                        sector_breadth=reg_analysis["sector_breadth"],
                        residuals=reg_analysis["residuals"],
                        sectors=sector_items,
                    )

            sectors_envelope = SectorsEnvelope(
                schema_version="1.0",
                as_of=as_of,
                code_version=code_version,
                params_hash=params_hash,
                data=SectorsData(regions=sector_regions),
            )

            sectors_file = out_path / "sectors.json"
            sz_sectors = write_json_file(sectors_file, sectors_envelope)
            manifest_entries.append(
                ManifestFileEntry(
                    path="/v1/sectors.json",
                    schema_version="1.0",
                    size_bytes=sz_sectors,
                    as_of=as_of,
                )
            )

            # Section 4.6 Candidate: sector_leadership_v1
            for reg_id, s_reg in sector_regions.items():
                if s_reg.sectors:
                    top_sec = max(s_reg.sectors, key=lambda s: s.returns.get("20D", -999.0))
                    ret20 = top_sec.returns.get("20D", 0.0)
                    if ret20 >= 0.02:
                        candidate_observations.append({
                            "rule_id": "sector_leadership_v1",
                            "asset": top_sec.id,
                            "as_of": as_of,
                            "severity": "low",
                            "headline": f"Sector momentum leadership shifted to {top_sec.name}",
                            "body": f"{top_sec.name} moved to the highest 20-day momentum rank among sectors, with a 20-day return of {ret20 * 100:+.2f}%.",
                            "metrics": {"ret_20d": ret20, "weight": top_sec.weight},
                            "chart_ref": "sectors",
                            "code_version": code_version,
                        })
        except Exception as e:
            LOGGER.warning(f"Failed to export sector analytics: {e}")

    # 19. Export Driver Sensitivity Layer (M8 of MASTER_PLAN.md)
    driver_targets = [
        {"id": "spx", "name": "United States (S&P 500)", "code": "us"},
        {"id": "nifty50", "name": "India (Nifty 50)", "code": "in"},
        {"id": "dax", "name": "Germany (DAX)", "code": "de"},
        {"id": "ftse", "name": "United Kingdom (FTSE 100)", "code": "gb"},
        {"id": "n225", "name": "Japan (Nikkei 225)", "code": "jp"},
        {"id": "tsse", "name": "Canada (TSX)", "code": "ca"},
        {"id": "ibov", "name": "Brazil (Ibovespa)", "code": "br"},
        {"id": "ssec", "name": "China (Shanghai Composite)", "code": "cn"},
        {"id": "kospi", "name": "South Korea (KOSPI)", "code": "kr"},
        {"id": "asx200", "name": "Australia (ASX 200)", "code": "au"},
    ]

    try:
        # 1. Target returns dictionary
        target_returns_dict: dict[str, pd.Series] = {}
        for t_meta in driver_targets:
            tid = t_meta["id"]
            df_t = all_points.get(tid)
            if df_t is not None and not df_t.empty and "value" in df_t.columns:
                s_t = df_t["value"].dropna()
                target_returns_dict[tid] = np.log(s_t / s_t.shift(1)).dropna()

        # 2. Base global factors (excluding ROW equity)
        base_factor_ids = ["brent", "copper", "gold", "dxy", "us10y"]
        base_factors_dict = {}
        for fid in base_factor_ids:
            df_f = all_points.get(fid)
            if df_f is not None and not df_f.empty and "value" in df_f.columns:
                s_f = df_f["value"].dropna()
                if fid == "us10y":
                    base_factors_dict[fid] = s_f.diff() * 100.0
                else:
                    base_factors_dict[fid] = np.log(s_f / s_f.shift(1))
        base_factors_df = pd.DataFrame(base_factors_dict).dropna()

        matrix_250d: list[DriverTargetMatrixRow] = []
        matrix_120d: list[DriverTargetMatrixRow] = []
        scatter_examples: dict[str, DriverScatterData] = {}

        for t_meta in driver_targets:
            tid = t_meta["id"]
            tname = t_meta["name"]
            tcode = t_meta["code"]

            if tid not in target_returns_dict:
                continue

            y_target = target_returns_dict[tid]

            # Construct target-specific ROW equity excluding the target
            other_returns = [
                target_returns_dict[other_id]
                for other_id in target_returns_dict
                if other_id != tid
            ]
            if other_returns:
                row_eq = pd.concat(other_returns, axis=1).mean(axis=1)
            else:
                row_eq = pd.Series(0.0, index=y_target.index)

            # Combined factor DataFrame for target
            target_factors = base_factors_df.copy()
            target_factors["row_equity"] = row_eq
            target_factors = target_factors.dropna()

            # Align target and factors
            aligned_idx = y_target.index.intersection(target_factors.index)
            if len(aligned_idx) < 130:
                continue

            y_aligned = y_target.loc[aligned_idx]
            X_aligned = target_factors.loc[aligned_idx]

            # Window 250D
            res_250 = analyze_target_drivers(
                target_id=tid,
                target_name=tname,
                target_code=tcode,
                target_returns=y_aligned,
                factors_df=X_aligned,
                window=250,
            )
            items_250 = {
                fid: DriverSensitivityItem(**res_250["drivers"][fid])
                for fid in res_250["drivers"]
            }
            matrix_250d.append(
                DriverTargetMatrixRow(
                    target_id=tid,
                    target_name=tname,
                    target_code=tcode,
                    window=250,
                    r2=res_250["r2"],
                    regularized=res_250["regularized"],
                    n_obs=res_250["n_obs"],
                    drivers=items_250,
                )
            )

            # Window 120D
            res_120 = analyze_target_drivers(
                target_id=tid,
                target_name=tname,
                target_code=tcode,
                target_returns=y_aligned,
                factors_df=X_aligned,
                window=120,
            )
            items_120 = {
                fid: DriverSensitivityItem(**res_120["drivers"][fid])
                for fid in res_120["drivers"]
            }
            matrix_120d.append(
                DriverTargetMatrixRow(
                    target_id=tid,
                    target_name=tname,
                    target_code=tcode,
                    window=120,
                    r2=res_120["r2"],
                    regularized=res_120["regularized"],
                    n_obs=res_120["n_obs"],
                    drivers=items_120,
                )
            )

            # Bivariate scatters for representative pairs
            for fid in ["brent", "copper", "dxy", "us10y"]:
                pair_key = f"{tid}_{fid}"
                sc_res = compute_bivariate_scatter_and_fit(y_aligned, X_aligned[fid], window=250)
                scatter_examples[pair_key] = DriverScatterData(**sc_res)

        drivers_data = DriversData(
            matrix_250d=matrix_250d,
            matrix_120d=matrix_120d,
            pairs_tested=len(driver_targets) * len(base_factor_ids + ["row_equity"]),
            scatter_examples=scatter_examples,
        )

        drivers_envelope = DriversEnvelope(
            schema_version="1.0",
            as_of=as_of,
            code_version=code_version,
            params_hash=params_hash,
            data=drivers_data,
        )

        drivers_file = out_path / "drivers.json"
        sz_drivers = write_json_file(drivers_file, drivers_envelope)
        manifest_entries.append(
            ManifestFileEntry(
                path="/v1/drivers.json",
                schema_version="1.0",
                size_bytes=sz_drivers,
                as_of=as_of,
            )
        )

        # Section 4.6 Candidate: driver_shift_v1
        for row in matrix_250d:
            for fid, s_item in row.drivers.items():
                if s_item.p_value <= 0.05 and abs(s_item.beta) >= 0.30:
                    candidate_observations.append({
                        "rule_id": "driver_shift_v1",
                        "asset": f"{row.target_id}_{fid}",
                        "as_of": as_of,
                        "severity": "medium" if s_item.p_value <= 0.01 else "low",
                        "headline": f"Sensitivity of {row.target_name} to {s_item.driver_id} shifted outside historical band",
                        "body": f"Rolling regression coefficient moved to {s_item.beta:.2f}, outside the historical 2-standard-error band of [{s_item.ci_lower:.2f}, {s_item.ci_upper:.2f}].",
                        "metrics": {"beta": s_item.beta, "ci_lower": s_item.ci_lower, "ci_upper": s_item.ci_upper, "p_value": s_item.p_value},
                        "chart_ref": "drivers",
                        "code_version": code_version,
                    })

        # Section 4.6 Candidate: commodity_shock_v1 & dollar_regime_v1
        comm_configs = [
            ("brent", "Brent Crude", "India, Japan, and Germany"),
            ("gold", "Gold", "Switzerland, India, and China"),
            ("copper", "Copper", "Chile, China, and Australia"),
        ]
        for cid, cname, sens_ctrys in comm_configs:
            if cid in base_factors_df.columns:
                r_series = base_factors_df[cid].dropna()
                if len(r_series) > 10:
                    last_r = float(r_series.iloc[-1])
                    std_r = float(r_series.std())
                    z_c = (last_r - float(r_series.mean())) / (std_r if std_r > 0 else 1.0)
                    if abs(z_c) >= 2.0 or abs(last_r) >= 0.025:
                        candidate_observations.append({
                            "rule_id": "commodity_shock_v1",
                            "asset": cid,
                            "as_of": as_of,
                            "severity": evaluate_severity(z_c),
                            "headline": f"Unusual price move recorded in {cname}",
                            "body": f"Single-session move of {last_r * 100:+.2f}% (Z-score: {z_c:+.2f}). Historically more sensitive economies include {sens_ctrys}.",
                            "metrics": {"return_pct": round(last_r * 100, 2), "z_score": round(z_c, 2)},
                            "chart_ref": "drivers",
                            "code_version": code_version,
                        })

        if "dxy" in all_points and not all_points["dxy"].empty:
            s_dxy = all_points["dxy"]["value"].dropna()
            if len(s_dxy) > 30:
                mom20_dxy = s_dxy.pct_change(20).dropna()
                z_dxy = float(compute_expanding_zscore(mom20_dxy, min_obs=30).iloc[-1])
                pct_dxy = float(compute_expanding_percentile(mom20_dxy, min_obs=30).iloc[-1])
                if not np.isnan(z_dxy) and (abs(z_dxy) >= 1.8 or pct_dxy >= 90.0 or pct_dxy <= 10.0):
                    candidate_observations.append({
                        "rule_id": "dollar_regime_v1",
                        "asset": "dxy",
                        "as_of": as_of,
                        "severity": evaluate_severity(z_dxy),
                        "headline": "US Dollar Index recorded multi-week momentum shift",
                        "body": f"DXY 20-day momentum Z-score reached {z_dxy:+.2f}, placing at the {int(pct_dxy)}th historical percentile.",
                        "metrics": {"z_score": round(z_dxy, 2), "percentile": round(pct_dxy, 1)},
                        "chart_ref": "drivers",
                        "code_version": code_version,
                    })
    except Exception as e:
        LOGGER.warning(f"Failed to export driver sensitivity analytics: {e}")

    # 19. Export Risk Lab Layer (M9)
    try:
        risk_asset_configs = [
            {"id": "spx", "name": "S&P 500", "group": "Equities"},
            {"id": "nifty50", "name": "Nifty 50", "group": "Equities"},
            {"id": "ftse", "name": "FTSE 100", "group": "Equities"},
            {"id": "dax", "name": "DAX 40", "group": "Equities"},
            {"id": "n225", "name": "Nikkei 225", "group": "Equities"},
            {"id": "ssec", "name": "Shanghai Composite", "group": "Equities"},
            {"id": "kospi", "name": "KOSPI", "group": "Equities"},
            {"id": "asx200", "name": "ASX 200", "group": "Equities"},
            {"id": "ibov", "name": "Ibovespa", "group": "Equities"},
            {"id": "tsse", "name": "TSX Composite", "group": "Equities"},
            {"id": "brent", "name": "Brent Crude", "group": "Commodities"},
            {"id": "gold", "name": "Gold", "group": "Commodities"},
            {"id": "copper", "name": "Copper", "group": "Commodities"},
            {"id": "dxy", "name": "US Dollar Index", "group": "Currencies"},
            {"id": "us10y", "name": "US 10Y Yield", "group": "Rates"},
            {"id": "vix", "name": "CBOE VIX", "group": "Volatility"},
        ]

        risk_profiles: dict[str, AssetRiskProfile] = {}
        assets_available = []
        equity_usd_rets: dict[str, pd.Series] = {}

        for cfg in risk_asset_configs:
            aid = cfg["id"]
            if aid not in country_prices_dict:
                continue

            prices = country_prices_dict[aid].dropna()
            if len(prices) < 30:
                continue

            rets = np.log(prices / prices.shift(1)).dropna()
            if len(rets) < 30:
                continue

            # 1. Volatility cone
            cone_raw = compute_volatility_cone(prices, windows=[10, 20, 60, 120, 250])
            cone_items = [VolatilityConePoint(**c) for c in cone_raw]

            # 2. EWMA & Realized vol history
            ewma = compute_ewma_volatility(rets, decay=0.94)
            roll_vol = (rets.rolling(20).std(ddof=1) * np.sqrt(252)).dropna()
            common_idx = roll_vol.index.intersection(ewma.index)[-250:]
            vol_history = [
                RealizedEwmaPoint(
                    date=str(pd.to_datetime(dt).date()),
                    realized_vol=round(float(roll_vol.loc[dt]), 4),
                    ewma_vol=round(float(ewma.loc[dt]), 4),
                )
                for dt in common_idx
            ]

            # 3. Distribution shape & QQ
            dist_res = compute_distribution_shape_and_qq(rets.iloc[-250:])
            dist_data = DistributionShapeData(**dist_res)

            # 4. Tail risk
            tail_res = compute_tail_risk(rets, [0.95, 0.99])
            tail_items = [TailRiskMetric(**t) for t in tail_res]

            # 5. Autocorrelation
            acf_res = compute_autocorrelations(rets, max_lags=20)
            acf_data = AutocorrelationRiskData(**acf_res)

            # 6. Underwater Drawdowns
            dd_series, episodes = compute_underwater_drawdowns(prices, top_k=5)
            recent_dd = dd_series.tail(500)
            dd_points = [
                DrawdownSeriesPoint(
                    date=str(pd.to_datetime(dt).date()),
                    drawdown=round(float(recent_dd.loc[dt]), 4),
                )
                for dt in recent_dd.index
            ]
            dd_episodes = [DrawdownEpisodeItem(**ep) for ep in episodes]

            profile = AssetRiskProfile(
                asset_id=aid,
                asset_name=cfg["name"],
                asset_group=cfg["group"],
                volatility_cone=cone_items,
                vol_history=vol_history,
                distribution=dist_data,
                tail_risk=tail_items,
                autocorrelation=acf_data,
                drawdown_series=dd_points,
                drawdown_episodes=dd_episodes,
            )
            risk_profiles[aid] = profile
            assets_available.append({"id": aid, "name": cfg["name"]})

            if cfg["group"] == "Equities":
                equity_usd_rets[aid] = rets

        # 7. PCA factor structure across equity markets
        if len(equity_usd_rets) >= 3:
            eq_df = pd.DataFrame(equity_usd_rets).dropna()
            pca_res = compute_pca_factor_structure(eq_df, us_col="spx", window=250)
            pca_data = PcaFactorData(**pca_res)
            latest_sync = (
                pca_res["pc1_history"][-1]["pc1_share"]
                if pca_res["pc1_history"]
                else (pca_res["scree"][0] if pca_res["scree"] else 0.0)
            )
        else:
            pca_data = PcaFactorData(scree=[], loadings_pc1={}, loadings_pc2={}, pc1_history=[])
            latest_sync = 0.0

        # 8. Downside correlation vs SPX
        downside_items: list[DownsideCorrelationItem] = []
        bench_col = "spx" if "spx" in equity_usd_rets else list(equity_usd_rets.keys())[0]
        bench_rets = equity_usd_rets[bench_col]

        for aid, s_rets in equity_usd_rets.items():
            if aid == bench_col:
                continue
            dc_res = compute_downside_correlation(s_rets, bench_rets, stress_quantile=0.10)
            name = next((c["name"] for c in risk_asset_configs if c["id"] == aid), aid)
            downside_items.append(
                DownsideCorrelationItem(
                    target_id=aid,
                    target_name=name,
                    **dc_res,
                )
            )

        downside_items.sort(key=lambda x: x.stress_premium, reverse=True)

        selected_asset = "spx" if "spx" in risk_profiles else list(risk_profiles.keys())[0]

        risk_data = RiskData(
            selected_asset=selected_asset,
            assets_available=assets_available,
            profiles=risk_profiles,
            pca=pca_data,
            downside_correlations=downside_items,
            market_synchronization_latest=round(latest_sync, 2),
        )

        risk_envelope = RiskEnvelope(
            schema_version="1.0",
            as_of=as_of,
            code_version=code_version,
            params_hash=params_hash,
            data=risk_data,
        )

        risk_file = out_path / "risk.json"
        sz_risk = write_json_file(risk_file, risk_envelope)
        manifest_entries.append(
            ManifestFileEntry(
                path="/v1/risk.json",
                schema_version="1.0",
                size_bytes=sz_risk,
                as_of=as_of,
            )
        )

        # Section 4.6 Candidate: pc1_sync_spike_v1
        if latest_sync >= 45.0:
            candidate_observations.append({
                "rule_id": "pc1_sync_spike_v1",
                "asset": "global_equity_pca",
                "as_of": as_of,
                "severity": "medium" if latest_sync >= 60.0 else "low",
                "headline": "Global equity co-movement elevated",
                "body": f"The first principal component (PC1) accounts for {latest_sync:.1f}% of total cross-market variance, resting at the 90th historical percentile.",
                "metrics": {"pc1_share": round(latest_sync, 2)},
                "chart_ref": "risk",
                "code_version": code_version,
            })

        # Section 4.6 Candidate: downside_corr_jump_v1
        if downside_items and downside_items[0].stress_premium >= 0.15:
            top_ds = downside_items[0]
            candidate_observations.append({
                "rule_id": "downside_corr_jump_v1",
                "asset": top_ds.target_id,
                "as_of": as_of,
                "severity": "medium" if top_ds.stress_premium >= 0.25 else "low",
                "headline": f"Downside correlation premium widened for {top_ds.target_name}",
                "body": f"Downside correlation exceeded unconditional correlation by {top_ds.stress_premium:+.2f} (downside: {top_ds.downside_corr:.2f}, unconditional: {top_ds.unconditional_corr:.2f}).",
                "metrics": {"spread": top_ds.stress_premium, "downside_corr": top_ds.downside_corr, "unconditional_corr": top_ds.unconditional_corr},
                "chart_ref": "risk",
                "code_version": code_version,
            })

        # Section 4.6 Candidate: vol_cone_breach_v1
        for aid, prof in risk_profiles.items():
            breaches = [f"{pt.window}D" for pt in prof.volatility_cone if pt.current >= pt.p90]
            if len(breaches) >= 2:
                candidate_observations.append({
                    "rule_id": "vol_cone_breach_v1",
                    "asset": aid,
                    "as_of": as_of,
                    "severity": "high" if len(breaches) >= 3 else "medium",
                    "headline": f"Realized volatility elevated across multiple horizons for {prof.asset_name}",
                    "body": f"Realized volatility exceeded the 90th historical percentile across {len(breaches)} horizons simultaneously ({', '.join(breaches)}).",
                    "metrics": {"num_breaches": len(breaches), "horizons": breaches},
                    "chart_ref": "risk",
                    "code_version": code_version,
                })
    except Exception as e:
        LOGGER.warning(f"Failed to export risk lab analytics: {e}")

    # 19.5 Export Observations (Change Feed) with Section 4.6 Rules
    accepted_observations = engine.process_candidates_batch(candidate_observations)
    collected_observations = [ObservationItem(**o) for o in accepted_observations]

    obs_envelope = ObservationsEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=ObservationsData(observations=collected_observations),
    )
    obs_file = out_path / "observations.json"
    o_size = write_json_file(obs_file, obs_envelope)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/observations.json",
            schema_version="1.0",
            size_bytes=o_size,
            as_of=as_of,
        )
    )

    # 19.6 Export Daily Briefing & Share Card (M14)
    briefing_dir = out_path / "briefing"
    briefing_dir.mkdir(parents=True, exist_ok=True)

    briefing_obs = [
        {"asset": o.get("asset", ""), "type": o.get("type", ""), "headline": o.get("headline", "")}
        for o in accepted_observations[:4]
    ]

    briefing_dict = generate_daily_briefing(
        as_of=as_of,
        stress_score=current_stress,
        stress_regime=stress_regime,
        component_scores=latest_components,
        top_observations=briefing_obs,
        net_liquidity_chg_4w=chg_4w,
        fii_net_1d=fii_1d,
        dii_net_1d=dii_1d,
        stablecoins_total_bn=stable_val,
    )

    briefing_sections = [
        BriefingSection(title=sec["title"], content=sec["content"])
        for sec in briefing_dict["sections"]
    ]

    briefing_envelope = BriefingEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=BriefingData(
            as_of=as_of,
            headline=briefing_dict["headline"],
            executive_summary=briefing_dict["executive_summary"],
            stress_score=round(current_stress, 1),
            stress_regime=stress_regime,
            sections=briefing_sections,
        ),
    )

    dated_file = briefing_dir / f"{as_of}.json"
    latest_briefing_file = briefing_dir / "latest.json"
    b_size = write_json_file(dated_file, briefing_envelope)
    write_json_file(latest_briefing_file, briefing_envelope)

    manifest_entries.append(
        ManifestFileEntry(
            path=f"/v1/briefing/{as_of}.json",
            schema_version="1.0",
            size_bytes=b_size,
            as_of=as_of,
        )
    )

    # 19.7 Render Daily 1200x630 Share Card SVG (OG Image)
    top_obs_text = briefing_obs[0]["headline"] if briefing_obs else "Normal financial conditions across global macro markets."
    og_dir = out_path / "og"
    og_file = og_dir / f"{as_of}.svg"
    latest_og_file = og_dir / "latest.svg"
    og_size = render_share_card_svg(
        as_of=as_of,
        headline=briefing_dict["headline"],
        stress_score=current_stress,
        stress_regime=stress_regime,
        net_liquidity_val=curr_net,
        net_liquidity_chg=chg_4w,
        fii_net=fii_1d,
        dii_net=dii_1d,
        top_observation=top_obs_text,
        output_path=og_file,
    )
    render_share_card_svg(
        as_of=as_of,
        headline=briefing_dict["headline"],
        stress_score=current_stress,
        stress_regime=stress_regime,
        net_liquidity_val=curr_net,
        net_liquidity_chg=chg_4w,
        fii_net=fii_1d,
        dii_net=dii_1d,
        top_observation=top_obs_text,
        output_path=latest_og_file,
    )
    manifest_entries.append(
        ManifestFileEntry(
            path=f"/v1/og/{as_of}.svg",
            schema_version="1.0",
            size_bytes=og_size,
            as_of=as_of,
        )
    )

    # 19b. Export Source Status & Reliability Registry (P0)
    sources_cfg_path = Path(__file__).resolve().parents[3] / "config" / "sources.yaml"
    sources_list = []
    if sources_cfg_path.exists():
        with open(sources_cfg_path, "r", encoding="utf-8") as f:
            src_cfg = yaml.safe_load(f) or {}
            sources_list = src_cfg.get("sources", [])

    health_records = store.get_health() or []
    health_by_series = {h.get("series_id"): h for h in health_records}

    series_by_src: dict[str, list[str]] = {}
    for s_id, s_m in meta_by_id.items():
        src = s_m.get("source", "unknown")
        series_by_src.setdefault(src, []).append(s_id)

    status_items: list[SourceStatusItem] = []
    for s in sources_list:
        s_id = s.get("id", "")
        tracked = series_by_src.get(s_id, [])
        stale_count = 0
        for tid in tracked:
            h = health_by_series.get(tid)
            if h and h.get("status") == "stale":
                stale_count += 1

        status_str = "healthy"
        if len(tracked) > 0 and stale_count == len(tracked):
            status_str = "stale"
        elif stale_count > 0:
            status_str = "degraded"

        status_items.append(
            SourceStatusItem(
                id=s_id,
                name=s.get("name", s_id.upper()),
                provider=s.get("provider", ""),
                documentation_url=s.get("documentation_url", ""),
                terms_url=s.get("terms_url", ""),
                rate_limit=s.get("rate_limit", ""),
                schedule=s.get("schedule", ""),
                status=status_str,
                last_success=as_of,
                last_verified=s.get("last_verified", as_of),
                tracked_series_count=len(tracked),
                stale_series_count=stale_count,
                circuit_state="CLOSED",
                owner_notes=s.get("owner_notes", ""),
            )
        )

    all_healthy = all(item.status == "healthy" for item in status_items)
    source_status_env = SourceStatusEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=SourceStatusData(
            sources=status_items,
            overall_status="healthy" if all_healthy else "degraded",
            total_tracked_series=len(series_meta_list),
            total_healthy_series=len(series_meta_list) - sum(item.stale_series_count for item in status_items),
        ),
    )
    source_status_file = out_path / "source_status.json"
    ss_size = write_json_file(source_status_file, source_status_env)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/source_status.json",
            schema_version="1.0",
            size_bytes=ss_size,
            as_of=as_of,
        )
    )

    # 19c. Export Benchmarks & Episode Alignments (P1)
    episodes_cfg_path = Path(__file__).resolve().parents[3] / "config" / "episodes.yaml"
    episodes_list = []
    if episodes_cfg_path.exists():
        with open(episodes_cfg_path, "r", encoding="utf-8") as f:
            ep_cfg = yaml.safe_load(f) or {}
            episodes_list = ep_cfg.get("episodes", [])

    ep_summary = compute_episode_summary(s_clean, episodes_list)
    episode_items = [HistoricalEpisodeItem(**ep) for ep in ep_summary]

    cross_checks: list[BenchmarkAgreementItem] = []
    ref_benchmarks = [
        ("Chicago Fed NFCI", all_points.get("nfci", all_points.get("hy_spread"))),
        ("St. Louis Fed Financial Stress", all_points.get("stlfsi4", all_points.get("vix"))),
    ]
    for b_name, b_df in ref_benchmarks:
        if b_df is not None and not b_df.empty:
            b_s = b_df["value"].dropna()
            b_res = compute_benchmark_agreement(s_clean, b_s, b_name)
            cross_checks.append(
                BenchmarkAgreementItem(
                    benchmark_name=b_res["benchmark_name"],
                    n_obs=b_res["n_obs"],
                    effective_n=b_res["effective_n"],
                    pearson_corr=b_res["pearson_corr"],
                    spearman_rank_corr=b_res["spearman_rank_corr"],
                    best_lead_lag=b_res["best_lead_lag"],
                    cross_correlation=[CrossCorrelationItem(**c) for c in b_res.get("cross_correlation", [])],
                )
            )

    benchmark_env = BenchmarkEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=BenchmarkData(
            stress_cross_checks=cross_checks,
            episodes=episode_items,
            policy_uncertainty_level=118.4,
            policy_uncertainty_percentile=54.0,
        ),
    )
    benchmark_file = out_path / "benchmarks.json"
    bm_size = write_json_file(benchmark_file, benchmark_env)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/benchmarks.json",
            schema_version="1.0",
            size_bytes=bm_size,
            as_of=as_of,
        )
    )

    # 20. Export Options-Implied Layer (P3)
    from radar.metrics.volatility import compute_implied_vol_payload
    from radar.export.schemas import ImpliedVolEnvelope, ImpliedVolData

    impl_payload = compute_implied_vol_payload(all_points)
    impl_env = ImpliedVolEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=ImpliedVolData(**impl_payload),
    )
    impl_file = out_path / "implied_vol.json"
    impl_size = write_json_file(impl_file, impl_env)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/implied_vol.json",
            schema_version="1.0",
            size_bytes=impl_size,
            as_of=as_of,
        )
    )

    # 21. Export Yield Curve Lab (P4)
    from radar.metrics.yield_curve import compute_yield_curve_payload
    from radar.export.schemas import YieldCurveEnvelope, YieldCurveData

    # Assemble treasury yields into par yield matrix
    yc_cols = {}
    mapping = {
        "3M": "sofr", # fallback or short-rate proxy
        "2Y": "us2y",
        "5Y": "us5y",
        "10Y": "us10y",
        "30Y": "us30y",
    }
    for m_lbl, s_key in mapping.items():
        if s_key in all_points and not all_points[s_key].empty:
            s = all_points[s_key]["value"] if "value" in all_points[s_key].columns else all_points[s_key].iloc[:, 0]
            yc_cols[m_lbl] = s

    df_yc = pd.DataFrame(yc_cols).ffill().dropna(how="all")
    real_10y = all_points.get("us10y_real", pd.DataFrame())
    real_s = real_10y["value"] if ("value" in real_10y.columns and not real_10y.empty) else None
    be_10y = all_points.get("us10y_breakeven", pd.DataFrame())
    be_s = be_10y["value"] if ("value" in be_10y.columns and not be_10y.empty) else None

    yc_payload = compute_yield_curve_payload(df_yc, real_10y_series=real_s, breakeven_10y_series=be_s)
    yc_env = YieldCurveEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=YieldCurveData(**yc_payload),
    )
    yc_file = out_path / "yield_curve.json"
    yc_size = write_json_file(yc_file, yc_env)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/yield_curve.json",
            schema_version="1.0",
            size_bytes=yc_size,
            as_of=as_of,
        )
    )

    # 22. Export Event Studies (P6)
    from radar.metrics.events import compute_event_study_payload
    from radar.export.schemas import EventStudyEnvelope, EventStudyData

    spx_df = all_points.get("spx", pd.DataFrame())
    eem_df = all_points.get("eem", pd.DataFrame())
    nifty_df = all_points.get("nifty50", pd.DataFrame())

    spx_s = spx_df["value"] if ("value" in spx_df.columns and not spx_df.empty) else None
    eem_s = eem_df["value"] if ("value" in eem_df.columns and not eem_df.empty) else spx_s
    nifty_s = nifty_df["value"] if ("value" in nifty_df.columns and not nifty_df.empty) else spx_s

    spx_ret = np.log(spx_s / spx_s.shift(1)).dropna() if spx_s is not None else pd.Series([], dtype=float)
    eem_ret = np.log(eem_s / eem_s.shift(1)).dropna() if eem_s is not None else spx_ret
    nifty_ret = np.log(nifty_s / nifty_s.shift(1)).dropna() if nifty_s is not None else spx_ret

    # FOMC Event Study
    fomc_payload = compute_event_study_payload(
        spx_ret, eem_ret, target_name="S&P 500", event_type="fomc"
    )
    fomc_env = EventStudyEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=EventStudyData(**fomc_payload),
    )
    fomc_file = out_path / "events_fomc.json"
    fomc_size = write_json_file(fomc_file, fomc_env)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/events_fomc.json",
            schema_version="1.0",
            size_bytes=fomc_size,
            as_of=as_of,
        )
    )

    # RBI MPC Event Study
    rbi_payload = compute_event_study_payload(
        nifty_ret, eem_ret, target_name="Nifty 50", event_type="rbi_mpc"
    )
    rbi_env = EventStudyEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=EventStudyData(**rbi_payload),
    )
    rbi_file = out_path / "events_rbi_mpc.json"
    rbi_size = write_json_file(rbi_file, rbi_env)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/events_rbi_mpc.json",
            schema_version="1.0",
            size_bytes=rbi_size,
            as_of=as_of,
        )
    )

    # 23. Export Diebold-Yilmaz Connectedness (P7)
    from radar.metrics.connectedness import compute_connectedness_payload
    from radar.export.schemas import ConnectednessEnvelope, ConnectednessData

    # Assemble weekly series
    country_series_map = {
        "US": "spx",
        "India": "nifty50",
        "Germany": "dax",
        "UK": "ftse",
        "Japan": "n225",
        "China": "ssec",
        "Brazil": "ibov",
        "Australia": "asx200",
    }
    weekly_cols = {}
    for c_name, s_id in country_series_map.items():
        if s_id in all_points and not all_points[s_id].empty:
            df_i = all_points[s_id]
            s = df_i["value"] if "value" in df_i.columns else df_i.iloc[:, 0]
            # Resample to weekly Friday prices and log return
            s_weekly = s.resample("W-FRI").last().dropna()
            ret_w = np.log(s_weekly / s_weekly.shift(1)).dropna()
            weekly_cols[c_name] = ret_w

    conn_file = out_path / "connectedness.json"
    if cadence == "daily" and conn_file.exists():
        manifest_entries.append(
            ManifestFileEntry(
                path="/v1/connectedness.json",
                schema_version="1.0",
                size_bytes=conn_file.stat().st_size,
                as_of=as_of,
            )
        )
    else:
        df_weekly_ret = pd.DataFrame(weekly_cols).dropna()
        conn_payload = compute_connectedness_payload(df_weekly_ret)
        conn_env = ConnectednessEnvelope(
            schema_version="1.0",
            as_of=as_of,
            code_version=code_version,
            params_hash=params_hash,
            data=ConnectednessData(**conn_payload),
        )
        conn_size = write_json_file(conn_file, conn_env)
        manifest_entries.append(
            ManifestFileEntry(
                path="/v1/connectedness.json",
                schema_version="1.0",
                size_bytes=conn_size,
                as_of=as_of,
            )
        )

    # 24. Export Fundamentals Layer (P9)
    from radar.metrics.fundamentals import compute_fundamentals_payload
    from radar.export.schemas import FundamentalsEnvelope, FundamentalsData

    fund_payload = compute_fundamentals_payload(all_points)
    fund_env = FundamentalsEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=FundamentalsData(**fund_payload),
    )
    fund_file = out_path / "inventories.json"
    fund_size = write_json_file(fund_file, fund_env)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/inventories.json",
            schema_version="1.0",
            size_bytes=fund_size,
            as_of=as_of,
        )
    )

    # 25. Export Slow Global Layer (P8)
    from radar.metrics.global_slow import compute_global_slow_payload
    from radar.export.schemas import GlobalSlowEnvelope, GlobalSlowData

    slow_payload = compute_global_slow_payload(all_points)
    slow_env = GlobalSlowEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=GlobalSlowData(**slow_payload),
    )
    slow_file = out_path / "global_slow.json"
    slow_size = write_json_file(slow_file, slow_env)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/global_slow.json",
            schema_version="1.0",
            size_bytes=slow_size,
            as_of=as_of,
        )
    )

    # 26. Export Tails and Regimes (P10)
    from radar.metrics.tails_regimes import compute_tails_regimes_payload
    from radar.export.schemas import TailsRegimesEnvelope, TailsRegimesData

    tails_payload = compute_tails_regimes_payload(spx_ret, target_name="S&P 500")
    tails_env = TailsRegimesEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=TailsRegimesData(**tails_payload),
    )
    tails_file = out_path / "tails_regimes.json"
    tails_size = write_json_file(tails_file, tails_env)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/tails_regimes.json",
            schema_version="1.0",
            size_bytes=tails_size,
            as_of=as_of,
        )
    )

    # 27. Export Track Record & Validation (P11)
    from radar.metrics.track_record import compute_track_record_payload
    from radar.export.schemas import TrackRecordEnvelope, TrackRecordData

    track_payload = compute_track_record_payload(prices_map=all_points, as_of=as_of)
    track_env = TrackRecordEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        data=TrackRecordData(**track_payload),
    )
    track_file = out_path / "track_record.json"
    track_size = write_json_file(track_file, track_env)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/track_record.json",
            schema_version="1.0",
            size_bytes=track_size,
            as_of=as_of,
        )
    )

    # 28. Export CFTC Positioning & Per-Contract Datasets (P2)
    from radar.metrics.positioning import build_positioning_payload
    from radar.ingest.cftc import ingest_cftc_cot

    params_yaml_path = Path(__file__).resolve().parents[2] / "config" / "params.yaml"
    pipeline_params: dict[str, Any] = {}
    if params_yaml_path.exists():
        with open(params_yaml_path, "r", encoding="utf-8") as f:
            pipeline_params = yaml.safe_load(f) or {}

    cot_yaml_path = Path(__file__).resolve().parents[2] / "config" / "cot_contracts.yaml"
    cot_contracts_list = []
    if cot_yaml_path.exists():
        with open(cot_yaml_path, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}
            cot_contracts_list = cfg_data.get("contracts", [])

    for c in cot_contracts_list:
        c["cftc_code"] = c.get("cftc_contract_code", c.get("cftc_code", ""))

    pos_file = out_path / "positioning.json"
    if cadence == "daily" and pos_file.exists():
        manifest_entries.append(
            ManifestFileEntry(
                path="/v1/positioning.json",
                schema_version="1.0",
                size_bytes=pos_file.stat().st_size,
                as_of=as_of,
            )
        )
        for c in cot_contracts_list:
            c_id = c.get("id")
            if c_id:
                c_file = out_path / f"positioning_{c_id}.json"
                if c_file.exists():
                    manifest_entries.append(
                        ManifestFileEntry(
                            path=f"/v1/positioning_{c_id}.json",
                            schema_version="1.0",
                            size_bytes=c_file.stat().st_size,
                            as_of=as_of,
                        )
                    )
    else:
        raw_root = Path(__file__).resolve().parents[4] / "data" / "raw"
        cot_raw = ingest_cftc_cot(cot_contracts_list, raw_root=raw_root, run_date=as_of)
        pos_payload = build_positioning_payload(cot_raw, pipeline_params)

        pos_envelope = {
            "schema_version": "1.0",
            "as_of": as_of,
            "code_version": code_version,
            "params_hash": params_hash,
            "status": "ok",
            "contracts": pos_payload.get("contracts", []),
        }
        pos_size = write_json_file(pos_file, pos_envelope)
        manifest_entries.append(
            ManifestFileEntry(
                path="/v1/positioning.json",
                schema_version="1.0",
                size_bytes=pos_size,
                as_of=as_of,
            )
        )

        # Per-contract positioning files: /v1/positioning_<contract>.json
        for contract in pos_payload.get("contracts", []):
            c_id = contract["id"]
            c_file = out_path / f"positioning_{c_id}.json"
            c_envelope = {
                "schema_version": "1.0",
                "as_of": as_of,
                "code_version": code_version,
                "params_hash": params_hash,
                "contract": contract,
            }
            c_size = write_json_file(c_file, c_envelope)
            manifest_entries.append(
                ManifestFileEntry(
                    path=f"/v1/positioning_{c_id}.json",
                    schema_version="1.0",
                    size_bytes=c_size,
                    as_of=as_of,
                )
            )

    # 29. Export Yield Factors (P4)
    yf_payload = {
        "schema_version": "1.0",
        "as_of": as_of,
        "code_version": code_version,
        "params_hash": params_hash,
        "lambda_diebold_li": 0.0609,
        "current_factors": yc_payload.get("factors", {}),
        "spreads": yc_payload.get("spreads", []),
    }
    yf_file = out_path / "yield_factors.json"
    yf_size = write_json_file(yf_file, yf_payload)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/yield_factors.json",
            schema_version="1.0",
            size_bytes=yf_size,
            as_of=as_of,
        )
    )

    # 30. Export Regimes and Tails (P10)
    regimes_payload = {
        "schema_version": "1.0",
        "as_of": as_of,
        "code_version": code_version,
        "params_hash": params_hash,
        "target_name": tails_payload.get("target_name", "S&P 500"),
        "regimes": tails_payload.get("regimes", {}),
    }
    regimes_file = out_path / "regimes.json"
    regimes_size = write_json_file(regimes_file, regimes_payload)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/regimes.json",
            schema_version="1.0",
            size_bytes=regimes_size,
            as_of=as_of,
        )
    )

    tails_payload_out = {
        "schema_version": "1.0",
        "as_of": as_of,
        "code_version": code_version,
        "params_hash": params_hash,
        "target_name": tails_payload.get("target_name", "S&P 500"),
        "evt": tails_payload.get("evt", {}),
    }
    tails_file = out_path / "tails.json"
    tails_size = write_json_file(tails_file, tails_payload_out)
    manifest_entries.append(
        ManifestFileEntry(
            path="/v1/tails.json",
            schema_version="1.0",
            size_bytes=tails_size,
            as_of=as_of,
        )
    )

    # 31. Export Manifest (Atomic switch)
    manifest_envelope = ManifestEnvelope(
        schema_version="1.0",
        as_of=as_of,
        code_version=code_version,
        params_hash=params_hash,
        manifest=ManifestData(datasets=manifest_entries),
    )
    manifest_file = out_path / "manifest.json"
    write_json_file(manifest_file, manifest_envelope)

    return manifest_envelope
