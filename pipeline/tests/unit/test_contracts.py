# pipeline/tests/unit/test_contracts.py
import json
import pytest

from radar.export.schemas import (
    CompactSeriesData,
    CompactSeriesEnvelope,
    ManifestData,
    ManifestEnvelope,
    ManifestFileEntry,
    PulseData,
    PulseEnvelope,
    PulseStat,
    StateMapData,
    StateMapEnvelope,
    StateMapItem,
    VolLandscapeData,
    VolLandscapeEnvelope,
    VolLandscapeItem,
)


def test_manifest_schema_serialization():
    manifest = ManifestEnvelope(
        schema_version="1.0",
        as_of="2026-11-20",
        code_version="v1.0.0",
        params_hash="abcdef123456",
        manifest=ManifestData(
            datasets=[
                ManifestFileEntry(
                    path="/v1/pulse.json",
                    schema_version="1.0",
                    size_bytes=4200,
                    as_of="2026-11-20",
                )
            ]
        ),
    )

    raw_json = manifest.model_dump_json()
    parsed = json.loads(raw_json)
    assert parsed["schema_version"] == "1.0"
    assert parsed["manifest"]["datasets"][0]["path"] == "/v1/pulse.json"


def test_compact_series_schema_serialization():
    series = CompactSeriesEnvelope(
        schema_version="1.0",
        as_of="2026-11-20",
        code_version="v1.0.0",
        params_hash="abcdef123456",
        data=CompactSeriesData(
            id="spx",
            name="S&P 500",
            unit="index",
            frequency="D",
            d=["2026-11-19", "2026-11-20"],
            v=[5900.25, 5920.50],
        ),
    )

    raw_json = series.model_dump_json()
    parsed = json.loads(raw_json)
    assert parsed["data"]["id"] == "spx"
    assert len(parsed["data"]["d"]) == len(parsed["data"]["v"])


def test_state_map_schema_serialization():
    state_map = StateMapEnvelope(
        schema_version="1.0",
        as_of="2026-11-20",
        code_version="v1.0.0",
        params_hash="abcdef123456",
        data=StateMapData(
            assets=[
                StateMapItem(
                    asset="spx",
                    name="S&P 500",
                    group="equity_index",
                    region="US",
                    trend_t=2.45,
                    vol_percentile=42.0,
                    quadrant="Orderly trend",
                    momentum_20d=0.035,
                )
            ]
        ),
    )

    raw_json = state_map.model_dump_json()
    parsed = json.loads(raw_json)
    assert parsed["data"]["assets"][0]["quadrant"] == "Orderly trend"


def test_vol_landscape_schema_serialization():
    vol_landscape = VolLandscapeEnvelope(
        schema_version="1.0",
        as_of="2026-11-20",
        code_version="v1.0.0",
        params_hash="abcdef123456",
        data=VolLandscapeData(
            landscape=[
                VolLandscapeItem(
                    asset="spx",
                    name="S&P 500",
                    group="equity_index",
                    region="US",
                    vol10=0.125,
                    vol20=0.142,
                    vol60=0.165,
                    vol_trend=-0.139,
                    vol20_percentile=38.5,
                    n=2520,
                    n_eff=310.5,
                )
            ]
        ),
    )

    raw_json = vol_landscape.model_dump_json()
    parsed = json.loads(raw_json)
    assert parsed["data"]["landscape"][0]["asset"] == "spx"
    assert parsed["data"]["landscape"][0]["n_eff"] == 310.5


def test_pulse_schema_serialization():
    pulse = PulseEnvelope(
        schema_version="1.0",
        as_of="2026-11-20",
        code_version="v1.0.0",
        params_hash="abcdef123456",
        data=PulseData(
            headline="Global risk assets remain in orderly trend amidst quiet volatility",
            risk_regime="Risk-on / Quiet",
            stats=[
                PulseStat(
                    id="spx",
                    name="S&P 500",
                    value=5920.50,
                    change_1d=0.35,
                    unit="index",
                    percentile=94.0,
                    sparkline=[5880.0, 5895.0, 5920.5],
                )
            ],
            observations_count=4,
        ),
    )

    raw_json = pulse.model_dump_json()
    parsed = json.loads(raw_json)
    assert parsed["data"]["stats"][0]["value"] == 5920.50


def test_liquidity_schema_serialization():
    from radar.export.schemas import LiquidityEnvelope, LiquidityData, LiquidityAudit

    liq = LiquidityEnvelope(
        schema_version="1.0",
        as_of="2026-11-20",
        code_version="v1.0.0",
        params_hash="abcdef123456",
        data=LiquidityData(
            current_net_liquidity=5824.44,
            change_4w=-45.2,
            change_4w_pct=34.5,
            change_13w=-120.4,
            change_13w_pct=28.2,
            stablecoins_total_bn=282.5,
            stablecoins_30d_pct=4.8,
            dates=["2026-11-18", "2026-11-19", "2026-11-20"],
            net_liquidity=[5820.0, 5822.0, 5824.44],
            fed_assets=[6741.45, 6741.45, 6741.45],
            tga=[916.61, 916.61, 916.61],
            rrp=[0.40, 0.40, 0.40],
            spx_overlay=[6900.0, 6905.0, 6910.39],
            rolling_corr_60d=[0.45, 0.46, 0.47],
            audit=LiquidityAudit(
                formula="Net Liquidity = (WALCL / 1000) - (WTREGEN / 1000) - RRPONTSYD",
                parameters={"change_windows": [20, 65], "stablecoin_window": 30, "rolling_corr_window": 60},
                series_used=["fed_assets (WALCL)", "tga (WTREGEN)", "rrp (RRPONTSYD)", "spx (^GSPC)"],
                n=750,
                n_eff=128.4,
            ),
        ),
    )
    raw_json = liq.model_dump_json()
    parsed = json.loads(raw_json)
    assert parsed["data"]["current_net_liquidity"] == 5824.44
    assert len(parsed["data"]["dates"]) == 3


def test_sectors_schema_serialization():
    from radar.export.schemas import (
        SectorsEnvelope,
        SectorsData,
        SectorRegionData,
        SectorItem,
        SectorRotationPoint,
    )

    item = SectorItem(
        id="xlk",
        name="Information Technology",
        weight=0.315,
        returns={"1D": 0.012, "5D": 0.034},
        percentiles={"1D": 68.5, "5D": 72.0},
        contributions={"1D": 0.00378, "5D": 0.01071},
        rs_ratio=102.4,
        rs_mom=101.1,
        phase="Leading",
        trail=[SectorRotationPoint(date="2026-11-20", rs_ratio=102.4, rs_mom=101.1)],
    )

    region = SectorRegionData(
        region_id="us",
        country_name="United States",
        benchmark_name="S&P 500",
        index_id="spx",
        as_of="2024-10-31",
        weights_source="S&P Dow Jones Indices S&P 500 Factsheet",
        weights_source_url="https://www.spglobal.com",
        index_returns={"1D": 0.008, "5D": 0.025},
        sector_breadth={"1D": 0.636, "5D": 0.727},
        residuals={"1D": 0.0005, "5D": 0.0012},
        sectors=[item],
    )

    sectors = SectorsEnvelope(
        schema_version="1.0",
        as_of="2026-11-20",
        code_version="v1.0.0",
        params_hash="abcdef123456",
        data=SectorsData(regions={"us": region}),
    )

    raw_json = sectors.model_dump_json()
    parsed = json.loads(raw_json)
    assert parsed["data"]["regions"]["us"]["benchmark_name"] == "S&P 500"
    assert parsed["data"]["regions"]["us"]["sectors"][0]["name"] == "Information Technology"
    assert parsed["data"]["regions"]["us"]["sectors"][0]["trail"][0]["rs_ratio"] == 102.4


def test_risk_schema_serialization():
    from radar.export.schemas import (
        RiskEnvelope,
        RiskData,
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

    cone = [VolatilityConePoint(window=20, min=0.08, p10=0.10, p25=0.12, p50=0.15, p75=0.18, p90=0.22, max=0.35, current=0.14, percentile=42.0)]
    vol_hist = [RealizedEwmaPoint(date="2026-11-20", realized_vol=0.14, ewma_vol=0.138)]
    dist = DistributionShapeData(skewness=-0.45, excess_kurtosis=2.1, jb_stat=45.2, jb_pvalue=0.0001, qq_theoretical=[-2.0, 0.0, 2.0], qq_empirical=[-2.5, 0.0, 2.4], mean=0.0003, std=0.0095)
    tail = [TailRiskMetric(confidence=0.95, hist_var=0.016, hist_es=0.024, gauss_var=0.015, gauss_es=0.019, tail_ratio=1.26)]
    acf = AutocorrelationRiskData(lags=[1, 2], acf_returns=[0.02, -0.01], acf_squared=[0.18, 0.12], band=0.08, n_eff=600.0)
    dd_s = [DrawdownSeriesPoint(date="2026-11-20", drawdown=-0.035)]
    dd_ep = [DrawdownEpisodeItem(peak_date="2026-07-15", trough_date="2026-08-05", recovery_date="2026-09-10", depth=-0.085, duration_days=21, recovery_days=36, status="Recovered")]

    profile = AssetRiskProfile(
        asset_id="spx",
        asset_name="S&P 500",
        asset_group="Equities",
        volatility_cone=cone,
        vol_history=vol_hist,
        distribution=dist,
        tail_risk=tail,
        autocorrelation=acf,
        drawdown_series=dd_s,
        drawdown_episodes=dd_ep,
    )

    pca = PcaFactorData(
        scree=[54.2, 18.5, 9.1],
        loadings_pc1={"spx": 0.42, "nifty50": 0.31},
        loadings_pc2={"spx": 0.28, "nifty50": -0.35},
        pc1_history=[{"date": "2026-11-20", "pc1_share": 54.2}],
    )

    dc = [DownsideCorrelationItem(target_id="nifty50", target_name="Nifty 50", all_day_corr=0.45, downside_corr=0.62, stress_premium=0.17, stress_threshold=-0.012, n_all=500, n_stress=50)]

    envelope = RiskEnvelope(
        schema_version="1.0",
        as_of="2026-11-20",
        code_version="v1.0.0",
        params_hash="abcdef123456",
        data=RiskData(
            selected_asset="spx",
            assets_available=[{"id": "spx", "name": "S&P 500"}],
            profiles={"spx": profile},
            pca=pca,
            downside_correlations=dc,
            market_synchronization_latest=54.2,
        ),
    )

    raw_json = envelope.model_dump_json()
    parsed = json.loads(raw_json)
    assert parsed["data"]["selected_asset"] == "spx"
    assert parsed["data"]["pca"]["scree"][0] == 54.2
    assert parsed["data"]["downside_correlations"][0]["stress_premium"] == 0.17


def test_context_schema_serialization():
    from radar.export.schemas import ContextEnvelope, ContextData, AnalogItem

    ctx = ContextEnvelope(
        schema_version="1.0",
        as_of="2026-11-20",
        code_version="v1.0.0",
        params_hash="abcdef123456",
        data=ContextData(
            id="spx",
            name="S&P 500",
            group="Equities",
            region="United States",
            unit="index",
            current_value=5920.50,
            percentile=94.2,
            z_score=1.85,
            n=1000,
            n_eff=450.0,
            histogram_bins=[0.01, 0.02],
            histogram_counts=[100, 200],
            kde_grid=[-0.03, 0.0, 0.03],
            kde_density=[0.5, 2.0, 0.5],
            normal_density=[0.4, 2.1, 0.4],
            qq_theoretical=[-2.0, 0.0, 2.0],
            qq_empirical=[-2.2, 0.0, 2.3],
            ecdf_x=[-0.03, 0.0, 0.03],
            ecdf_y=[0.05, 0.50, 0.95],
            calendar_heatmap=[{"year": 2024, "month": 1, "return": 0.025}],
            analogs=[AnalogItem(date="2024-05-10", value=5100.0, distance=0.015, fwd_5d=0.012, fwd_20d=0.034)],
            peer_rank=1,
            peer_count=10,
            description="Historical distribution context for S&P 500",
            source_note="Attribution: YAHOO",
        ),
    )

    raw_json = ctx.model_dump_json()
    parsed = json.loads(raw_json)
    assert parsed["data"]["id"] == "spx"
    assert parsed["data"]["kde_grid"][1] == 0.0
    assert parsed["data"]["analogs"][0]["fwd_5d"] == 0.012



