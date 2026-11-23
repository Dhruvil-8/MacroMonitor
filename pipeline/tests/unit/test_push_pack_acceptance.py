"""Acceptance test suite for the Push Pack (P0 - P12).

Governed by docs/PUSH_PACK_PLAN.md:
- P0: Source resilience and data health
- P1: Benchmarks and episodes
- P2: 16 CFTC futures contracts and lag labels
- P3: Options-implied VIX term structure and SKEW
- P4: Nelson-Siegel yield curve and curves.yaml configuration
- P5: Time-varying Kalman beta estimation
- P6: Event study CAR ribbons and decision calendars
- P7: Diebold-Yilmaz connectedness and GFEVD
- P8: Slow global layer and BIS REER
- P9: EIA 5-year seasonal inventory bands
- P10: EVT POT tails and Markov-switching volatility
- P11: Track record out-of-sample forward evaluation
- P12: Navigation groups, Command Palette, and database migrations (0003_push_pack.sql)
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml


def test_curves_yaml_configuration():
    repo_root = Path(__file__).resolve().parents[3]
    curves_path = repo_root / "pipeline" / "config" / "curves.yaml"
    assert curves_path.exists(), "pipeline/config/curves.yaml must exist"

    cfg = yaml.safe_load(curves_path.read_text(encoding="utf-8"))
    assert "curves" in cfg
    curves = cfg["curves"]
    assert len(curves) >= 4, "Must configure at least US, ECB, UK, and Japan curves"

    us_curve = next((c for c in curves if c["id"] == "us_treasury_par"), None)
    assert us_curve is not None
    assert us_curve["lambda_diebold_li"] == 0.0609
    assert len(us_curve["maturities_months"]) >= 10
    assert len(us_curve["spreads"]) >= 3


def test_migration_0003_push_pack_sql():
    repo_root = Path(__file__).resolve().parents[3]
    mig_path = repo_root / "pipeline" / "src" / "radar" / "db" / "migrations" / "0003_push_pack.sql"
    assert mig_path.exists(), "0003_push_pack.sql must exist"

    sql = mig_path.read_text(encoding="utf-8")
    assert "create table if not exists event_calendar" in sql
    assert "create table if not exists cot_positions" in sql
    assert "create table if not exists curve_points" in sql
    assert "create table if not exists weekly_levels" in sql
    assert "alter table event_calendar enable row level security" in sql


def test_cftc_16_contracts_positioning_payload():
    repo_root = Path(__file__).resolve().parents[3]
    cot_yaml_path = repo_root / "pipeline" / "config" / "cot_contracts.yaml"
    assert cot_yaml_path.exists()
    cfg = yaml.safe_load(cot_yaml_path.read_text(encoding="utf-8"))
    assert len(cfg.get("contracts", [])) >= 16

    # Verify positioning.json in exported bundle
    pos_json_path = repo_root / "web" / "public" / "data" / "v1" / "positioning.json"
    assert pos_json_path.exists(), "web/public/data/v1/positioning.json must exist"
    data = json.loads(pos_json_path.read_text(encoding="utf-8"))
    contracts = data.get("contracts", [])
    assert len(contracts) >= 16, f"Expected 16 contracts, got {len(contracts)}"

    for c in contracts:
        assert "id" in c
        assert "name" in c
        assert "asset_class" in c
        assert "net_pct_oi" in c
        assert "percentile" in c
        assert "lag_label" in c
        assert "Positions as of" in c["lag_label"]


def test_yield_factors_and_regimes_exports():
    repo_root = Path(__file__).resolve().parents[3]
    v1 = repo_root / "web" / "public" / "data" / "v1"

    yf_path = v1 / "yield_factors.json"
    assert yf_path.exists(), "yield_factors.json must exist"
    yf_data = json.loads(yf_path.read_text(encoding="utf-8"))
    assert "current_factors" in yf_data
    assert "level" in yf_data["current_factors"]

    regimes_path = v1 / "regimes.json"
    assert regimes_path.exists(), "regimes.json must exist"
    reg_data = json.loads(regimes_path.read_text(encoding="utf-8"))
    assert "regimes" in reg_data
    assert "current_regime" in reg_data["regimes"]

    tails_path = v1 / "tails.json"
    assert tails_path.exists(), "tails.json must exist"
    tails_data = json.loads(tails_path.read_text(encoding="utf-8"))
    assert "evt" in tails_data
    assert "shape_xi" in tails_data["evt"]


def test_manifest_contains_push_pack_datasets():
    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = repo_root / "web" / "public" / "data" / "v1" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_paths = [d["path"] for d in manifest["manifest"]["datasets"]]

    expected = [
        "/v1/benchmarks.json",
        "/v1/positioning.json",
        "/v1/positioning_sp500_emini.json",
        "/v1/implied_vol.json",
        "/v1/yield_curve.json",
        "/v1/yield_factors.json",
        "/v1/events_fomc.json",
        "/v1/events_rbi_mpc.json",
        "/v1/connectedness.json",
        "/v1/global_slow.json",
        "/v1/inventories.json",
        "/v1/regimes.json",
        "/v1/tails.json",
        "/v1/track_record.json",
        "/v1/source_status.json",
    ]
    for ep in expected:
        assert ep in dataset_paths, f"Manifest missing {ep}"
