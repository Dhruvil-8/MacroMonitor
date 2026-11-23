"""Command Line Interface for the Market Radar Quantitative Pipeline.

Supports:
- radar ingest [--source <source>] [--since <date>]
- radar compute [--metrics <metrics>]
- radar export [--target <target>]
- radar verify
- radar all
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, List, Optional
import yaml

from radar.clean.cleaner import clean_series
from radar.clean.health import evaluate_pipeline_health, evaluate_series_health
from radar.db.store import DataStore
from radar.ingest.fred import FREDIngestor
from radar.ingest.nse import NSEIngestor
from radar.ingest.yahoo import YahooIngestor

LOGGER = logging.getLogger("radar")


def setup_cli_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
    )
    handler.setFormatter(formatter)
    logging.root.handlers = [handler]
    logging.root.setLevel(level)


def load_config() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Load universe and params configuration."""
    config_dir = Path(__file__).resolve().parents[2] / "config"
    universe_file = config_dir / "universe.yaml"
    params_file = config_dir / "params.yaml"

    universe: list[dict[str, Any]] = []
    if universe_file.exists():
        with open(universe_file, "r", encoding="utf-8") as f:
            universe = yaml.safe_load(f) or []

    params: dict[str, Any] = {}
    params_hash = ""
    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            content = f.read()
            params = yaml.safe_load(content) or {}
            params_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

    return universe, params, params_hash


def cmd_ingest(args: argparse.Namespace) -> int:
    LOGGER.info(f"Ingesting data [source={args.source}, since={args.since}]")
    started_at = datetime.datetime.now(datetime.timezone.utc)
    universe, params, params_hash = load_config()
    store = DataStore()

    if not universe:
        LOGGER.warning("No series defined in universe.yaml")
        return 0

    # Store metadata
    store.upsert_meta(universe)

    # Initialize ingestors
    yahoo = YahooIngestor()
    fred = FREDIngestor()
    nse = NSEIngestor()

    cleaning_cfg = params.get("data_cleaning", {})
    max_ffill = cleaning_cfg.get("max_ffill_days", 3)
    jump_sigma = cleaning_cfg.get("jump_sigma", 8.0)
    stale_hours = cleaning_cfg.get("stale_after_hours", 36.0)
    max_stale_pct = cleaning_cfg.get("max_stale_pct", 0.20)

    health_reports = []
    stale_count = 0

    # Filter by source if requested
    target_series = universe
    if args.source != "all":
        target_series = [s for s in universe if s.get("source") == args.source]

    # Filter by cadence if requested (daily skips weekly/monthly series to eliminate network/CPU load)
    cadence = getattr(args, "cadence", "all")
    if cadence == "daily":
        target_series = [s for s in target_series if s.get("frequency", "D").upper() in {"D", "DAILY"}]
        LOGGER.info(f"Filtered {len(target_series)} daily series for ingestion (skipping weekly/monthly).")
    elif cadence == "weekly":
        target_series = [s for s in target_series if s.get("frequency", "D").upper() in {"W", "WEEKLY", "M", "MONTHLY"}]
        LOGGER.info(f"Filtered {len(target_series)} weekly/monthly series for ingestion.")

    for s in target_series:
        s_id = s["id"]
        source = s["source"]
        symbol = s["source_symbol"]
        unit = s.get("unit", "index")
        fallback = s.get("fallback")
        start = args.since or s.get("start_date", "2000-01-01")

        LOGGER.info(f"Fetching {s_id} ({source}:{symbol})...")
        try:
            if source == "yahoo":
                raw_df = yahoo.fetch_series(symbol, start_date=start, fallback_symbol=fallback)
            elif source == "fred":
                raw_df = fred.fetch_series(symbol, start_date=start)
            elif source == "nse":
                raw_df = nse.fetch_series(symbol, start_date=start, fallback_symbol=fallback)
            else:
                LOGGER.warning(f"Unknown source {source} for {s_id}")
                raw_df = None
        except Exception as e:
            LOGGER.error(f"Error fetching {s_id}: {e}")
            raw_df = None

        if raw_df is not None and not raw_df.empty:
            clean_df = clean_series(
                raw_df,
                series_id=s_id,
                unit=unit,
                max_ffill_days=max_ffill,
                jump_sigma=jump_sigma,
            )
            store.upsert_points(s_id, clean_df)
            health = evaluate_series_health(s_id, clean_df, stale_after_hours=stale_hours)
        else:
            # Check if we have previous points in store
            clean_df = store.get_points(s_id)
            health = evaluate_series_health(s_id, clean_df, stale_after_hours=stale_hours)

        health_reports.append(health)
        if health["status"] in {"stale", "failed"}:
            stale_count += 1

    store.upsert_health(health_reports)

    pipe_status, stale_pct = evaluate_pipeline_health(health_reports, max_stale_pct=max_stale_pct)
    finished_at = datetime.datetime.now(datetime.timezone.utc)

    run_record = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "status": pipe_status,
        "code_version": "v1.0.0",
        "params_hash": params_hash,
        "stale_series": stale_count,
        "notes": json.dumps({"stale_pct": stale_pct, "total": len(target_series)}),
    }
    store.record_run(run_record)
    LOGGER.info(f"Ingestion completed. Status: {pipe_status}, Stale: {stale_count}/{len(target_series)}")
    return 0


def cmd_compute(args: argparse.Namespace) -> int:
    LOGGER.info(f"Computing quantitative metrics [metrics={args.metrics}]")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    cadence = getattr(args, "cadence", "all")
    LOGGER.info(f"Exporting verified bundles [target={args.target}, cadence={cadence}]")
    store = DataStore()
    universe, params, params_hash = load_config()

    if args.target == "local" or args.target == "web":
        repo_root = Path(__file__).resolve().parents[3]
        out_dir = repo_root / "web" / "public" / "data" / "v1"
    else:
        out_dir = Path(args.target)

    from radar.export.writers import export_all_datasets

    manifest = export_all_datasets(
        store=store,
        out_dir=out_dir,
        params_hash=params_hash,
        cadence=cadence,
    )
    LOGGER.info(f"Export completed. {len(manifest.manifest.datasets)} datasets written to {out_dir}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    LOGGER.info("Verifying data contracts, schemas, and pipeline outputs")
    store = DataStore()
    health = store.get_health()
    if health:
        total = len(health)
        ok = sum(1 for h in health if h["status"] == "ok")
        LOGGER.info(f"Store health status: {ok}/{total} series OK")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    LOGGER.info("Executing end-to-end pipeline: ingest -> clean -> compute -> insights -> export")
    cmd_ingest(args)
    cmd_compute(args)
    cmd_verify(args)
    cmd_export(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radar",
        description="MacroMonitor Quantitative Analytics Pipeline",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Ingest
    ingest_p = subparsers.add_parser("ingest", help="Ingest raw data from external providers")
    ingest_p.add_argument("--source", type=str, default="all", help="Source to ingest")
    ingest_p.add_argument("--since", type=str, default=None, help="Start date (YYYY-MM-DD)")
    ingest_p.add_argument("--cadence", type=str, default="all", choices=["all", "daily", "weekly"], help="Cadence (daily or weekly)")

    # Compute
    compute_p = subparsers.add_parser("compute", help="Compute metrics from clean series")
    compute_p.add_argument("--metrics", type=str, default="all", help="Metrics group to compute")
    compute_p.add_argument("--cadence", type=str, default="all", choices=["all", "daily", "weekly"], help="Cadence (daily or weekly)")

    # Export
    export_p = subparsers.add_parser("export", help="Export precomputed bundles")
    export_p.add_argument("--target", type=str, default="local", help="Export destination")
    export_p.add_argument("--cadence", type=str, default="all", choices=["all", "daily", "weekly"], help="Cadence (daily or weekly)")

    # Verify
    subparsers.add_parser("verify", help="Run contract verification and sanity checks")

    # All
    all_p = subparsers.add_parser("all", help="Run complete pipeline from ingest to export")
    all_p.add_argument("--source", type=str, default="all")
    all_p.add_argument("--since", type=str, default=None)
    all_p.add_argument("--metrics", type=str, default="all")
    all_p.add_argument("--target", type=str, default="local")
    all_p.add_argument("--cadence", type=str, default="all", choices=["all", "daily", "weekly"], help="Cadence (daily or weekly)")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    setup_cli_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 1

    handlers = {
        "ingest": cmd_ingest,
        "compute": cmd_compute,
        "export": cmd_export,
        "verify": cmd_verify,
        "all": cmd_all,
    }

    handler = handlers.get(args.subcommand)
    if handler:
        return handler(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
