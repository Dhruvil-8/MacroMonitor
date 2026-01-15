"""
File: scripts/backfill.py

Backfill and data retention management script.

Provides:
- Full historical backfill
- Retention management (purge old data)
- Data validation

Usage:
    python scripts/backfill.py                          # Standard backfill
    python scripts/backfill.py --retention-years 5      # Keep only 5 years
    python scripts/backfill.py --start-date 2020-01-01  # Custom start
    python scripts/backfill.py --validate               # Validate existing data
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.data_engine import (
    load_asset_universe,
    flatten_universe,
    update_price_store,
    PRICES_PATH,
    DATA_DIR,
)
from src.alerts.formatter import ALERTS_DIR
from src.utils import setup_json_logging


def apply_retention(
    parquet_path: Path,
    retention_years: int,
    logger,
) -> None:
    """
    Apply retention policy to parquet data.
    
    Args:
        parquet_path: Path to prices parquet
        retention_years: Number of years to retain
        logger: Logger instance
    """
    if not parquet_path.exists():
        logger.warning("No parquet file to apply retention to")
        return
    
    prices = pd.read_parquet(parquet_path)
    original_rows = len(prices)
    
    cutoff_date = datetime.now() - timedelta(days=retention_years * 365)
    prices_filtered = prices[prices.index >= cutoff_date]
    
    rows_removed = original_rows - len(prices_filtered)
    
    if rows_removed > 0:
        prices_filtered.to_parquet(parquet_path)
        logger.info(
            "Applied retention policy",
            extra={
                "retention_years": retention_years,
                "rows_removed": rows_removed,
                "rows_remaining": len(prices_filtered),
            }
        )
    else:
        logger.info("No data older than retention period", extra={"retention_years": retention_years})


def cleanup_old_alerts(
    alerts_dir: Path,
    retention_days: int,
    logger,
) -> None:
    """
    Remove alert files older than retention period.
    
    Args:
        alerts_dir: Path to alerts directory
        retention_days: Number of days to retain
        logger: Logger instance
    """
    if not alerts_dir.exists():
        return
    
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    
    removed_count = 0
    for jsonl_file in alerts_dir.glob("*.jsonl"):
        file_date = jsonl_file.stem
        if file_date < cutoff_str:
            jsonl_file.unlink()
            removed_count += 1
    
    if removed_count > 0:
        logger.info(
            "Cleaned up old alert files",
            extra={"removed_count": removed_count, "retention_days": retention_days}
        )


def validate_data(
    parquet_path: Path,
    logger,
) -> bool:
    """
    Validate the price data.
    
    Args:
        parquet_path: Path to prices parquet
        logger: Logger instance
        
    Returns:
        True if valid, False otherwise
    """
    if not parquet_path.exists():
        logger.error("No price data found")
        return False
    
    prices = pd.read_parquet(parquet_path)
    
    # Check for empty data
    if prices.empty:
        logger.error("Price data is empty")
        return False
    
    # Check for too many missing values
    missing_pct = prices.isnull().sum().sum() / prices.size
    if missing_pct > 0.2:
        logger.warning(
            "High percentage of missing values",
            extra={"missing_pct": f"{missing_pct:.1%}"}
        )
    
    # Check date range
    date_range = (prices.index.max() - prices.index.min()).days
    
    # Check for data freshness
    days_old = (datetime.now() - prices.index.max().to_pydatetime().replace(tzinfo=None)).days
    if days_old > 5:
        logger.warning(
            "Data may be stale",
            extra={"days_since_last_update": days_old}
        )
    
    logger.info(
        "Data validation complete",
        extra={
            "rows": len(prices),
            "columns": len(prices.columns),
            "start_date": str(prices.index.min().date()),
            "end_date": str(prices.index.max().date()),
            "date_range_days": date_range,
            "missing_pct": f"{missing_pct:.1%}",
        }
    )
    
    return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill and data retention management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--retention-years",
        type=int,
        default=8,
        help="Number of years to retain (default: 8)",
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Custom start date for backfill (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate existing data, don't download",
    )
    
    parser.add_argument(
        "--cleanup-alerts",
        type=int,
        default=None,
        help="Clean up alerts older than N days",
    )
    
    parser.add_argument(
        "--apply-retention",
        action="store_true",
        help="Apply retention policy without downloading new data",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_json_logging()
    
    logger.info("Starting backfill script", extra=vars(args))
    
    try:
        # Validation only mode
        if args.validate:
            success = validate_data(PRICES_PATH, logger)
            sys.exit(0 if success else 1)
        
        # Apply retention only mode
        if args.apply_retention:
            apply_retention(PRICES_PATH, args.retention_years, logger)
            sys.exit(0)
        
        # Clean up old alerts
        if args.cleanup_alerts:
            cleanup_old_alerts(ALERTS_DIR, args.cleanup_alerts, logger)
        
        # Load configuration
        config_path = Path("config/asset_universe.yaml")
        universe = load_asset_universe(config_path)
        flat_universe = flatten_universe(universe)
        
        logger.info("Configuration loaded", extra={"assets": len(flat_universe)})
        
        # Determine start date
        if args.start_date:
            start_date = args.start_date
        else:
            start_date = (
                datetime.now() - timedelta(days=args.retention_years * 365)
            ).strftime("%Y-%m-%d")
        
        # Run backfill
        logger.info("Starting backfill", extra={"start_date": start_date})
        
        prices = update_price_store(
            asset_map=flat_universe,
            start_date=start_date,
        )
        
        # Apply retention
        apply_retention(PRICES_PATH, args.retention_years, logger)
        
        # Validate result
        validate_data(PRICES_PATH, logger)
        
        logger.info("Backfill complete", extra={"rows": len(prices)})
        
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
