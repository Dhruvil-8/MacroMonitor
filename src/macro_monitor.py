"""
File: src/macro_monitor.py

Main orchestrator for macro transmission monitoring.

CLI tool that:
1. Loads configuration (asset universe, pairings, thresholds)
2. Updates/backfills price store via yfinance
3. Computes log returns
4. Iterates through macro-target pairings
5. Computes correlation z-scores, beta, and deviation
6. Evaluates alert gates
7. Saves alerts to structured JSON
8. Logs all actions in structured JSON format

Usage:
    python src/macro_monitor.py --backfill              # Full backfill
    python src/macro_monitor.py --date 2024-01-15       # Specific date
    python src/macro_monitor.py                         # Latest data
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.data_engine import (
    load_yaml,
    load_asset_universe,
    flatten_universe,
    update_price_store,
    prepare_analysis_ready_data,
    DataEngineError,
)
from src.engine.returns import compute_log_returns, compute_mixed_returns


from src.engine.correlation import compute_rolling_correlation_zscore, CorrelationStats
from src.engine.beta import compute_beta_and_deviation, BetaStats
from src.engine.regime_detector import (
    compute_macro_move_zscore,
    evaluate_alert_gate,
    determine_severity,
    compute_regime_score,
    filter_alerts_by_limit,
)
from src.alerts.formatter import Alert, save_alerts, format_alert_text
from src.utils import setup_json_logging, validate_config, resolve_asset_name, get_trading_date


# Configuration paths
CONFIG_DIR = Path("config")
ASSET_UNIVERSE_PATH = CONFIG_DIR / "asset_universe.yaml"
PAIRING_MATRIX_PATH = CONFIG_DIR / "pairing_matrix.yaml"
THRESHOLDS_PATH = CONFIG_DIR / "thresholds.yaml"


def load_configs() -> tuple:
    """
    Load all configuration files.
    
    Returns:
        Tuple of (asset_universe, pairing_matrix, thresholds)
        
    Raises:
        SystemExit with code 2 on error
    """
    try:
        asset_universe = load_asset_universe(ASSET_UNIVERSE_PATH)
        pairing_matrix = load_yaml(PAIRING_MATRIX_PATH)
        thresholds = load_yaml(THRESHOLDS_PATH)
        
        if not isinstance(pairing_matrix, list):
            raise ValueError("pairing_matrix.yaml must be a list of pairings")
        
        return asset_universe, pairing_matrix, thresholds
    
    except (FileNotFoundError, ValueError) as e:
        print(f"FATAL: Configuration error: {e}", file=sys.stderr)
        sys.exit(2)


def process_pairing(
    macro_name: str,
    target_name: str,
    rationale: str,
    returns: pd.DataFrame,
    thresholds: Dict,
    date: str,
    logger,
) -> Optional[Alert]:
    """
    Process a single macro-target pairing and generate alert if triggered.
    
    Args:
        macro_name: Semantic name of macro driver
        target_name: Semantic name of target asset
        rationale: Explanation of the relationship
        returns: DataFrame of log returns
        thresholds: Threshold configuration
        date: Current date string
        logger: Logger instance
        
    Returns:
        Alert object if triggered, None otherwise
    """
    try:
        # Check if both assets exist
        if macro_name not in returns.columns:
            logger.warning(
                f"Macro asset not in returns",
                extra={"macro": macro_name}
            )
            return None
        
        if target_name not in returns.columns:
            logger.warning(
                f"Target asset not in returns",
                extra={"target": target_name}
            )
            return None
        
        macro_returns = returns[macro_name].dropna()
        target_returns = returns[target_name].dropna()
        
        # Extract threshold parameters
        corr_window = thresholds.get("corr_window", 30)
        hist_corr_window = thresholds.get("hist_corr_window", 180)
        beta_window = thresholds.get("beta_window", 180)
        macro_vol_window = thresholds.get("macro_vol_window", 180)
        deviation_params = thresholds.get("deviation_threshold_params", {})
        
        # Compute correlation z-score
        try:
            corr_stats: CorrelationStats = compute_rolling_correlation_zscore(
                macro_returns=macro_returns,
                target_returns=target_returns,
                corr_window=corr_window,
                hist_window=hist_corr_window,
            )
        except ValueError as e:
            logger.debug(f"Insufficient data for correlation: {e}")
            return None
        
        # Compute beta and deviation
        try:
            beta_stats: BetaStats = compute_beta_and_deviation(
                macro_returns=macro_returns,
                target_returns=target_returns,
                beta_window=beta_window,
                deviation_params=deviation_params,
            )
        except ValueError as e:
            logger.debug(f"Insufficient data for beta: {e}")
            return None
        
        # Compute macro move z-score
        try:
            macro_move_z = compute_macro_move_zscore(
                macro_returns=macro_returns,
                vol_window=macro_vol_window,
            )
        except ValueError as e:
            logger.debug(f"Insufficient data for macro z-score: {e}")
            return None
        
        # Evaluate alert gate
        should_alert = evaluate_alert_gate(
            z_score=corr_stats.z_score,
            macro_move_z=macro_move_z,
            deviation_flag=beta_stats.deviation_flag,
            thresholds=thresholds,
        )
        
        if not should_alert:
            return None
        
        # Get today's moves as percentages
        macro_today = float(macro_returns.iloc[-1]) * 100
        target_today = float(target_returns.iloc[-1]) * 100
        
        # Determine severity
        severity = determine_severity(corr_stats.z_score, macro_move_z)
        
        # Compute regime score for prioritization
        regime_score = compute_regime_score(
            corr_z=corr_stats.z_score,
            macro_z=macro_move_z,
            deviation_flag=beta_stats.deviation_flag,
        )
        
        # Create alert
        alert = Alert(
            date=date,
            macro=macro_name,
            target=target_name,
            rationale=rationale,
            mean_corr=corr_stats.mean_hist,
            curr_corr=corr_stats.current_corr,
            corr_z_score=corr_stats.z_score,
            beta=beta_stats.beta,
            macro_move_pct=macro_today,
            target_move_pct=target_today,
            deviation=beta_stats.deviation_flag,
            severity=severity,
            extra={
                "hist_corr_window": hist_corr_window,
                "corr_window": corr_window,
                "beta_window": beta_window,
                "macro_move_z": macro_move_z,
                "std_hist_corr": corr_stats.std_hist,
                "expected_move_pct": beta_stats.expected_move * 100,
                "residual_pct": beta_stats.residual * 100,
                "target_vol": beta_stats.target_vol,
                "regime_score": regime_score,
            },
        )
        
        return alert
    
    except Exception as e:
        logger.error(
            f"Error processing pairing",
            extra={
                "macro": macro_name,
                "target": target_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        return None


def run_pipeline(
    backfill: bool = False,
    date: Optional[str] = None,
    start_date: Optional[str] = None,
) -> int:
    """
    Run the full macro transmission monitoring pipeline.
    
    Args:
        backfill: If True, perform full historical backfill
        date: Specific date to analyze (YYYY-MM-DD)
        start_date: Start date for backfill (YYYY-MM-DD)
        
    Returns:
        Exit code (0 for success, 2 for fatal error)
    """
    # Setup logging
    logger = setup_json_logging()
    
    logger.info("Starting macro transmission monitor", extra={"backfill": backfill, "date": date})
    
    try:
        # Step 1: Load configuration
        logger.info("Loading configuration")
        asset_universe, pairing_matrix, thresholds = load_configs()
        
        # Flatten universe
        flat_universe = flatten_universe(asset_universe)
        logger.info("Configuration loaded", extra={"assets": len(flat_universe), "pairings": len(pairing_matrix)})
        
        # Validate configuration
        validate_config(asset_universe, pairing_matrix)
        logger.info("Configuration validated")
        
        # Step 2: Update price store
        logger.info("Updating price store")
        try:
            prices = update_price_store(
                asset_map=flat_universe,
                start_date=start_date if backfill else None,
            )
        except DataEngineError as e:
            logger.error(f"Failed to update prices: {e}")
            return 2
        
        # Step 3: Prepare analysis-ready data
        logger.info("Preparing analysis data")
        prices_named = prepare_analysis_ready_data(prices, flat_universe)
        
        # Step 4: Compute returns (Mixed)
        logger.info("Computing returns (mixed)")
        # Define linear assets that require difference calculation instead of log returns
        linear_assets = ["Yield_Curve", "US_10Y", "US_2Y", "US_30Y", "German_10Y", "UK_10Y", "Japan_10Y"]
        returns = compute_mixed_returns(prices_named, linear_assets)
        
        # Determine analysis date
        if date:
            analysis_date = get_trading_date(date)
            # Filter returns to only include data up to analysis date for historical replay
            analysis_ts = pd.Timestamp(analysis_date)
            returns = returns[returns.index <= analysis_ts]
            
            if len(returns) == 0:
                logger.error("No data available for the specified date")
                return 2
        else:
            analysis_date = returns.index.max().strftime("%Y-%m-%d")
        
        logger.info("Analyzing", extra={"date": analysis_date, "return_rows": len(returns)})
        
        # Step 5: Iterate through pairings
        alerts: List[Alert] = []
        
        for pairing in pairing_matrix:
            macro_name = pairing.get("macro")
            target_raw = pairing.get("target")
            rationale = pairing.get("rationale", "")
            
            # Resolve target name (handle group names)
            target_name = resolve_asset_name(target_raw, asset_universe, flat_universe)
            if target_name is None:
                logger.warning(f"Could not resolve target: {target_raw}")
                continue
            
            # Process pairing
            alert = process_pairing(
                macro_name=macro_name,
                target_name=target_name,
                rationale=rationale,
                returns=returns,
                thresholds=thresholds,
                date=analysis_date,
                logger=logger,
            )
            
            if alert:
                alerts.append(alert)
                logger.info(
                    "Alert triggered",
                    extra={
                        "macro": macro_name,
                        "target": target_name,
                        "severity": alert.severity,
                        "z_score": round(alert.corr_z_score, 2),
                    }
                )
        
        # Step 6: Apply alert limit
        max_alerts = thresholds.get("max_allowed_alerts_per_day", 20)
        if len(alerts) > max_alerts:
            logger.warning(
                f"Alert limit exceeded, filtering",
                extra={"total": len(alerts), "max": max_alerts}
            )
            # Convert to dicts for filtering, then back to Alert objects
            alert_dicts = [a.to_dict() for a in alerts]
            filtered_dicts = filter_alerts_by_limit(alert_dicts, max_alerts)
            
            # Rebuild Alert objects from filtered dicts
            filtered_alerts = []
            for ad in filtered_dicts:
                filtered_alerts.append(Alert(**ad))
            alerts = filtered_alerts
        
        # Step 7: Save alerts
        if alerts:
            save_alerts(alerts, analysis_date)
            
            # Print human-readable alerts to stdout
            print("\n" + "=" * 70)
            print(f"MACRO TRANSMISSION ALERTS FOR {analysis_date}")
            print(f"Total alerts: {len(alerts)}")
            print("=" * 70)
            
            for alert in alerts:
                try:
                    print(format_alert_text(alert))
                except UnicodeEncodeError:
                    # Fallback for Windows consoles that can't handle emojis
                    text = format_alert_text(alert)
                    print(text.encode('ascii', 'replace').decode('ascii'))
        else:
            logger.info("No alerts triggered", extra={"date": analysis_date})
            print(f"\nNo transmission breaks detected for {analysis_date}")
        
        # Step 8: Summary
        logger.info(
            "Pipeline complete",
            extra={
                "date": analysis_date,
                "alerts_generated": len(alerts),
                "high_severity": sum(1 for a in alerts if a.severity == "high"),
                "medium_severity": sum(1 for a in alerts if a.severity == "medium"),
            }
        )
        
        return 0
    
    except Exception as e:
        logger.error(
            "Fatal error in pipeline",
            extra={
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        return 2


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Global Macro Transmission Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/macro_monitor.py --backfill              # Full 8-year backfill
    python src/macro_monitor.py --backfill --start-date 2020-01-01
    python src/macro_monitor.py --date 2024-03-15       # Analyze specific date
    python src/macro_monitor.py                         # Analyze latest data
        """,
    )
    
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Perform full historical backfill",
    )
    
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date to analyze (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for backfill (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    exit_code = run_pipeline(
        backfill=args.backfill,
        date=args.date,
        start_date=args.start_date,
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
