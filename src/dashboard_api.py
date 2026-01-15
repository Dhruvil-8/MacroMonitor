"""
File: src/dashboard_api.py

Dashboard API layer for Streamlit frontend.

Provides read-only functions that the dashboard imports to fetch
pre-computed results. No analytics logic is duplicated here - 
all computations happen in the engine modules.

The dashboard should NEVER recompute core metrics. It only reads
from data/processed/ and provides visualization-ready data.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.engine.data_engine import load_yaml, PRICES_PATH
from src.alerts.formatter import load_alerts, get_alert_history, ALERTS_DIR


# Configuration paths
CONFIG_DIR = Path("config")
ASSET_UNIVERSE_PATH = CONFIG_DIR / "asset_universe.yaml"
PAIRING_MATRIX_PATH = CONFIG_DIR / "pairing_matrix.yaml"
THRESHOLDS_PATH = CONFIG_DIR / "thresholds.yaml"


def get_latest_alerts() -> List[Dict[str, Any]]:
    """
    Get the most recent alerts.
    
    Returns:
        List of alert dictionaries from latest.json
    """
    return load_alerts(date=None)


def get_alerts_for_date(date: str) -> List[Dict[str, Any]]:
    """
    Get alerts for a specific date.
    
    Args:
        date: Date string in YYYY-MM-DD format
        
    Returns:
        List of alert dictionaries
    """
    return load_alerts(date=date)


def get_alert_history_days(days: int = 30) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get alert history for past N days.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Dict mapping dates to alert lists
    """
    return get_alert_history(days=days)


def get_available_dates() -> List[str]:
    """
    Get list of dates with available alert data.
    
    Returns:
        Sorted list of date strings (newest first)
    """
    if not ALERTS_DIR.exists():
        return []
    
    dates = []
    for jsonl_file in ALERTS_DIR.glob("*.jsonl"):
        dates.append(jsonl_file.stem)
    
    return sorted(dates, reverse=True)


def get_latest_date() -> Optional[str]:
    """
    Get the most recent date with data.
    
    Returns:
        Date string or None
    """
    if not (ALERTS_DIR / "latest.json").exists():
        return None
    
    try:
        with open(ALERTS_DIR / "latest.json", "r") as f:
            data = json.load(f)
        return data.get("date")
    except Exception:
        return None


def get_price_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get price data from the parquet store.
    
    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        DataFrame of prices
    """
    if not PRICES_PATH.exists():
        return pd.DataFrame()
    
    prices = pd.read_parquet(PRICES_PATH)
    
    if start_date:
        prices = prices[prices.index >= start_date]
    if end_date:
        prices = prices[prices.index <= end_date]
    
    return prices


def get_returns_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get log returns data for interactive analysis.
    
    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        DataFrame of log returns
    """
    from src.engine.returns import compute_log_returns
    
    prices = get_price_data(start_date, end_date)
    if prices.empty:
        return pd.DataFrame()
    
    # Ensure raw prices are prepared? 
    # prices.parquet has raw prices. 
    # yield curve logic is in data_engine.prepare_analysis_ready_data.
    # We should reproduce that minimal prep here or call it.
    from src.engine.data_engine import prepare_analysis_ready_data, load_yaml
    
    # We need asset map to resolve names? 
    # prices.parquet has TICKERS.
    # We want SEMANTIC NAMES for display.
    
    try:
        universe = load_yaml(ASSET_UNIVERSE_PATH)
        # We need to flatten the universe manually or use data_engine helper
        # Import helper locally to avoid circular imports at top level if any
        from src.engine.data_engine import flatten_universe
        asset_map = flatten_universe(universe)
        
        # Prepare data (renaming, yield curve, ffill)
        proc_df = prepare_analysis_ready_data(prices, asset_map)
        
        # Compute returns
        returns = compute_log_returns(proc_df)
        return returns
        
    except Exception as e:
        print(f"Error computing returns: {e}")
        return pd.DataFrame()


def get_asset_universe() -> Dict[str, Dict[str, str]]:
    """
    Get the asset universe configuration.
    
    Returns:
        Nested dict of asset groups
    """
    try:
        return load_yaml(ASSET_UNIVERSE_PATH)
    except Exception:
        return {}


def get_pairing_matrix() -> List[Dict[str, str]]:
    """
    Get the pairing matrix configuration.
    
    Returns:
        List of pairing dictionaries
    """
    try:
        return load_yaml(PAIRING_MATRIX_PATH)
    except Exception:
        return []


def get_thresholds() -> Dict[str, Any]:
    """
    Get the threshold configuration.
    
    Returns:
        Threshold configuration dict
    """
    try:
        return load_yaml(THRESHOLDS_PATH)
    except Exception:
        return {}


def get_alerts_summary() -> Dict[str, Any]:
    """
    Get summary statistics for alerts.
    
    Returns:
        Dict with alert summary stats
    """
    alerts = get_latest_alerts()
    history = get_alert_history_days(30)
    
    # Count by severity
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for alert in alerts:
        sev = alert.get("severity", "low")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    # Historical trend
    daily_counts = []
    for date, date_alerts in sorted(history.items()):
        daily_counts.append({
            "date": date,
            "count": len(date_alerts),
            "high": sum(1 for a in date_alerts if a.get("severity") == "high"),
        })
    
    return {
        "latest_date": get_latest_date(),
        "latest_count": len(alerts),
        "severity_counts": severity_counts,
        "daily_history": daily_counts[-30:],  # Last 30 days
    }


def get_correlation_heatmap_data() -> Tuple[List[str], List[str], List[List[float]]]:
    """
    Get data for correlation z-score heatmap.
    
    Extracts macro-target pairs and their z-scores from latest alerts.
    
    Returns:
        Tuple of (macro_names, target_names, z_score_matrix)
    """
    alerts = get_latest_alerts()
    pairings = get_pairing_matrix()
    
    # Get unique macros and targets
    macros = sorted(set(p["macro"] for p in pairings))
    targets = sorted(set(p["target"] for p in pairings))
    
    # Build z-score lookup
    z_scores = {}
    for alert in alerts:
        key = (alert["macro"], alert["target"])
        z_scores[key] = alert["corr_z_score"]
    
    # Build matrix
    matrix = []
    for macro in macros:
        row = []
        for target in targets:
            z = z_scores.get((macro, target), 0.0)
            row.append(z)
        matrix.append(row)
    
    return macros, targets, matrix


def get_scatter_data() -> List[Dict[str, Any]]:
    """
    Get data for macro_move_z vs corr_z_score scatter plot.
    
    Returns:
        List of dicts with plotting data
    """
    alerts = get_latest_alerts()
    
    data = []
    for alert in alerts:
        data.append({
            "macro": alert["macro"],
            "target": alert["target"],
            "corr_z_score": alert["corr_z_score"],
            "macro_move_z": alert.get("extra", {}).get("macro_move_z", 0),
            "severity": alert["severity"],
            "label": f"{alert['macro']} → {alert['target']}",
        })
    
    return data


def format_download_json() -> str:
    """
    Format latest alerts as downloadable JSON string.
    
    Returns:
        JSON string
    """
    alerts = get_latest_alerts()
    return json.dumps(
        {
            "date": get_latest_date(),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "count": len(alerts),
            "alerts": alerts,
        },
        indent=2,
    )
