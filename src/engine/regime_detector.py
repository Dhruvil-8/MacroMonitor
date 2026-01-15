"""
File: src/engine/regime_detector.py

Regime detection and alert gating logic for macro transmission monitoring.

Provides functions to:
- Compute macro move z-scores (how unusual is today's macro move)
- Evaluate the AND-gate for alert triggering
- Determine alert severity levels
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_macro_move_zscore(
    macro_returns: pd.Series,
    vol_window: int,
) -> float:
    """
    Compute z-score of today's macro move vs historical volatility.
    
    macro_move_z = macro_today / std(macro_last_vol_window)
    
    Args:
        macro_returns: Series of macro driver returns (DatetimeIndex)
        vol_window: Lookback window for volatility (e.g., 180 days)
        
    Returns:
        Z-score of today's macro move
        
    Raises:
        ValueError: If insufficient data
    """
    if len(macro_returns.dropna()) < vol_window:
        raise ValueError(
            f"Insufficient data for macro z-score: need {vol_window}, have {len(macro_returns.dropna())}"
        )
    
    # Get recent returns for volatility
    recent = macro_returns.dropna().tail(vol_window)
    
    # Today's move
    macro_today = recent.iloc[-1]
    
    # Historical volatility (std of the window excluding today)
    hist_returns = recent.iloc[:-1]
    hist_vol = hist_returns.std()
    
    if hist_vol == 0 or np.isnan(hist_vol):
        return 0.0
    
    macro_move_z = macro_today / hist_vol
    
    return float(macro_move_z)


def evaluate_alert_gate(
    z_score: float,
    macro_move_z: float,
    deviation_flag: bool,
    thresholds: Dict,
) -> bool:
    """
    Evaluate the AND-gate for alert triggering.
    
    Alert triggers when ALL conditions are met:
    1. |z_score| >= z_threshold (correlation has broken from historical)
    2. |macro_move_z| >= macro_move_threshold (macro moved significantly)
    3. deviation_flag == True (actual target move deviates from expected)
    
    Args:
        z_score: Correlation z-score from compute_rolling_correlation_zscore
        macro_move_z: Macro move z-score from compute_macro_move_zscore
        deviation_flag: Boolean from compute_beta_and_deviation
        thresholds: Dict containing:
            - z_threshold (default 2.0)
            - macro_move_threshold (default 1.5)
            
    Returns:
        True if alert should trigger
    """
    z_threshold = thresholds.get("z_threshold", 2.0)
    macro_move_threshold = thresholds.get("macro_move_threshold", 1.5)
    
    condition_1 = abs(z_score) >= z_threshold
    condition_2 = abs(macro_move_z) >= macro_move_threshold
    condition_3 = deviation_flag
    
    # Quant Enhancement: Trigger on extreme macro shocks (> 4 sigma) regardless of correlation structure
    # This ensures we catch Black Swan events even if transmission holds strictly (e.g. VIX -> SP500 beta works but magnitude is extreme)
    condition_shock = abs(macro_move_z) >= 4.0
    
    return (condition_1 and condition_2 and condition_3) or condition_shock


def determine_severity(
    z_score: float,
    macro_move_z: float,
) -> str:
    """
    Determine alert severity level based on z-scores.
    
    Severity mapping (deterministic):
    - high: |z_score| >= 3.0 AND |macro_move_z| >= 2.5
    - medium: default when alert triggered
    - low: not used for triggers but included in diagnostics
    
    Args:
        z_score: Correlation z-score
        macro_move_z: Macro move z-score
        
    Returns:
        Severity level string: "low", "medium", or "high"
    """
    abs_z = abs(z_score)
    abs_macro_z = abs(macro_move_z)
    
    if abs_z >= 3.0 and abs_macro_z >= 2.5:
        return "high"
    elif abs_z >= 2.0 or abs_macro_z >= 1.5:
        return "medium"
    else:
        return "low"


def compute_regime_score(
    corr_z: float,
    macro_z: float,
    deviation_flag: bool,
) -> float:
    """
    Compute a composite regime break score (0-100).
    
    Higher scores indicate more significant transmission breakdowns.
    Useful for ranking and prioritizing alerts.
    
    Args:
        corr_z: Correlation z-score
        macro_z: Macro move z-score
        deviation_flag: Whether deviation threshold breached
        
    Returns:
        Score from 0 to 100
    """
    # Base score from z-scores (capped)
    corr_component = min(abs(corr_z), 5.0) * 10  # 0-50
    macro_component = min(abs(macro_z), 5.0) * 6  # 0-30
    
    # Deviation bonus
    deviation_component = 20 if deviation_flag else 0
    
    total = corr_component + macro_component + deviation_component
    
    return min(100.0, total)


def is_market_stress_period(
    vix_returns: pd.Series,
    threshold: float = 2.5,
    window: int = 5,
) -> bool:
    """
    Detect if we are in a market stress period based on VIX spikes.
    
    Args:
        vix_returns: VIX return series
        threshold: Z-score threshold for stress detection
        window: Lookback window for stress detection
        
    Returns:
        True if in stress period
    """
    if len(vix_returns.dropna()) < window:
        return False
    
    recent = vix_returns.dropna().tail(window)
    
    # VIX spike = large positive returns
    max_spike = recent.max()
    
    # Check against 20-day vol
    if len(vix_returns.dropna()) >= 20:
        vol = vix_returns.dropna().tail(20).std()
        if vol > 0:
            spike_z = max_spike / vol
            return spike_z >= threshold
    
    return False


def filter_alerts_by_limit(
    alerts: list,
    max_alerts: int,
) -> list:
    """
    Filter alerts to respect daily limit, prioritizing by severity and score.
    
    Args:
        alerts: List of alert dictionaries
        max_alerts: Maximum allowed alerts per day
        
    Returns:
        Filtered list of top alerts
    """
    if len(alerts) <= max_alerts:
        return alerts
    
    # Sort by severity (high > medium > low) then by score descending
    severity_order = {"high": 0, "medium": 1, "low": 2}
    
    sorted_alerts = sorted(
        alerts,
        key=lambda a: (
            severity_order.get(a.get("severity", "low"), 2),
            -a.get("extra", {}).get("regime_score", 0)
        )
    )
    
    return sorted_alerts[:max_alerts]
