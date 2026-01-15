"""
File: src/engine/beta.py

Beta computation and deviation analysis for macro transmission monitoring.

Calculates rolling beta between macro drivers and target assets,
and detects when target moves deviate significantly from expected (beta * macro move).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BetaStats:
    """Statistics from beta and deviation analysis."""
    beta: float              # Regression beta coefficient
    expected_move: float     # Expected target move = beta * macro_today
    residual: float          # Actual - expected move
    deviation_flag: bool     # True if deviation exceeds threshold
    target_vol: float        # Rolling volatility of target


def compute_beta_and_deviation(
    macro_returns: pd.Series,
    target_returns: pd.Series,
    beta_window: int,
    deviation_params: dict,
) -> BetaStats:
    """
    Compute beta and detect deviation from expected relationship.
    
    beta = cov(macro, target) / var(macro)
    expected = beta * macro_today
    residual = target_today - expected
    deviation_threshold = max(
        target_vol_multiplier * target_vol,
        beta_macro_multiplier * |beta| * |macro_today|
    )
    
    Args:
        macro_returns: Series of macro driver returns (DatetimeIndex)
        target_returns: Series of target asset returns (DatetimeIndex)
        beta_window: Window for beta calculation (e.g., 180 days)
        deviation_params: Dict with keys:
            - target_vol_multiplier (default 0.75)
            - beta_macro_multiplier (default 0.5)
            
    Returns:
        BetaStats with beta and deviation information
        
    Raises:
        ValueError: If insufficient data
    """
    # Align series
    aligned = pd.concat([macro_returns, target_returns], axis=1).dropna()
    aligned.columns = ["macro", "target"]
    
    if len(aligned) < beta_window:
        raise ValueError(
            f"Insufficient data for beta: need {beta_window}, have {len(aligned)}"
        )
    
    # Use last beta_window rows
    window_data = aligned.tail(beta_window)
    
    # Calculate beta = cov / var
    macro_var = window_data["macro"].var()
    
    if macro_var == 0 or np.isnan(macro_var):
        # No variance in macro - cannot compute meaningful beta
        return BetaStats(
            beta=0.0,
            expected_move=0.0,
            residual=float(window_data["target"].iloc[-1]),
            deviation_flag=False,
            target_vol=float(window_data["target"].std()),
        )
    
    covariance = window_data["macro"].cov(window_data["target"])
    beta = covariance / macro_var
    
    # Get today's moves
    macro_today = window_data["macro"].iloc[-1]
    target_today = window_data["target"].iloc[-1]
    
    # Expected and residual
    expected_move = beta * macro_today
    residual = target_today - expected_move
    
    # Target volatility
    target_vol = window_data["target"].std()
    
    # Deviation threshold parameters
    vol_mult = deviation_params.get("target_vol_multiplier", 0.75)
    beta_mult = deviation_params.get("beta_macro_multiplier", 0.5)
    
    # Deviation threshold = max of two components
    deviation_threshold = max(
        vol_mult * target_vol,
        beta_mult * abs(beta) * abs(macro_today)
    )
    
    # Check for deviation
    deviation_flag = abs(residual) > deviation_threshold
    
    return BetaStats(
        beta=float(beta),
        expected_move=float(expected_move),
        residual=float(residual),
        deviation_flag=deviation_flag,
        target_vol=float(target_vol),
    )


def compute_rolling_beta(
    macro_returns: pd.Series,
    target_returns: pd.Series,
    beta_window: int,
) -> pd.Series:
    """
    Compute full rolling beta time series.
    
    Args:
        macro_returns: Macro driver returns
        target_returns: Target asset returns
        beta_window: Rolling window size
        
    Returns:
        Series of rolling beta values
    """
    aligned = pd.concat([macro_returns, target_returns], axis=1).dropna()
    aligned.columns = ["macro", "target"]
    
    def calc_beta(window):
        if len(window) < beta_window:
            return np.nan
        macro = window["macro"]
        target = window["target"]
        var = macro.var()
        if var == 0:
            return np.nan
        return macro.cov(target) / var
    
    rolling_beta = aligned.rolling(window=beta_window).apply(
        lambda x: calc_beta(pd.DataFrame({"macro": x})),
        raw=False
    )
    
    # Alternative simpler approach using rolling cov/var
    rolling_cov = aligned["macro"].rolling(beta_window).cov(aligned["target"])
    rolling_var = aligned["macro"].rolling(beta_window).var()
    
    rolling_beta = rolling_cov / rolling_var
    rolling_beta = rolling_beta.replace([np.inf, -np.inf], np.nan)
    
    return rolling_beta


def compute_beta_stability(
    rolling_beta: pd.Series,
    lookback: int = 90,
) -> float:
    """
    Compute beta stability measure (inverse of beta volatility).
    
    Lower values indicate unstable/changing beta relationship.
    
    Args:
        rolling_beta: Series of rolling beta values
        lookback: Window for stability calculation
        
    Returns:
        Stability score (higher = more stable)
    """
    recent = rolling_beta.dropna().tail(lookback)
    if len(recent) < lookback // 2:
        return 0.0
    
    beta_vol = recent.std()
    beta_mean = abs(recent.mean())
    
    if beta_mean == 0:
        return 0.0
    
    # Coefficient of variation (inverted for stability)
    cv = beta_vol / beta_mean
    stability = 1.0 / (1.0 + cv)  # Bounded 0-1
    
    return float(stability)
