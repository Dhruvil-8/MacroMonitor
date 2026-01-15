"""
File: src/engine/correlation.py

Rolling correlation and z-score computation for macro transmission analysis.

Detects structural breaks in historical correlations between macro drivers
and target assets by comparing current rolling correlation to historical distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class CorrelationStats:
    """Statistics from rolling correlation analysis."""
    current_corr: float    # Current rolling correlation value
    mean_hist: float       # Mean of historical correlation distribution
    std_hist: float        # Standard deviation of historical distribution
    z_score: float         # Z-score of current vs historical


def compute_rolling_correlation_zscore(
    macro_returns: pd.Series,
    target_returns: pd.Series,
    corr_window: int,
    hist_window: int,
) -> CorrelationStats:
    """
    Compute rolling correlation and z-score vs historical distribution.
    
    Args:
        macro_returns: Series of macro driver returns (DatetimeIndex)
        target_returns: Series of target asset returns (DatetimeIndex)
        corr_window: Window for rolling correlation (e.g., 30 days)
        hist_window: Window for historical distribution (e.g., 180 days)
        
    Returns:
        CorrelationStats with current correlation and z-score
        
    Raises:
        ValueError: If insufficient history for computation
    """
    # Align series by date (inner join)
    aligned = pd.concat([macro_returns, target_returns], axis=1).dropna()
    aligned.columns = ["macro", "target"]
    
    min_required = corr_window + hist_window
    if len(aligned) < min_required:
        raise ValueError(
            f"Insufficient history: need {min_required} observations, have {len(aligned)}"
        )
    
    # Compute rolling correlation
    rolling_corr = aligned["macro"].rolling(window=corr_window).corr(aligned["target"])
    
    # Get current (last valid) rolling correlation
    current_corr = rolling_corr.dropna().iloc[-1] if not rolling_corr.dropna().empty else 0.0
    
    # Get historical distribution (last hist_window values of rolling correlation)
    hist_corr = rolling_corr.dropna().tail(hist_window)
    
    if len(hist_corr) < hist_window // 2:
        raise ValueError(
            f"Insufficient historical correlation data: need {hist_window}, have {len(hist_corr)}"
        )
    
    mean_hist = hist_corr.mean()
    std_hist = hist_corr.std()
    
    # Calculate z-score (handle zero std safely)
    if std_hist == 0 or np.isnan(std_hist):
        z_score = 0.0
    else:
        z_score = (current_corr - mean_hist) / std_hist
    
    return CorrelationStats(
        current_corr=float(current_corr),
        mean_hist=float(mean_hist),
        std_hist=float(std_hist),
        z_score=float(z_score),
    )


def compute_correlation_timeseries(
    macro_returns: pd.Series,
    target_returns: pd.Series,
    corr_window: int,
) -> pd.Series:
    """
    Compute full rolling correlation time series.
    
    Args:
        macro_returns: Series of macro driver returns
        target_returns: Series of target asset returns
        corr_window: Window for rolling correlation
        
    Returns:
        Series of rolling correlations indexed by date
    """
    aligned = pd.concat([macro_returns, target_returns], axis=1).dropna()
    aligned.columns = ["macro", "target"]
    
    rolling_corr = aligned["macro"].rolling(window=corr_window).corr(aligned["target"])
    
    return rolling_corr


def compute_correlation_matrix(
    returns: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Compute current pairwise correlation matrix for all assets.
    
    Args:
        returns: DataFrame of returns (assets as columns)
        window: Lookback window for correlation
        
    Returns:
        Correlation matrix as DataFrame
    """
    recent = returns.tail(window).dropna(how="all", axis=1)
    return recent.corr()


def detect_correlation_regime_change(
    rolling_corr: pd.Series,
    lookback: int = 180,
    threshold: float = 2.0,
) -> Tuple[bool, float]:
    """
    Detect if correlation has undergone a regime change.
    
    Args:
        rolling_corr: Series of rolling correlations
        lookback: Historical lookback for regime detection
        threshold: Z-score threshold for regime change
        
    Returns:
        Tuple of (regime_change_detected, z_score)
    """
    if len(rolling_corr.dropna()) < lookback:
        return False, 0.0
    
    hist = rolling_corr.dropna().tail(lookback)
    current = hist.iloc[-1]
    prior = hist.iloc[:-1]
    
    mean_prior = prior.mean()
    std_prior = prior.std()
    
    if std_prior == 0:
        return False, 0.0
    
    z_score = (current - mean_prior) / std_prior
    
    return abs(z_score) >= threshold, float(z_score)
