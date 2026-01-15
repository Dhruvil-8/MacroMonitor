"""
File: src/engine/returns.py

Return computation utilities.

Provides log return calculations for price time series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes log returns: r_t = ln(price_t / price_{t-1})
    
    Args:
        prices: DataFrame of prices with DatetimeIndex
        
    Returns:
        DataFrame of log returns with same columns as input.
        First row will be NaN (no prior price).
        Rows where all values are NaN are dropped.
        
    Raises:
        ValueError: If price matrix contains entirely empty columns
    """
    if prices.empty:
        raise ValueError("Price matrix is empty")
    
    # Check for entirely empty columns
    empty_cols = prices.columns[prices.isnull().all()]
    if len(empty_cols) > 0:
        raise ValueError(f"Price matrix contains entirely empty columns: {list(empty_cols)}")

    # Exclude Yield_Curve from log returns (linear instrument)
    valid_cols = [c for c in prices.columns if c != "Yield_Curve"]
    if not valid_cols:
        raise ValueError("No valid columns for log returns (only Yield_Curve present?)")
        
    prices_valid = prices[valid_cols]

    # Compute log returns
    # Add small epsilon or handle negative prices if any (Yield Curve excluded, but just in case)
    # Since we excluded linear instruments, assume assets are positive.
    log_prices = np.log(prices_valid)
    returns = log_prices.diff()

    # Handle infinities (from log of 0 or very small numbers)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows where ALL values are NaN (keeps rows with partial data)
    returns = returns.dropna(how="all")

    return returns.astype("float64")


def compute_mixed_returns(prices: pd.DataFrame, linear_cols: list[str]) -> pd.DataFrame:
    """
    Computes returns using mixed methodology:
    - Log returns for standard assets (prices)
    - Simple differences for linear assets (rates, spreads, Yield_Curve)
    
    Args:
        prices: Price DataFrame
        linear_cols: List of column names to treat as linear (diff)
        
    Returns:
        DataFrame of returns
    """
    if prices.empty:
        return pd.DataFrame()

    # Split columns
    all_cols = prices.columns.tolist()
    linear_target = [c for c in linear_cols if c in all_cols]
    log_target = [c for c in all_cols if c not in linear_target]
    
    returns_list = []
    
    # 1. Log Returns
    if log_target:
        try:
            p_log = prices[log_target]
            # Replace <= 0 with NaN to avoid errors
            p_log = p_log.where(p_log > 0)
            log_ret = np.log(p_log).diff()
            returns_list.append(log_ret)
        except Exception as e:
            print(f"Error computing log returns: {e}")

    # 2. Linear Returns (Differences)
    if linear_target:
        p_lin = prices[linear_target]
        lin_ret = p_lin.diff()
        returns_list.append(lin_ret)
        
    if not returns_list:
        return pd.DataFrame()
        
    # Combine
    combined = pd.concat(returns_list, axis=1)
    combined = combined.dropna(how="all")
    combined = combined.replace([np.inf, -np.inf], np.nan)
    
    return combined.astype("float64")


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes simple returns: r_t = (price_t - price_{t-1}) / price_{t-1}
    
    Args:
        prices: DataFrame of prices with DatetimeIndex
        
    Returns:
        DataFrame of simple returns
    """
    if prices.empty:
        raise ValueError("Price matrix is empty")
    
    returns = prices.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")
    
    return returns.astype("float64")


def get_return_for_date(
    returns: pd.DataFrame,
    asset: str,
    date: pd.Timestamp,
) -> float:
    """
    Get the return for a specific asset on a specific date.
    
    Args:
        returns: DataFrame of returns
        asset: Asset name (column)
        date: Target date
        
    Returns:
        Return value as float, or NaN if not available
    """
    if asset not in returns.columns:
        return np.nan
    
    if date not in returns.index:
        return np.nan
    
    return float(returns.loc[date, asset])
