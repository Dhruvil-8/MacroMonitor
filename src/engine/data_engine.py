"""
File: src/engine/data_engine.py

Market data ingestion and normalization layer.

Responsibilities:
- Load configuration YAML files
- Load asset universe YAML
- Incrementally fetch missing price history using yfinance
- Normalize column naming
- Persist parquet store
- Provide analysis-ready price matrix

No analytics logic belongs here.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
import yfinance as yf


LOGGER = logging.getLogger(__name__)

# Default paths
DATA_DIR = Path("data/processed")
PRICES_PATH = DATA_DIR / "prices.parquet"


class DataEngineError(RuntimeError):
    """Exception raised for data engine errors."""
    pass


def load_yaml(path: Path) -> dict:
    """
    Load YAML file and return dict.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Parsed YAML content as dictionary
        
    Raises:
        ValueError: If file cannot be parsed
        FileNotFoundError: If file does not exist
    """
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
        
        if content is None:
            return {}
        
        if not isinstance(content, (dict, list)):
            raise ValueError(f"Invalid YAML format in {path}: expected dict or list")
        
        return content
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error in {path}: {e}") from e


def load_asset_universe(config_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load asset universe configuration.
    
    Args:
        config_path: Path to asset_universe.yaml
        
    Returns:
        Nested dict of asset groups
        
    Raises:
        DataEngineError: If file not found or invalid format
    """
    if not config_path.exists():
        raise DataEngineError(f"Asset universe file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        universe = yaml.safe_load(f)

    if not isinstance(universe, dict):
        raise DataEngineError("Invalid asset universe format: expected dict")

    return universe


def flatten_universe(universe: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Converts nested universe groups into a flat mapping.
    
    Args:
        universe: Nested dict from asset_universe.yaml
        
    Returns:
        Flat mapping: semantic_name -> ticker
        
    Raises:
        DataEngineError: If duplicate asset names detected
    """
    flat: Dict[str, str] = {}

    for group_name, group_assets in universe.items():
        if not isinstance(group_assets, dict):
            continue

        for semantic_name, ticker in group_assets.items():
            if semantic_name in flat:
                raise DataEngineError(f"Duplicate asset name detected: {semantic_name}")
            flat[semantic_name] = ticker

    return flat


def _download_prices(
    tickers: List[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices from yfinance.
    
    Args:
        tickers: List of Yahoo Finance ticker symbols
        start: Start date (None for historical max)
        end: End date (None for today)
        
    Returns:
        DataFrame with tickers as columns, dates as index
        
    Raises:
        DataEngineError: If download fails or returns empty
    """
    LOGGER.info(
        "Downloading market data",
        extra={"tickers": len(tickers), "start": str(start), "end": str(end)}
    )

    try:
        data = yf.download(
            tickers=tickers,
            start=None if start is None else start.strftime("%Y-%m-%d"),
            end=None if end is None else end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as e:
        raise DataEngineError(f"Failed to download market data: {e}") from e

    if data.empty:
        raise DataEngineError("Downloaded data is empty - check ticker validity")

    # Normalize to: columns = ticker symbols, values = adjusted close
    if isinstance(data.columns, pd.MultiIndex):
        close_frames = []
        for ticker in tickers:
            if ticker in data.columns.levels[0]:
                series = data[ticker]["Close"].rename(ticker)
                close_frames.append(series)
            else:
                LOGGER.warning(f"Ticker not found in download: {ticker}")
        
        if not close_frames:
            raise DataEngineError("No valid ticker data downloaded")
        
        prices = pd.concat(close_frames, axis=1)
    else:
        # Single ticker case
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.sort_index(inplace=True)

    return prices


def update_price_store(
    asset_map: Dict[str, str],
    parquet_path: Optional[Path] = None,
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Incrementally fetch missing ADJUSTED CLOSE prices for tickers.
    
    - If parquet_path exists, only fetch from last_date + 1.
    - Use yfinance.download with multi-ticker batch.
    - Persist combined parquet at parquet_path with index = date (UTC naive).
    
    Args:
        asset_map: Mapping of semantic_name -> ticker
        parquet_path: Path to parquet store (default: data/processed/prices.parquet)
        start_date: Override start date for backfill (YYYY-MM-DD format)
        
    Returns:
        Combined DataFrame (columns = tickers, index = dates)
    """
    if parquet_path is None:
        parquet_path = PRICES_PATH
    
    # Ensure directory exists
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    tickers = sorted(set(asset_map.values()))

    if parquet_path.exists() and start_date is None:
        existing = pd.read_parquet(parquet_path)
        last_date = existing.index.max()
        fetch_start = last_date + pd.Timedelta(days=1)
        LOGGER.info(
            "Existing price store found",
            extra={"last_date": str(last_date), "rows": len(existing)}
        )
    else:
        existing = None
        if start_date:
            fetch_start = pd.Timestamp(start_date)
        else:
            # Default: 10 years of history
            fetch_start = pd.Timestamp.now() - pd.Timedelta(days=10*365)
        LOGGER.info(
            "Starting fresh backfill",
            extra={"start_date": str(fetch_start)}
        )

    try:
        new_data = _download_prices(tickers=tickers, start=fetch_start)
    except DataEngineError as e:
        if existing is not None:
            LOGGER.warning(f"Download failed, using existing data: {e}")
            return existing
        raise

    if existing is not None:
        # Merge new columns into existing if needed
        for col in new_data.columns:
            if col not in existing.columns:
                existing[col] = pd.NA
        
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new_data

    # Remove empty rows (e.g. weekends/holidays where no data was retrieved)
    combined.dropna(how='all', inplace=True)
    
    combined.sort_index(inplace=True)
    combined = combined.astype("float64")
    combined.to_parquet(parquet_path)

    LOGGER.info(
        "Price store updated",
        extra={
            "rows": len(combined),
            "columns": len(combined.columns),
            "start": str(combined.index.min()),
            "end": str(combined.index.max()),
        }
    )

    return combined


def prepare_analysis_ready_data(
    prices: pd.DataFrame,
    asset_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Converts ticker-based columns into semantic asset names.
    
    Args:
        prices: DataFrame with ticker symbols as columns
        asset_map: Mapping of semantic_name -> ticker
        
    Returns:
        DataFrame with semantic names as columns, float64 dtype
    """
    reverse_map = {ticker: name for name, ticker in asset_map.items()}

    # Rename columns from tickers to semantic names
    renamed = prices.rename(columns=reverse_map)

    # Check for missing assets
    available_cols = set(renamed.columns)
    expected_names = set(asset_map.keys())
    missing_assets = expected_names - available_cols
    
    if missing_assets:
        LOGGER.warning(
            "Missing assets in price matrix",
            extra={"assets": list(missing_assets), "count": len(missing_assets)}
        )

    # Derived Metrics: Yield Curve (10Y - 2Y)
    if "US_10Y" in renamed.columns and "US_2Y" in renamed.columns:
        renamed["Yield_Curve"] = renamed["US_10Y"] - renamed["US_2Y"]

    # Forward fill to handle holidays/trading halts
    renamed = renamed.ffill()
    
    renamed = renamed.sort_index()
    renamed = renamed.astype("float64")

    return renamed


def get_prices_for_date(
    prices: pd.DataFrame,
    date: str,
) -> pd.Series:
    """
    Get prices for a specific date.
    
    Args:
        prices: Price DataFrame
        date: Date string in YYYY-MM-DD format
        
    Returns:
        Series of prices for the date
        
    Raises:
        KeyError: If date not found
    """
    target_date = pd.Timestamp(date)
    if target_date not in prices.index:
        # Find nearest available date
        available_dates = prices.index[prices.index <= target_date]
        if len(available_dates) == 0:
            raise KeyError(f"No data available for or before {date}")
        target_date = available_dates[-1]
    
    return prices.loc[target_date]
