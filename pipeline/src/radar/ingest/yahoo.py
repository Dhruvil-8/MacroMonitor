# pipeline/src/radar/ingest/yahoo.py
"""Yahoo Finance Ingestor.

Batch chunked downloader with rate-limit protection, exponential backoff, and fallback symbol support.
"""

import time
from typing import Any
import pandas as pd
import yfinance as yf

from radar.ingest.base import BaseIngestor


class YahooIngestor(BaseIngestor):
    def __init__(self, cache_dir: Any = None, chunk_size: int = 5, delay: float = 0.15) -> None:
        super().__init__("yahoo", cache_dir)
        self.chunk_size = chunk_size
        self.delay = delay

    def fetch_series(
        self,
        symbol: str,
        start_date: str = "2000-01-01",
        end_date: str | None = None,
        fallback_symbol: str | None = None,
    ) -> pd.DataFrame:
        """Fetch a single symbol from Yahoo with fallback."""
        df = self._fetch_single(symbol, start_date, end_date)
        if df.empty and fallback_symbol:
            alt_sym = fallback_symbol.split(":")[-1]
            df = self._fetch_single(alt_sym, start_date, end_date)

        # Cache raw response
        if not df.empty:
            cache_file = self.get_cache_path(symbol)
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass

        return df

    def _fetch_single(
        self, symbol: str, start_date: str, end_date: str | None
    ) -> pd.DataFrame:
        for attempt in range(3):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                if not hist.empty and "Close" in hist.columns:
                    out = hist[["Close"]].rename(columns={"Close": "value"})
                    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
                    out = out[~out.index.duplicated(keep="last")]
                    return out
            except Exception:
                time.sleep(1.0 * (2**attempt))
        return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([]))

    def fetch_batch(
        self,
        symbols: list[str],
        start_date: str = "2000-01-01",
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch multiple symbols in respectful chunks to prevent rate limiting."""
        results = {}
        for i in range(0, len(symbols), self.chunk_size):
            chunk = symbols[i : i + self.chunk_size]
            for sym in chunk:
                df = self.fetch_series(sym, start_date=start_date, end_date=end_date)
                results[sym] = df
                time.sleep(self.delay)
        return results
