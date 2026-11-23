# pipeline/src/radar/ingest/nse.py
"""NSE (National Stock Exchange of India) Ingestor.

Fetches Indian indices and derivatives/flow metrics.
Provides reliable fallback to Yahoo Finance Indian ticker proxies (^NSEI, ^NSEBANK, etc.).
"""

from typing import Any
import pandas as pd

from radar.ingest.base import BaseIngestor
from radar.ingest.yahoo import YahooIngestor


class NSEIngestor(BaseIngestor):
    # Mapping from NSE symbol to reliable Yahoo proxy
    YAHOO_PROXIES = {
        "NIFTY 50": "^NSEI",
        "NIFTY BANK": "^NSEBANK",
        "NIFTY IT": "^CNXIT",
        "NIFTY MIDCAP 50": "^NSEMDCP50",
        "INDIA VIX": "^INDIAVIX",
    }

    def __init__(self, cache_dir: Any = None) -> None:
        super().__init__("nse", cache_dir)
        self.yahoo_fallback = YahooIngestor(cache_dir=cache_dir)

    def fetch_series(
        self,
        symbol: str,
        start_date: str = "2005-01-01",
        end_date: str | None = None,
        fallback_symbol: str | None = None,
    ) -> pd.DataFrame:
        """Fetch Indian market series with proxy fallback."""
        proxy = self.YAHOO_PROXIES.get(symbol)
        if not proxy and fallback_symbol:
            proxy = fallback_symbol.split(":")[-1]

        if proxy:
            df = self.yahoo_fallback.fetch_series(proxy, start_date=start_date, end_date=end_date)
            if not df.empty:
                try:
                    df.to_parquet(self.get_cache_path(symbol))
                except Exception:
                    pass
                return df

        return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([]))
