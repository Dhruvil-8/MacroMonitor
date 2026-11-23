# pipeline/src/radar/ingest/fred.py
"""FRED (Federal Reserve Bank of St. Louis) Ingestor.

Uses official FRED direct CSV endpoint (public, zero-key required)
or FRED API when FRED_API_KEY environment variable is provided.
"""

import os
import time
from typing import Any
import pandas as pd
import requests

from radar.ingest.base import BaseIngestor


class FREDIngestor(BaseIngestor):
    def __init__(self, cache_dir: Any = None) -> None:
        super().__init__("fred", cache_dir)
        self.api_key = os.environ.get("FRED_API_KEY")

    def fetch_series(
        self,
        symbol: str,
        start_date: str = "2000-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch macroeconomic series from FRED using API key if provided or public CSV."""
        headers = {
            "User-Agent": "MacroMonitor/1.0 (macro-statistics-tool; contact@macromonitor.org)"
        }

        # 1. If FRED_API_KEY is present, query official FRED REST endpoint
        if self.api_key:
            api_url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={symbol}&api_key={self.api_key}&file_type=json"
                f"&observation_start={start_date}"
            )
            if end_date:
                api_url += f"&observation_end={end_date}"
            try:
                resp = requests.get(api_url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    obs = data.get("observations", [])
                    if obs:
                        df = pd.DataFrame(obs)[["date", "value"]]
                        df["value"] = pd.to_numeric(df["value"], errors="coerce")
                        df = df.dropna(subset=["value"])
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date").sort_index()
                        try:
                            df.to_parquet(self.get_cache_path(symbol))
                        except Exception:
                            pass
                        return df[["value"]]
            except Exception as e:
                pass

        # 2. Public FRED CSV download fallback (zero-key required)
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={symbol}"
        headers = {
            "User-Agent": "MacroMonitor/1.0 (macro-statistics-tool; contact@macromonitor.org)"
        }

        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200 and len(resp.content) > 50:
                    from io import StringIO
                    csv_data = StringIO(resp.text)
                    df = pd.read_csv(csv_data)
                    if len(df.columns) >= 2:
                        date_col = df.columns[0]
                        val_col = df.columns[1]
                        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
                        df = df.dropna(subset=[val_col])
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df.set_index(date_col).rename(columns={val_col: "value"})
                        df = df.sort_index()

                        # Filter by start_date
                        df = df[df.index >= pd.to_datetime(start_date)]
                        if end_date:
                            df = df[df.index <= pd.to_datetime(end_date)]

                        # Cache raw data
                        try:
                            df.to_parquet(self.get_cache_path(symbol))
                        except Exception:
                            pass

                        return df[["value"]]
            except Exception:
                time.sleep(1.0 * (2**attempt))

        return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([]))
