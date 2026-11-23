# pipeline/tests/unit/test_ingest.py
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from radar.ingest.yahoo import YahooIngestor
from radar.ingest.fred import FREDIngestor
from radar.ingest.nse import NSEIngestor


def test_yahoo_ingestor_parses_close_and_caches(tmp_path):
    ingestor = YahooIngestor(cache_dir=tmp_path)

    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    mock_hist = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_hist
        mock_ticker_cls.return_value = mock_ticker

        df = ingestor.fetch_series("^GSPC", start_date="2026-01-01")

        assert not df.empty
        assert "value" in df.columns
        assert len(df) == 5
        assert df.iloc[0]["value"] == 100.0


def test_fred_ingestor_parses_csv_and_returns_df(tmp_path):
    ingestor = FREDIngestor(cache_dir=tmp_path)

    csv_text = "DATE,DGS10\n2026-01-02,4.50\n2026-01-05,4.55\n2026-01-06,.\n2026-01-07,4.52\n"

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = csv_text.encode("utf-8")
        mock_resp.text = csv_text
        mock_get.return_value = mock_resp

        df = ingestor.fetch_series("DGS10", start_date="2026-01-01")

        assert not df.empty
        assert "value" in df.columns
        # 4.50, 4.55, 4.52 ('.' is NaN and dropped)
        assert len(df) == 3
        assert df.iloc[0]["value"] == 4.50


def test_nse_ingestor_uses_proxy(tmp_path):
    ingestor = NSEIngestor(cache_dir=tmp_path)

    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    mock_df = pd.DataFrame({"value": [24000.0, 24100.0, 24200.0]}, index=dates)

    with patch.object(ingestor.yahoo_fallback, "fetch_series", return_value=mock_df) as mock_fetch:
        df = ingestor.fetch_series("NIFTY 50", start_date="2026-01-01")

        assert not df.empty
        assert df.iloc[0]["value"] == 24000.0
        mock_fetch.assert_called_once_with("^NSEI", start_date="2026-01-01", end_date=None)
