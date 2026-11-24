# pipeline/tests/unit/test_cli.py
from unittest.mock import MagicMock, patch
import pandas as pd
from radar.cli import build_parser, main
from radar.db.store import DataStore


def test_cli_parser_help():
    parser = build_parser()
    assert parser.prog == "radar"


def test_cli_subcommands(tmp_path):
    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    mock_df = pd.DataFrame({"value": [100.0, 101.0, 102.0]}, index=dates)
    mock_manifest = MagicMock()
    mock_manifest.manifest.datasets = []

    mock_store = DataStore(base_dir=tmp_path)

    with patch("radar.cli.DataStore", return_value=mock_store), \
         patch("radar.ingest.yahoo.YahooIngestor.fetch_series", return_value=mock_df), \
         patch("radar.ingest.fred.FREDIngestor.fetch_series", return_value=mock_df), \
         patch("radar.ingest.nse.NSEIngestor.fetch_series", return_value=mock_df), \
         patch("radar.export.writers.export_all_datasets", return_value=mock_manifest), \
         patch("radar.metrics.run_analogues.run", return_value=None):

        assert main(["ingest", "--source", "fred"]) == 0
        assert main(["compute", "--metrics", "all"]) == 0
        assert main(["verify"]) == 0
        assert main(["export", "--target", "local"]) == 0
        assert main(["all"]) == 0
