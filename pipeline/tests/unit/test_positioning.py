"""Unit tests for CFTC COT positioning metrics and no look-ahead bias."""

from __future__ import annotations

import unittest

from radar.ingest.cftc import parse_cftc_rows
from radar.metrics.positioning import compute_cot_contract_metrics


class TestCftcPositioning(unittest.TestCase):
    def test_cftc_parser_and_no_lookahead(self) -> None:
        raw = [
            {
                "cftc_contract_market_code": "13874A",
                "report_date_as_yyyy_mm_dd": f"2026-01-{i + 1:02d}T00:00:00.000",
                "open_interest_all": "1000",
                "noncomm_positions_long_all": str(300 + i * 10),
                "noncomm_positions_short_all": "200",
                "comm_positions_long_all": "400",
                "comm_positions_short_all": "500",
            }
            for i in range(10)
        ]
        grouped = parse_cftc_rows(raw)
        rows = grouped["13874A"]
        self.assertEqual(len(rows), 10)

        full = compute_cot_contract_metrics(rows, min_obs=4)
        # Index 5: long = 350, short = 200 -> net = 150, OI = 1000 -> net_pct_oi = 0.15
        self.assertEqual(full[5]["net_speculative"], 150)
        self.assertAlmostEqual(full[5]["net_pct_oi"], 0.15, places=4)
        self.assertEqual(full[5]["percentile"], 100.0)

        # Look-ahead check at index 6
        trunc = compute_cot_contract_metrics(rows[:7], min_obs=4)
        self.assertEqual(full[6]["percentile"], trunc[6]["percentile"])
        self.assertEqual(full[6]["z_score"], trunc[6]["z_score"])


if __name__ == "__main__":
    unittest.main()
