"""
File: scripts/validate_scenarios.py

Historical scenario validation and replay scripts.

Validates system against known macro events:
- March 2020 COVID crash
- Feb-Mar 2022 Russia/Ukraine energy shock  
- Nov 2022 USD/rates spike
- Oil crash 2014-2015

Usage:
    python scripts/validate_scenarios.py                  # Run all scenarios
    python scripts/validate_scenarios.py --scenario covid # Run specific scenario
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.data_engine import load_asset_universe, flatten_universe, PRICES_PATH, prepare_analysis_ready_data
from src.engine.returns import compute_mixed_returns
from src.engine.correlation import compute_rolling_correlation_zscore
from src.engine.beta import compute_beta_and_deviation
from src.engine.regime_detector import (
    compute_macro_move_zscore,
    evaluate_alert_gate,
    determine_severity,
)
from src.alerts.formatter import Alert, save_alerts, format_alert_text
from src.utils import setup_json_logging, get_trading_date


# Define historical scenarios
SCENARIOS = {
    "covid_march_2020": {
        "name": "March 2020 COVID Crash",
        "start_date": "2020-03-01",
        "end_date": "2020-03-31",
        "key_dates": ["2020-03-09", "2020-03-12", "2020-03-16", "2020-03-20"],
        "expected_pairs": [
            ("VIX", "SP500"),
            ("VIX", "HangSeng"),
            ("DXY", "EM_Equities"),
        ],
        "description": "Global equity crash with VIX spike to 82",
    },
    "ukraine_energy_2022": {
        "name": "Feb-Mar 2022 Russia/Ukraine Energy Shock",
        "start_date": "2022-02-20",
        "end_date": "2022-03-15",
        "key_dates": ["2022-02-24", "2022-03-07", "2022-03-08"],
        "expected_pairs": [
            ("Brent", "Europe"),
            ("NatGas", "Europe"),
            ("DXY", "EURUSD"),
        ],
        "description": "Energy price spike and European asset stress",
    },
    "usd_rates_spike_2022": {
        "name": "Nov 2022 USD/Rates Spike",
        "start_date": "2022-10-15",
        "end_date": "2022-11-30",
        "key_dates": ["2022-11-03", "2022-11-10"],
        "expected_pairs": [
            ("DXY", "EM_Equities"),
            ("US_10Y", "NASDAQ"),
            ("US_10Y", "Gold"),
        ],
        "description": "Fed hawkishness driving dollar strength",
    },
    "oil_crash_2014": {
        "name": "Oil Crash 2014-2015",
        "start_date": "2014-09-01",
        "end_date": "2015-02-28",
        "key_dates": ["2014-11-28", "2014-12-16", "2015-01-05"],
        "expected_pairs": [
            ("Brent", "US_HighYield"),
            ("WTI", "Russell2000"),
        ],
        "description": "Oil price collapse affecting energy-heavy credit",
    },
}


# Output directory
VALIDATION_DIR = Path("data/processed/validation")


def run_scenario(
    scenario_id: str,
    scenario: Dict,
    prices: pd.DataFrame,
    asset_map: Dict[str, str],
    thresholds: Dict,
    logger,
) -> Dict:
    """
    Run validation for a specific historical scenario.
    
    Returns dict with scenario results.
    """
    logger.info(f"Running scenario: {scenario['name']}")
    
    results = {
        "scenario_id": scenario_id,
        "name": scenario["name"],
        "description": scenario["description"],
        "start_date": scenario["start_date"],
        "end_date": scenario["end_date"],
        "alerts": [],
        "expected_pairs": scenario["expected_pairs"],
        "pairs_detected": [],
        "success": False,
    }
    
    # Filter prices to scenario date range
    start = pd.Timestamp(scenario["start_date"])
    end = pd.Timestamp(scenario["end_date"])
    
    # Need historical data for lookback, so expand start
    lookback_days = thresholds.get("hist_corr_window", 180) + 30
    data_start = start - pd.Timedelta(days=lookback_days)
    
    scenario_prices = prices[(prices.index >= data_start) & (prices.index <= end)]
    
    scenario_prices = prices[(prices.index >= data_start) & (prices.index <= end)]
    
    if len(scenario_prices) < 100:
        logger.warning(f"Insufficient data for scenario: {scenario_id}")
        results["error"] = "Insufficient data"
        return results
    
    # Prepare data (yield curve, etc)
    scenario_prices = prepare_analysis_ready_data(scenario_prices, asset_map)

    # Compute returns
    linear_assets = ["Yield_Curve", "US_10Y", "US_2Y", "US_30Y"]
    returns = compute_mixed_returns(scenario_prices, linear_assets)
    
    # Load pairings
    from src.engine.data_engine import load_yaml
    pairings = load_yaml(Path("config/pairing_matrix.yaml"))
    
    # Analyze key dates
    for key_date in scenario["key_dates"]:
        date_ts = pd.Timestamp(key_date)
        
        if date_ts not in returns.index:
            continue
        
        # Get returns up to this date
        date_returns = returns[returns.index <= date_ts]
        
        # Check each pairing
        for pairing in pairings:
            macro_name = pairing["macro"]
            target_name = pairing["target"]
            
            if macro_name not in date_returns.columns:
                print(f"DEBUG: {macro_name} not in columns")
                continue
            if target_name not in date_returns.columns:
                print(f"DEBUG: {target_name} not in columns")
                continue
            
            macro_returns = date_returns[macro_name].dropna()
            target_returns = date_returns[target_name].dropna()
            
            try:
                # Compute metrics
                corr_stats = compute_rolling_correlation_zscore(
                    macro_returns, target_returns,
                    corr_window=thresholds.get("corr_window", 30),
                    hist_window=thresholds.get("hist_corr_window", 180),
                )
                
                beta_stats = compute_beta_and_deviation(
                    macro_returns, target_returns,
                    beta_window=thresholds.get("beta_window", 180),
                    deviation_params=thresholds.get("deviation_threshold_params", {}),
                )
                
                macro_move_z = compute_macro_move_zscore(
                    macro_returns,
                    vol_window=thresholds.get("macro_vol_window", 180),
                )
                
                # Check directly
                if macro_name == "VIX" and target_name == "Bitcoin":
                    print(f"DEBUG Check {key_date}: z={corr_stats.z_score:.2f}, macro_z={macro_move_z:.2f}, dev={beta_stats.deviation_flag}")
                
                # Check if alert triggers
                if evaluate_alert_gate(
                    corr_stats.z_score,
                    macro_move_z,
                    beta_stats.deviation_flag,
                    thresholds,
                ):
                    alert = Alert(
                        date=key_date,
                        macro=macro_name,
                        target=target_name,
                        rationale=pairing.get("rationale", ""),
                        mean_corr=corr_stats.mean_hist,
                        curr_corr=corr_stats.current_corr,
                        corr_z_score=corr_stats.z_score,
                        beta=beta_stats.beta,
                        macro_move_pct=float(macro_returns.iloc[-1]) * 100,
                        target_move_pct=float(target_returns.iloc[-1]) * 100,
                        deviation=beta_stats.deviation_flag,
                        severity=determine_severity(corr_stats.z_score, macro_move_z),
                    )
                    
                    results["alerts"].append(alert.to_dict())
                    
                    pair_tuple = (macro_name, target_name)
                    if pair_tuple not in results["pairs_detected"]:
                        results["pairs_detected"].append(pair_tuple)
                        
            except (ValueError, Exception) as e:
                continue
    
    # Check success criteria
    expected_set = set(tuple(p) for p in scenario["expected_pairs"])
    detected_set = set(tuple(p) for p in results["pairs_detected"])
    
    # Success if at least one expected pair detected
    results["expected_detected"] = list(expected_set & detected_set)
    results["success"] = len(results["expected_detected"]) > 0
    
    logger.info(
        f"Scenario complete: {scenario_id}",
        extra={
            "alerts_generated": len(results["alerts"]),
            "expected_pairs": len(scenario["expected_pairs"]),
            "detected_pairs": len(results["pairs_detected"]),
            "success": results["success"],
        }
    )
    
    return results


def generate_report(all_results: List[Dict]) -> str:
    """Generate markdown validation report."""
    lines = [
        "# Macro Transmission Monitor - Validation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
    ]
    
    passed = sum(1 for r in all_results if r.get("success"))
    total = len(all_results)
    
    lines.append(f"- **Scenarios Tested**: {total}")
    lines.append(f"- **Passed**: {passed}")
    lines.append(f"- **Failed**: {total - passed}")
    lines.append("")
    
    for result in all_results:
        status = "✅ PASSED" if result.get("success") else "❌ FAILED"
        lines.append(f"## {result['name']} {status}")
        lines.append("")
        lines.append(f"*{result['description']}*")
        lines.append("")
        lines.append(f"- Period: {result['start_date']} to {result['end_date']}")
        lines.append(f"- Alerts Generated: {len(result.get('alerts', []))}")
        lines.append(f"- Expected Pairs: {result['expected_pairs']}")
        lines.append(f"- Detected Pairs: {result.get('pairs_detected', [])}")
        lines.append("")
        
        if result.get("alerts"):
            lines.append("### Sample Alerts")
            lines.append("")
            for alert in result["alerts"][:3]:  # Show first 3
                lines.append(f"- **{alert['date']}**: {alert['macro']} → {alert['target']} (z={alert['corr_z_score']:.2f}, severity={alert['severity']})")
            lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Historical scenario validation")
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=list(SCENARIOS.keys()),
        help="Run specific scenario only",
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_json_logging()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if not PRICES_PATH.exists():
        logger.error("No price data found. Run backfill first.")
        sys.exit(1)
    
    prices = pd.read_parquet(PRICES_PATH)
    
    # Load config
    from src.engine.data_engine import load_yaml
    universe = load_yaml(Path("config/asset_universe.yaml"))
    asset_map = flatten_universe(universe)
    thresholds = load_yaml(Path("config/thresholds.yaml"))
    
    # Rename handled by prepare_analysis_ready_data inside run_scenario now?
    # No, run_scenario takes 'prices' (raw) and 'asset_map'.
    # But wait, run_scenario expects raw prices?
    # In my edit above, I added prepare_analysis_ready_data inside run_scenario.
    # So I should remove the manual rename here.
    
    # prices = prices.rename(columns=reverse_map) # REMOVED
    
    # Run scenarios
    scenarios_to_run = {args.scenario: SCENARIOS[args.scenario]} if args.scenario else SCENARIOS
    
    all_results = []
    for scenario_id, scenario in scenarios_to_run.items():
        result = run_scenario(
            scenario_id, scenario,
            prices, asset_map, thresholds,
            logger,
        )
        all_results.append(result)
        
        # Save individual result
        result_path = VALIDATION_DIR / f"{scenario_id}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
    
    # Generate report
    report = generate_report(all_results)
    report_path = VALIDATION_DIR / f"report-{datetime.now().strftime('%Y%m%d')}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    # Safely print report to console
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode('ascii', 'replace').decode('ascii'))
    logger.info(f"Report saved to {report_path}")
    
    # Exit with failure if any scenario failed
    if not all(r.get("success") for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
