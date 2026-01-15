"""
File: src/engine/__init__.py

Engine module exports for macro transmission monitoring.
"""

from src.engine.data_engine import (
    load_yaml,
    load_asset_universe,
    flatten_universe,
    update_price_store,
    prepare_analysis_ready_data,
    DataEngineError,
)
from src.engine.returns import compute_log_returns
from src.engine.correlation import CorrelationStats, compute_rolling_correlation_zscore
from src.engine.beta import BetaStats, compute_beta_and_deviation
from src.engine.regime_detector import compute_macro_move_zscore, evaluate_alert_gate

__all__ = [
    "load_yaml",
    "load_asset_universe",
    "flatten_universe",
    "update_price_store",
    "prepare_analysis_ready_data",
    "DataEngineError",
    "compute_log_returns",
    "CorrelationStats",
    "compute_rolling_correlation_zscore",
    "BetaStats",
    "compute_beta_and_deviation",
    "compute_macro_move_zscore",
    "evaluate_alert_gate",
]
