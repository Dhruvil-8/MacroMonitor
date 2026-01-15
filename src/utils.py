"""
File: src/utils.py

Utility functions for macro transmission monitoring.

Provides:
- Configuration validation
- Structured JSON logging setup
- Common helper functions
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pythonjsonlogger import jsonlogger


def setup_json_logging(
    level: int = logging.INFO,
    module_name: str = "macro_monitor",
) -> logging.Logger:
    """
    Configure structured JSON logging per specification.
    
    Log format:
    {
        "timestamp": "2026-01-15T10:00:00Z",
        "module": "engine.data_engine",
        "level": "INFO",
        "message": "price store updated",
        "metrics": {"rows": 1234}
    }
    
    Args:
        level: Logging level
        module_name: Root module name
        
    Returns:
        Configured root logger
    """
    
    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        """Custom formatter for structured JSON logs."""
        
        def add_fields(
            self,
            log_record: Dict[str, Any],
            record: logging.LogRecord,
            message_dict: Dict[str, Any],
        ) -> None:
            super().add_fields(log_record, record, message_dict)
            
            # Add structured fields
            log_record["timestamp"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            log_record["module"] = record.name
            log_record["level"] = record.levelname
            
            # Move extra fields to metrics
            metrics = {}
            keys_to_remove = []
            
            for key, value in log_record.items():
                if key not in {"timestamp", "module", "level", "message"}:
                    metrics[key] = value
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del log_record[key]
            
            if metrics:
                log_record["metrics"] = metrics
    
    # Configure root logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Add JSON handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = CustomJsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Configure all child loggers
    for name in ["src.engine", "src.alerts", "src.dashboard_api"]:
        child = logging.getLogger(name)
        child.setLevel(level)
        child.handlers.clear()
        child.addHandler(handler)
    
    return logger


def validate_config(
    asset_universe: Dict[str, Dict[str, str]],
    pairing_matrix: List[Dict[str, str]],
) -> None:
    """
    Validate configuration files on startup.
    
    Fails fast with descriptive error if:
    - Any macro in pairing_matrix is missing from asset_universe
    - Any target in pairing_matrix is missing from asset_universe
    - Pairing is missing required fields
    
    Args:
        asset_universe: Loaded asset universe config
        pairing_matrix: Loaded pairing matrix config
        
    Raises:
        ValueError: If validation fails
    """
    # Flatten universe to get all valid names
    valid_names: Set[str] = set()
    valid_groups: Set[str] = set()
    
    for group_name, group_assets in asset_universe.items():
        valid_groups.add(group_name)
        if isinstance(group_assets, dict):
            for asset_name in group_assets.keys():
                valid_names.add(asset_name)
    
    # Also allow group names as targets (e.g., "Europe" -> equities)
    all_valid = valid_names | valid_groups
    
    errors: List[str] = []
    
    for i, pairing in enumerate(pairing_matrix):
        # Check required fields
        if "macro" not in pairing:
            errors.append(f"Pairing {i}: missing 'macro' field")
            continue
        if "target" not in pairing:
            errors.append(f"Pairing {i}: missing 'target' field")
            continue
        
        macro = pairing["macro"]
        target = pairing["target"]
        
        # Validate macro exists
        if macro not in all_valid:
            errors.append(
                f"Pairing {i}: macro '{macro}' not found in asset_universe"
            )
        
        # Validate target exists
        if target not in all_valid:
            errors.append(
                f"Pairing {i}: target '{target}' not found in asset_universe"
            )
        
        # Check rationale (warning only)
        if "rationale" not in pairing:
            errors.append(
                f"Pairing {i} ({macro}->{target}): missing 'rationale' field"
            )
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


def resolve_asset_name(
    name: str,
    asset_universe: Dict[str, Dict[str, str]],
    flat_universe: Dict[str, str],
) -> Optional[str]:
    """
    Resolve an asset name to its semantic name.
    
    Handles both direct asset names and group names.
    
    Args:
        name: Asset or group name
        asset_universe: Full nested universe
        flat_universe: Flattened universe mapping
        
    Returns:
        Resolved asset name, or None if not found
    """
    # Direct match in flat universe
    if name in flat_universe:
        return name
    
    # Group name - return first asset in group
    if name in asset_universe:
        group = asset_universe[name]
        if isinstance(group, dict) and group:
            return next(iter(group.keys()))
    
    return None


def get_trading_date(
    date_str: Optional[str] = None,
) -> str:
    """
    Get trading date in YYYY-MM-DD format.
    
    Args:
        date_str: Optional date string, or None for today
        
    Returns:
        Date string in YYYY-MM-DD format
    """
    if date_str:
        # Validate format
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")
    
    return datetime.now().strftime("%Y-%m-%d")


def format_percent(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:+.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide, returning default on zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))
