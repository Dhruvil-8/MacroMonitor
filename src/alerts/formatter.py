"""
File: src/alerts/formatter.py

Alert formatting, validation, and persistence for macro transmission monitoring.

Provides:
- Alert dataclass with all required fields
- JSON schema for validation
- Human-readable text formatter
- JSON formatter with schema validation
- Persistence to JSONL and latest.json files
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import jsonschema


LOGGER = logging.getLogger(__name__)

# Alert output directories
ALERTS_DIR = Path("data/processed/alerts")


# JSON Schema for MacroTransmissionAlert
ALERT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "MacroTransmissionAlert",
    "type": "object",
    "properties": {
        "date": {"type": "string", "format": "date"},
        "macro": {"type": "string"},
        "target": {"type": "string"},
        "rationale": {"type": "string"},
        "mean_corr": {"type": "number"},
        "curr_corr": {"type": "number"},
        "corr_z_score": {"type": "number"},
        "beta": {"type": "number"},
        "macro_move_pct": {"type": "number"},
        "target_move_pct": {"type": "number"},
        "deviation": {"type": "boolean"},
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        "extra": {"type": "object"},
    },
    "required": [
        "date",
        "macro",
        "target",
        "mean_corr",
        "curr_corr",
        "corr_z_score",
        "beta",
        "macro_move_pct",
        "target_move_pct",
        "deviation",
        "severity",
    ],
}


@dataclass
class Alert:
    """
    Represents a macro transmission alert.
    
    All numeric values are percentages where applicable.
    """
    date: str                  # ISO date (YYYY-MM-DD)
    macro: str                 # Macro driver semantic name
    target: str                # Target asset semantic name
    rationale: str             # Explanation of the relationship
    mean_corr: float           # Historical mean correlation
    curr_corr: float           # Current rolling correlation
    corr_z_score: float        # Z-score of correlation vs history
    beta: float                # Rolling beta coefficient
    macro_move_pct: float      # Macro move as percentage
    target_move_pct: float     # Target move as percentage
    deviation: bool            # Whether deviation threshold breached
    severity: str              # low, medium, or high
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def format_alert_text(alert: Alert) -> str:
    """
    Format alert as human-readable multi-line text.
    
    Suitable for email notifications or console output.
    
    Args:
        alert: Alert object
        
    Returns:
        Multi-line formatted string
    """
    severity_emoji = {
        "high": "🔴",
        "medium": "🟡",
        "low": "🟢",
    }
    
    emoji = severity_emoji.get(alert.severity, "⚪")
    
    # Determine direction of move
    macro_direction = "↑" if alert.macro_move_pct > 0 else "↓"
    target_direction = "↑" if alert.target_move_pct > 0 else "↓"
    
    # Determine if correlation broke up or down
    if alert.corr_z_score > 0:
        corr_signal = "correlation spiked ABOVE historical norm"
    else:
        corr_signal = "correlation dropped BELOW historical norm"
    
    text = f"""
{emoji} MACRO TRANSMISSION ALERT [{alert.severity.upper()}]
═══════════════════════════════════════════════════════════════

Date:       {alert.date}
Pair:       {alert.macro} → {alert.target}

WHAT HAPPENED:
The historical relationship between {alert.macro} and {alert.target} has broken down.
{alert.rationale}

METRICS:
  • Macro Move:         {alert.macro_move_pct:+.2f}% {macro_direction}
  • Target Move:        {alert.target_move_pct:+.2f}% {target_direction}
  • Current Correlation: {alert.curr_corr:.3f}
  • Historical Mean:     {alert.mean_corr:.3f}
  • Correlation Z-Score: {alert.corr_z_score:+.2f}σ ({corr_signal})
  • Beta:               {alert.beta:.3f}
  • Deviation Flag:     {"YES - target deviated from expected move" if alert.deviation else "No"}

INTERPRETATION:
{_generate_interpretation(alert)}

───────────────────────────────────────────────────────────────
""".strip()
    
    return text


def _generate_interpretation(alert: Alert) -> str:
    """Generate human-readable interpretation of the alert."""
    parts = []
    
    # Z-score interpretation
    if abs(alert.corr_z_score) >= 3.0:
        parts.append(f"The {abs(alert.corr_z_score):.1f}σ correlation deviation is EXTREME and rare.")
    elif abs(alert.corr_z_score) >= 2.0:
        parts.append(f"The {abs(alert.corr_z_score):.1f}σ correlation deviation is significant.")
    
    # Beta interpretation
    expected_move = alert.beta * alert.macro_move_pct
    actual_move = alert.target_move_pct
    
    if alert.deviation:
        if actual_move > expected_move:
            parts.append(
                f"{alert.target} moved MORE than expected "
                f"(expected {expected_move:+.2f}%, got {actual_move:+.2f}%)."
            )
        else:
            parts.append(
                f"{alert.target} moved LESS than expected "
                f"(expected {expected_move:+.2f}%, got {actual_move:+.2f}%)."
            )
    
    # Macro context
    if abs(alert.macro_move_pct) > 2.0:
        parts.append(f"{alert.macro} had an unusually large move of {alert.macro_move_pct:+.2f}%.")
    
    return " ".join(parts) if parts else "Standard transmission breakdown detected."


def format_alert_json(alert: Alert) -> Dict[str, Any]:
    """
    Format alert as JSON-serializable dictionary.
    
    Validates against ALERT_SCHEMA before returning.
    
    Args:
        alert: Alert object
        
    Returns:
        Dictionary validated against JSON schema
        
    Raises:
        jsonschema.ValidationError: If alert does not match schema
    """
    data = alert.to_dict()
    
    # Ensure numeric precision and native types (handling numpy types)
    for key in ["mean_corr", "curr_corr", "corr_z_score", "beta", "macro_move_pct", "target_move_pct"]:
        if key in data and data[key] is not None:
            data[key] = round(float(data[key]), 6)
            
    # Ensure booleans are native bools (not np.bool_)
    if "deviation" in data:
        data["deviation"] = bool(data["deviation"])
        
    # Handle extra dictionary recursively if needed, specifically for known fields
    if "extra" in data and isinstance(data["extra"], dict):
        for k, v in data["extra"].items():
            if hasattr(v, "item"):  # numpy scalar
                data["extra"][k] = v.item()
    
    # Validate against schema
    jsonschema.validate(instance=data, schema=ALERT_SCHEMA)
    
    return data


def validate_alert(alert_dict: Dict[str, Any]) -> bool:
    """
    Validate an alert dictionary against the schema.
    
    Args:
        alert_dict: Alert as dictionary
        
    Returns:
        True if valid, False otherwise
    """
    try:
        jsonschema.validate(instance=alert_dict, schema=ALERT_SCHEMA)
        return True
    except jsonschema.ValidationError:
        return False


def save_alerts(
    alerts: List[Alert],
    date: str,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Save alerts to persistent storage.
    
    Creates two files:
    1. data/processed/alerts/YYYY-MM-DD.jsonl - Newline-delimited JSON for the date
    2. data/processed/alerts/latest.json - Most recent alerts as JSON array
    
    Args:
        alerts: List of Alert objects
        date: Date string in YYYY-MM-DD format
        output_dir: Override output directory (default: data/processed/alerts/)
        
    Returns:
        Dict with paths to created files
    """
    if output_dir is None:
        output_dir = ALERTS_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert alerts to JSON
    alert_dicts = []
    for alert in alerts:
        try:
            alert_dict = format_alert_json(alert)
            alert_dicts.append(alert_dict)
        except jsonschema.ValidationError as e:
            LOGGER.error(f"Invalid alert skipped: {e.message}")
            continue
    
    # Save JSONL file for the date
    jsonl_path = output_dir / f"{date}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for alert_dict in alert_dicts:
            f.write(json.dumps(alert_dict, ensure_ascii=False) + "\n")
    
    # Save latest.json
    latest_path = output_dir / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": date,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "count": len(alert_dicts),
                "alerts": alert_dicts,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    
    LOGGER.info(
        "Alerts saved",
        extra={
            "date": date,
            "count": len(alert_dicts),
            "jsonl_path": str(jsonl_path),
            "latest_path": str(latest_path),
        }
    )
    
    return {
        "jsonl": jsonl_path,
        "latest": latest_path,
    }


def load_alerts(
    date: Optional[str] = None,
    alerts_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load alerts from storage.
    
    Args:
        date: Specific date to load (YYYY-MM-DD), or None for latest
        alerts_dir: Override alerts directory
        
    Returns:
        List of alert dictionaries
    """
    if alerts_dir is None:
        alerts_dir = ALERTS_DIR
    
    if date is None:
        # Load latest.json
        latest_path = alerts_dir / "latest.json"
        if not latest_path.exists():
            return []
        
        with open(latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data.get("alerts", [])
    else:
        # Load specific date JSONL
        jsonl_path = alerts_dir / f"{date}.jsonl"
        if not jsonl_path.exists():
            return []
        
        alerts = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    alerts.append(json.loads(line))
        
        return alerts


def get_alert_history(
    days: int = 30,
    alerts_dir: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load alert history for multiple days.
    
    Args:
        days: Number of days to look back
        alerts_dir: Override alerts directory
        
    Returns:
        Dict mapping dates to alert lists
    """
    if alerts_dir is None:
        alerts_dir = ALERTS_DIR
    
    if not alerts_dir.exists():
        return {}
    
    history = {}
    
    for jsonl_file in sorted(alerts_dir.glob("*.jsonl"), reverse=True)[:days]:
        date = jsonl_file.stem
        alerts = load_alerts(date=date, alerts_dir=alerts_dir)
        if alerts:
            history[date] = alerts
    
    return history
