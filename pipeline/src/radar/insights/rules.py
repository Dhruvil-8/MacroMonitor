# pipeline/src/radar/insights/rules.py
"""Observation generation rules and severity classification.

Implements Section 9.3 and Section 9.4.
"""

from typing import Any


def evaluate_severity(val: float) -> str:
    """Classify alert severity into low, medium, or high.

    Handles either a percentile in [0, 100] or a z-score in [-10, 10].

    Parameters
    ----------
    val : float
        Percentile (e.g. 96.5) or z-score (e.g. 3.2).

    Returns
    -------
    'low' | 'medium' | 'high'
    """
    v = float(val)
    # If v <= 10.0, treat as z-score
    if v <= 10.0:
        abs_z = abs(v)
        if abs_z >= 3.5:
            return "high"
        elif abs_z >= 2.5:
            return "medium"
        elif abs_z >= 2.0:
            return "low"
        return "low"
    else:
        # Treat as percentile (0-100)
        if v >= 99.0:
            return "high"
        elif v >= 95.0:
            return "medium"
        elif v >= 90.0:
            return "low"
        return "low"
