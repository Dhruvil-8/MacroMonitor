"""Financial Stress Composite Metrics Module.

Section 8.7:
- Components: Equity (VIX percentile, SPX drawdown percentile), Rates (10Y yield vol percentile),
  FX vol percentile, Commodity vol percentile, Cross-asset correlation percentile.
- Component score = empirical percentile rank in [0, 100].
- Composite score = mean of available components without imputation. Missing components dropped and listed.
- Regimes: Low Stress (< 40), Normal (40-70), Elevated (70-85), High Macro Stress (>= 85).
- Known crisis episodes validation.
"""

from typing import Any
import numpy as np
import pandas as pd


def compute_stress_composite(
    components: dict[str, pd.Series],
    min_components: int = 3,
) -> pd.DataFrame:
    """Compute Financial Stress Composite as the mean of available components.

    Strict zero-imputation policy: If a component is missing at date t, it is dropped
    from the average, and only non-NaN components contribute.

    Args:
        components: Dict mapping component name to percentile Series (0-100).
        min_components: Minimum number of available components required (default 3).

    Returns:
        DataFrame containing each component, 'composite' score, and 'available_count'.
    """
    df = pd.DataFrame(components)
    # Count non-NaN components per row
    available_count = df.notna().sum(axis=1)

    # Mean of available columns without imputation
    composite = df.mean(axis=1, skipna=True)

    # If below min_components, mark as NaN
    composite = composite.where(available_count >= min_components, np.nan)

    df["composite"] = composite
    df["available_count"] = available_count
    return df


def classify_stress_regime(score: float) -> str:
    """Classify stress score into 4 discrete macro regimes.

    Args:
        score: Stress score in [0, 100].

    Returns:
        One of 'Low Stress', 'Normal Market Conditions', 'Elevated Stress', or 'High Macro Stress'.
    """
    if np.isnan(score):
        return "Unknown"
    if score < 40.0:
        return "Low Stress"
    elif score < 70.0:
        return "Normal Market Conditions"
    elif score < 85.0:
        return "Elevated Stress"
    else:
        return "High Macro Stress"


def validate_crisis_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate that historical crisis episodes align with elevated stress thresholds.

    Historical benchmark thresholds:
    - 2008 GFC: >= 85.0
    - 2020 COVID: >= 85.0
    - August 2024: >= 65.0

    Args:
        episodes: List of episode dicts with 'name', 'date', 'stress'.

    Returns:
        Dict with 'all_passed' (bool) and 'validated' (list of episode results).
    """
    validated = []
    all_passed = True

    thresholds = {
        "GFC": 85.0,
        "COVID": 85.0,
        "August 2024": 65.0,
    }

    for ep in episodes:
        name = ep.get("name", "")
        stress = ep.get("stress", 0.0)
        target_thresh = 60.0
        for k, thresh in thresholds.items():
            if k.lower() in name.lower():
                target_thresh = thresh
                break

        passed = stress >= target_thresh
        if not passed:
            all_passed = False

        validated.append({
            "name": name,
            "date": ep.get("date", ""),
            "stress": stress,
            "threshold": target_thresh,
            "passed": passed,
        })

    return {"all_passed": all_passed, "validated": validated}
