# audit/oracles/oracle_stress_composite.py
"""Independent clean-room oracle for Financial Stress Composite (s11)."""

import numpy as np

def oracle_stress_composite(components: dict[str, float | None], min_components: int = 3) -> tuple[float | None, list[str]]:
    """Composite = mean of available components. Zero imputation.
    
    If valid components < min_components, returns None and list of missing.
    """
    valid = []
    missing = []
    for k, v in components.items():
        if v is not None and not np.isnan(v):
            valid.append(float(v))
        else:
            missing.append(k)
            
    if len(valid) < min_components:
        return None, missing
    
    score = float(sum(valid) / len(valid))
    return score, missing

def oracle_stress_regime(score: float | None) -> str:
    """Regime bands:
    - Score < 40.0: Low Stress / Calm
    - 40.0 <= Score < 70.0: Normal Market Conditions
    - 70.0 <= Score < 85.0: Elevated Stress
    - Score >= 85.0: High Macro Stress / Crisis
    """
    if score is None:
        return "Insufficient Data"
    if score < 40.0:
        return "Low Stress / Calm"
    elif score < 70.0:
        return "Normal Market Conditions"
    elif score < 85.0:
        return "Elevated Stress"
    else:
        return "High Macro Stress / Crisis"
