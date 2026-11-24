# audit/oracles/oracle_decomposition_and_fx.py
"""Independent clean-room oracle for Move Decomposer and FX Identity."""

import math
import numpy as np

def oracle_fx_identity(
    r_local: float,
    r_usd: float | None = None,
    r_fx: float | None = None
) -> tuple[float, float, float]:
    """Identity: (1 + r_usd) = (1 + r_local) * (1 + r_fx)
    
    If r_usd is given: r_fx = (1 + r_usd) / (1 + r_local) - 1
    If r_fx is given: r_usd = (1 + r_local) * (1 + r_fx) - 1
    """
    if r_usd is not None:
        derived_fx = (1.0 + r_usd) / (1.0 + r_local) - 1.0
        return float(r_usd), float(r_local), float(derived_fx)
    if r_fx is not None:
        derived_usd = (1.0 + r_local) * (1.0 + r_fx) - 1.0
        return float(derived_usd), float(r_local), float(r_fx)
    return float(r_local), float(r_local), 0.0

def oracle_multi_factor_decomposition(
    target_returns: np.ndarray,
    factor_returns: np.ndarray,
    horizon_days: int = 5
) -> tuple[np.ndarray, float, float]:
    """Estimate factor betas on prior window, then compute contributions over horizon.
    
    Target return over horizon: R_T = sum_{t in horizon} r_target,t
    Factor return over horizon: F_k = sum_{t in horizon} r_factor_k,t
    Contribution_k = beta_k * F_k
    Residual = R_T - sum_k Contribution_k
    Guarantee: R_T == sum_k Contribution_k + Residual (exact to 1e-12)
    """
    n_obs, n_factors = factor_returns.shape
    est_target = target_returns[:-horizon_days]
    est_factors = factor_returns[:-horizon_days]
    
    # OLS estimation: beta = (X^T X)^{-1} X^T y
    X = np.column_stack([np.ones(len(est_factors)), est_factors])
    y = est_target
    
    beta_all, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    betas = beta_all[1:]
    
    # Cumulative horizon returns
    horiz_target_return = float(np.sum(target_returns[-horizon_days:]))
    horiz_factor_returns = np.sum(factor_returns[-horizon_days:], axis=0)
    
    contributions = betas * horiz_factor_returns
    residual = horiz_target_return - float(np.sum(contributions))
    
    # Verify identity
    assert abs((np.sum(contributions) + residual) - horiz_target_return) < 1e-12
    return contributions, residual, horiz_target_return
