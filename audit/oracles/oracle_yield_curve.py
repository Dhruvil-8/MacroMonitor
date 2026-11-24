# audit/oracles/oracle_yield_curve.py
"""Independent clean-room oracle for Nelson-Siegel 3-Factor Yield Curve Fitting (P4)."""

import math
import numpy as np

def oracle_nelson_siegel(tau: float | np.ndarray, beta0: float, beta1: float, beta2: float, lambda_param: float = 0.0609) -> np.ndarray:
    """Diebold-Li / Nelson-Siegel formulation:
    
    y(tau) = beta0 + beta1 * ((1 - exp(-lambda*tau)) / (lambda*tau)) + beta2 * (((1 - exp(-lambda*tau))/(lambda*tau)) - exp(-lambda*tau))
    """
    t = np.asarray(tau, dtype=float)
    lt = lambda_param * t
    term1 = np.where(lt > 1e-6, (1.0 - np.exp(-lt)) / lt, 1.0)
    term2 = term1 - np.exp(-lt)
    return beta0 + beta1 * term1 + beta2 * term2

def oracle_fit_nelson_siegel(maturities: np.ndarray, yields: np.ndarray, lambda_param: float = 0.0609) -> tuple[float, float, float, float]:
    """Fit beta0 (Level), beta1 (Slope), beta2 (Curvature) via linear regression given fixed lambda."""
    t = np.asarray(maturities, dtype=float)
    y = np.asarray(yields, dtype=float)
    
    lt = lambda_param * t
    f1 = (1.0 - np.exp(-lt)) / lt
    f2 = f1 - np.exp(-lt)
    
    X = np.column_stack([np.ones_like(t), f1, f2])
    betas, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b0, b1, b2 = float(betas[0]), float(betas[1]), float(betas[2])
    
    y_pred = oracle_nelson_siegel(t, b0, b1, b2, lambda_param)
    rmse = float(math.sqrt(np.mean((y - y_pred) ** 2)))
    return b0, b1, b2, rmse
