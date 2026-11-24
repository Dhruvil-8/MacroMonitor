# audit/oracles/oracle_risk_and_tails.py
"""Independent clean-room oracle for Risk Lab: VaR, ES, and PCA Sign Alignment."""

import math
import numpy as np

def oracle_historical_var_es(returns: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """Historical Value at Risk and Expected Shortfall at confidence alpha (e.g. 0.95, 0.99).
    
    Convention: Losses reported as positive decimals.
    q = quantile(returns, 1 - alpha)
    VaR = -q
    ES = -mean(returns[returns <= q])
    """
    valid = returns[~np.isnan(returns)]
    if len(valid) < 10:
        return 0.0, 0.0
    cutoff = np.percentile(valid, (1.0 - alpha) * 100.0)
    var = -float(cutoff)
    tail = valid[valid <= cutoff]
    es = -float(np.mean(tail)) if len(tail) > 0 else var
    return max(0.0, var), max(0.0, es)

def oracle_pca_sign_alignment(returns_matrix: np.ndarray, us_asset_index: int = 0) -> tuple[float, np.ndarray]:
    """Compute 1st Principal Component variance share and ensure US asset loading is positive.
    
    If loading of us_asset_index < 0, flip sign of entire eigenvector: v1 = -v1.
    """
    # Standardize columns
    X = returns_matrix - np.mean(returns_matrix, axis=0)
    std = np.std(X, axis=0, ddof=1)
    std[std == 0] = 1.0
    Z = X / std
    
    # SVD
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    variance_shares = (S ** 2) / np.sum(S ** 2)
    pc1_share = float(variance_shares[0])
    
    v1 = Vt[0, :].copy()
    if v1[us_asset_index] < 0:
        v1 = -v1
        
    return pc1_share, v1
