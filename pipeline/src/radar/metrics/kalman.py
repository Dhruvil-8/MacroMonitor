"""Kalman Filter Time-Varying Beta Estimator for Move Decomposer and Drivers.

Specification P5:
- State-space model:
    Observation:  y_t = x_t' * beta_t + eps_t,        eps_t ~ N(0, sigma_eps^2)
    State:        beta_t = beta_{t-1} + eta_t,         eta_t ~ N(0, Q)
- Filtered estimates beta_{t|t} (point-in-time, strictly no future data).
- Posterior covariance P_{t|t} gives uncertainty intervals for time-varying betas.
- Walk-forward out-of-sample R^2 comparison against Rolling OLS for estimator selection.
"""

from __future__ import annotations

import math
from typing import Any, Literal
import numpy as np
import pandas as pd


class KalmanBetaFilter:
    """Multi-factor Kalman filter for time-varying regression coefficients."""

    def __init__(
        self,
        q_ratio: float = 1e-4,
        init_obs: int = 60,
    ) -> None:
        self.q_ratio = q_ratio
        self.init_obs = init_obs

    def fit_filter(
        self,
        y: np.ndarray,
        X: np.ndarray,
    ) -> dict[str, Any]:
        """Run forward Kalman filter on y and X (with constant term included in X if desired).

        Parameters
        ----------
        y : np.ndarray
            Shape (T,) target returns.
        X : np.ndarray
            Shape (T, K) factor returns.

        Returns
        -------
        dict with:
            betas: (T, K) filtered estimates beta_{t|t}
            se: (T, K) standard errors from sqrt(diag(P_{t|t}))
            pred_errors: (T,) out-of-sample one-step-ahead forecast errors (y_t - x_t' beta_{t|t-1})
            sigma_eps2: float estimated observation variance
            final_beta: (K,)
            final_intervals: (K, 2)
        """
        T, K = X.shape
        if T < self.init_obs:
            raise ValueError(f"Insufficient observations {T} < init_obs {self.init_obs}")

        # 1. Initialize prior from initial OLS
        X_init = X[: self.init_obs]
        y_init = y[: self.init_obs]

        # Use pseudo-inverse for numerical stability
        beta_init, residuals, rank, s = np.linalg.lstsq(X_init, y_init, rcond=None)
        res_init = y_init - X_init @ beta_init
        sigma_eps2 = float(np.sum(res_init**2) / max(1, self.init_obs - K))
        if sigma_eps2 <= 1e-12:
            sigma_eps2 = 1e-4

        # Initial state covariance P_0
        XtX_inv = np.linalg.pinv(X_init.T @ X_init)
        P = sigma_eps2 * XtX_inv
        beta = beta_init.copy()

        # State noise covariance Q
        Q = self.q_ratio * sigma_eps2 * np.eye(K)

        betas_filtered = np.zeros((T, K))
        se_filtered = np.zeros((T, K))
        pred_errors = np.zeros(T)

        I_k = np.eye(K)

        for t in range(T):
            x_t = X[t : t + 1].T  # (K, 1)
            y_t = float(y[t])

            # 1. State prediction
            # beta_{t|t-1} = beta_{t-1|t-1}
            P_pred = P + Q

            # 2. Observation prediction & error
            y_pred = float(np.squeeze(x_t.T @ beta))
            err = y_t - y_pred
            pred_errors[t] = err

            # Innovation variance F_t = x_t' P_{t|t-1} x_t + sigma_eps^2
            F_t = float(np.squeeze(x_t.T @ P_pred @ x_t)) + sigma_eps2
            if F_t <= 1e-12:
                F_t = 1e-12

            # 3. Kalman gain K_t = P_{t|t-1} x_t / F_t
            K_gain = (P_pred @ x_t) / F_t  # (K, 1)

            # 4. State update
            beta = beta + (K_gain * err).ravel()
            P = (I_k - K_gain @ x_t.T) @ P_pred

            # Symmetrize P
            P = 0.5 * (P + P.T)

            betas_filtered[t] = beta
            se_filtered[t] = np.sqrt(np.maximum(0.0, np.diag(P)))

        final_intervals = np.zeros((K, 2))
        for k in range(K):
            final_intervals[k, 0] = betas_filtered[-1, k] - 1.96 * se_filtered[-1, k]
            final_intervals[k, 1] = betas_filtered[-1, k] + 1.96 * se_filtered[-1, k]

        return {
            "betas": betas_filtered,
            "se": se_filtered,
            "pred_errors": pred_errors,
            "sigma_eps2": sigma_eps2,
            "final_beta": betas_filtered[-1],
            "final_intervals": final_intervals,
        }


def compare_kalman_vs_rolling_ols(
    y: pd.Series,
    X: pd.DataFrame,
    rolling_window: int = 120,
    eval_horizon: int = 250,
) -> dict[str, Any]:
    """Walk-forward comparison of Kalman filter vs Rolling OLS.

    Returns out-of-sample R^2 for both estimators over the eval_horizon.
    """
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < rolling_window + 50:
        return {
            "selected_estimator": "rolling_ols",
            "kalman_r2_oos": 0.0,
            "rolling_ols_r2_oos": 0.0,
            "rationale": "Insufficient history for walk-forward comparison",
        }

    y_vals = df["y"].values
    X_vals = df[X.columns].values
    T = len(y_vals)
    K = X_vals.shape[1]

    # 1. Fit Kalman Filter
    kf = KalmanBetaFilter(init_obs=min(60, len(y_vals) // 3))
    kf_res = kf.fit_filter(y_vals, X_vals)
    kf_errors = kf_res["pred_errors"]

    # 2. Fit Rolling OLS one-step-ahead errors
    ols_errors = np.zeros(T)
    start_eval = max(rolling_window, T - eval_horizon)

    for t in range(start_eval, T):
        y_win = y_vals[t - rolling_window : t]
        X_win = X_vals[t - rolling_window : t]
        b_win, _, _, _ = np.linalg.lstsq(X_win, y_win, rcond=None)
        y_pred = float(np.squeeze(X_vals[t] @ b_win))
        ols_errors[t] = y_vals[t] - y_pred

    # Evaluate over [start_eval : T]
    eval_y = y_vals[start_eval:]
    tot_var = np.sum((eval_y - np.mean(eval_y)) ** 2)
    if tot_var <= 1e-12:
        tot_var = 1e-6

    kalman_eval_err = kf_errors[start_eval:]
    ols_eval_err = ols_errors[start_eval:]

    kalman_r2 = max(0.0, 1.0 - float(np.sum(kalman_eval_err**2)) / tot_var)
    ols_r2 = max(0.0, 1.0 - float(np.sum(ols_eval_err**2)) / tot_var)

    selected = "kalman" if kalman_r2 >= ols_r2 else "rolling_ols"

    return {
        "selected_estimator": selected,
        "kalman_r2_oos": round(kalman_r2, 4),
        "rolling_ols_r2_oos": round(ols_r2, 4),
        "eval_observations": len(eval_y),
        "rationale": (
            f"Kalman filter achieved OOS R² of {kalman_r2:.1%} vs "
            f"Rolling OLS OOS R² of {ols_r2:.1%} over trailing {len(eval_y)} days."
        ),
    }
