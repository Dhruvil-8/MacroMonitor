# pipeline/src/radar/metrics/drivers.py
"""Driver sensitivity layer analytics (Section 4.3 of docs/MASTER_PLAN.md).

Implements:
1. Rolling multi-factor regressions (120-day and 250-day windows) with Newey-West HAC standard errors.
2. Stability metric: share of historical rolling windows sharing the same sign as the current estimate.
3. Pre-registered economic priors matching (e.g. oil importers vs exporters, dollar sensitivity).
4. Sensitivity matrix with stability and significance indicators for heatmap rendering (C06).
5. Forest plot data structures with 95% confidence intervals (C08).
6. Rolling beta drift over time with confidence bands (C07).
7. Bivariate scatter with fitted OLS line, confidence band, and residuals (C09).
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

from radar.metrics.decomposition import compute_vif

LOGGER = logging.getLogger(__name__)

# Launch countries classification for economic priors
OIL_IMPORTERS = {"in", "jp", "de", "gb", "kr", "cn", "nifty50", "n225", "dax", "ftse", "kospi", "ssec"}
OIL_EXPORTERS = {"ca", "br", "tsse", "ibov"}
EMERGING_MARKETS = {"in", "br", "cn", "kr", "nifty50", "ibov", "ssec", "kospi"}


def get_economic_prior(factor_id: str, target_id: str, target_code: str) -> str:
    """Resolve pre-registered economic prior for a (factor, target) pair.

    Returns: 'positive' | 'negative' | 'neutral'.
    """
    code = target_code.lower()
    tid = target_id.lower()

    if factor_id == "row_equity":
        return "positive"  # Global equity market beta is universally positive
    elif factor_id == "copper":
        return "positive"  # Industrial cyclical proxy
    elif factor_id == "gold":
        return "neutral"   # Ambiguous / store of value
    elif factor_id == "brent":
        if code in OIL_EXPORTERS or tid in OIL_EXPORTERS:
            return "positive"
        if code in OIL_IMPORTERS or tid in OIL_IMPORTERS:
            return "negative"
        return "neutral"
    elif factor_id == "dxy":
        # Dollar strengthening tightens global financial conditions
        if code in EMERGING_MARKETS or tid in EMERGING_MARKETS:
            return "negative"
        return "negative"
    elif factor_id == "us10y":
        return "negative"  # Rising discount rates compress equity multiples
    return "neutral"


def evaluate_prior_alignment(beta: float, expected_prior: str) -> bool:
    """Evaluate whether an estimated coefficient aligns with its economic prior."""
    if expected_prior == "positive":
        return beta > 0
    elif expected_prior == "negative":
        return beta < 0
    return True  # 'neutral' is always considered non-contradictory


def compute_driver_stability(rolling_betas: pd.Series) -> float:
    """Compute share of rolling windows sharing the same sign as the latest coefficient.

    Formula:
        stability = sum(I(sign(beta_t) == sign(beta_latest))) / N_windows
    Returns float in [0.0, 1.0].
    """
    s = rolling_betas.dropna()
    if len(s) == 0:
        return 0.0
    latest_val = float(s.iloc[-1])
    if abs(latest_val) < 1e-8:
        return 1.0
    target_sign = np.sign(latest_val)
    same_sign_count = int(np.sum(np.sign(s) == target_sign))
    return float(same_sign_count / len(s))


def fit_window_multi_factor(
    y: pd.Series,
    X: pd.DataFrame,
    vif_threshold: float = 5.0,
    ridge_alpha: float = 1.0,
    hac_lags: int = 5,
) -> dict[str, Any]:
    """Fit multi-factor regression on a single window with Newey-West HAC errors."""
    aligned = pd.concat([y, X], axis=1).dropna()
    n_obs = len(aligned)
    if n_obs < 30:
        return {
            "betas": {col: 0.0 for col in X.columns},
            "intervals": {col: (0.0, 0.0) for col in X.columns},
            "p_values": {col: 1.0 for col in X.columns},
            "t_stats": {col: 0.0 for col in X.columns},
            "r2": 0.0,
            "regularized": False,
            "n_obs": n_obs,
        }

    y_clean = aligned.iloc[:, 0].to_numpy()
    X_clean = aligned.iloc[:, 1:].to_numpy()
    factor_cols = list(aligned.columns[1:])

    # VIF Check
    vifs = compute_vif(X_clean)
    max_vif = float(np.max(vifs)) if len(vifs) > 0 else 1.0
    regularized = max_vif > vif_threshold

    betas: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    p_values: dict[str, float] = {}
    t_stats: dict[str, float] = {}
    r2 = 0.0

    if not regularized:
        try:
            X_with_const = sm.add_constant(X_clean)
            lags = min(hac_lags, max(1, n_obs // 10))
            ols_model = sm.OLS(y_clean, X_with_const).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
            r2 = float(max(0.0, min(1.0, ols_model.rsquared)))

            for i, col in enumerate(factor_cols):
                param_idx = i + 1  # 0 is constant
                b = float(ols_model.params[param_idx])
                se = float(ols_model.bse[param_idx]) if not np.isnan(ols_model.bse[param_idx]) else 0.0
                pval = float(ols_model.pvalues[param_idx]) if not np.isnan(ols_model.pvalues[param_idx]) else 1.0
                tstat = float(ols_model.tvalues[param_idx]) if not np.isnan(ols_model.tvalues[param_idx]) else 0.0

                betas[col] = round(b, 4)
                intervals[col] = (round(b - 1.96 * se, 4), round(b + 1.96 * se, 4))
                p_values[col] = round(pval, 4)
                t_stats[col] = round(tstat, 2)
        except Exception as e:
            LOGGER.debug(f"OLS HAC fitting failed, falling back to Ridge: {e}")
            regularized = True

    if regularized:
        # L2 Ridge regression fallback
        k = X_clean.shape[1]
        XtX = X_clean.T @ X_clean
        I = np.eye(k)
        beta_ridge = np.linalg.inv(XtX + ridge_alpha * I) @ X_clean.T @ y_clean

        # Conservative standard errors
        residuals = y_clean - X_clean @ beta_ridge
        sigma2 = np.sum(residuals**2) / max(1, n_obs - k)
        cov_ridge = sigma2 * np.linalg.inv(XtX + ridge_alpha * I) @ XtX @ np.linalg.inv(XtX + ridge_alpha * I)
        se_ridge = np.sqrt(np.maximum(0.0, np.diag(cov_ridge)))

        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
        ss_res = np.sum(residuals**2)
        r2 = float(max(0.0, min(1.0, 1.0 - ss_res / ss_tot))) if ss_tot > 0 else 0.0

        for i, col in enumerate(factor_cols):
            b = float(beta_ridge[i])
            se = float(se_ridge[i])
            tstat = b / se if se > 0 else 0.0
            betas[col] = round(b, 4)
            intervals[col] = (round(b - 1.96 * se, 4), round(b + 1.96 * se, 4))
            p_values[col] = 0.05 if abs(tstat) >= 1.96 else 0.50
            t_stats[col] = round(tstat, 2)

    return {
        "betas": betas,
        "intervals": intervals,
        "p_values": p_values,
        "t_stats": t_stats,
        "r2": round(r2, 4),
        "regularized": regularized,
        "n_obs": n_obs,
    }


def compute_rolling_multi_factor_series(
    y: pd.Series,
    X: pd.DataFrame,
    window: int = 250,
    step: int = 5,
) -> dict[str, pd.DataFrame]:
    """Compute rolling multi-factor regression over time.

    Returns dict mapping factor_id -> DataFrame with columns ['beta', 'ci_lower', 'ci_upper'].
    """
    aligned = pd.concat([y, X], axis=1).dropna()
    n_total = len(aligned)
    if n_total < window:
        return {}

    factor_cols = list(X.columns)
    records: dict[str, list[dict[str, Any]]] = {f: [] for f in factor_cols}

    # Step through history
    for end_idx in range(window, n_total + 1, step):
        dt = aligned.index[end_idx - 1]
        w_df = aligned.iloc[end_idx - window : end_idx]
        y_w = w_df.iloc[:, 0]
        X_w = w_df.iloc[:, 1:]

        fit = fit_window_multi_factor(y_w, X_w)
        for f in factor_cols:
            b = fit["betas"].get(f, 0.0)
            ci_low, ci_high = fit["intervals"].get(f, (0.0, 0.0))
            records[f].append({
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt)[:10],
                "beta": b,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
            })

    return {f: pd.DataFrame(records[f]) for f in factor_cols}


def compute_bivariate_scatter_and_fit(
    y: pd.Series,
    x: pd.Series,
    window: int = 250,
) -> dict[str, Any]:
    """Compute 2D scatter points, OLS fitted line, confidence band, and residuals."""
    aligned = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna().tail(window)
    if len(aligned) < 20:
        return {
            "scatter": [],
            "line": [],
            "band_upper": [],
            "band_lower": [],
            "beta": 0.0,
            "r2": 0.0,
            "t_stat": 0.0,
            "n_obs": len(aligned),
        }

    x_vals = aligned["x"].to_numpy()
    y_vals = aligned["y"].to_numpy()

    # Fit univariate OLS with HAC
    X_const = sm.add_constant(x_vals)
    ols = sm.OLS(y_vals, X_const).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    beta = float(ols.params[1])
    intercept = float(ols.params[0])
    se_beta = float(ols.bse[1])
    r2 = float(max(0.0, min(1.0, ols.rsquared)))
    t_stat = float(ols.tvalues[1])

    # Grid for fitted line & 95% confidence band
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    grid_x = np.linspace(x_min, x_max, 25)
    pred_y = intercept + beta * grid_x
    margin = 1.96 * se_beta * np.abs(grid_x)

    scatter_pts = [
        [round(float(xv) * 100, 3), round(float(yv) * 100, 3)]
        for xv, yv in zip(x_vals, y_vals)
    ]
    line_pts = [
        [round(float(gx) * 100, 3), round(float(py) * 100, 3)]
        for gx, py in zip(grid_x, pred_y)
    ]
    band_upper = [
        [round(float(gx) * 100, 3), round(float(py + m) * 100, 3)]
        for gx, py, m in zip(grid_x, pred_y, margin)
    ]
    band_lower = [
        [round(float(gx) * 100, 3), round(float(py - m) * 100, 3)]
        for gx, py, m in zip(grid_x, pred_y, margin)
    ]

    return {
        "scatter": scatter_pts,
        "line": line_pts,
        "band_upper": band_upper,
        "band_lower": band_lower,
        "beta": round(beta, 4),
        "r2": round(r2, 4),
        "t_stat": round(t_stat, 2),
        "n_obs": len(aligned),
    }


def analyze_target_drivers(
    target_id: str,
    target_name: str,
    target_code: str,
    target_returns: pd.Series,
    factors_df: pd.DataFrame,
    window: int = 250,
) -> dict[str, Any]:
    """Execute complete driver sensitivity analysis for a target over a given window."""
    # 1. Fit latest window
    latest_y = target_returns.dropna().tail(window)
    latest_X = factors_df.reindex(latest_y.index).dropna()
    latest_y = latest_y.reindex(latest_X.index)

    fit = fit_window_multi_factor(latest_y, latest_X)

    # 2. Rolling betas for stability calculation
    rolling_dict = compute_rolling_multi_factor_series(target_returns, factors_df, window=window, step=5)

    driver_items: dict[str, dict[str, Any]] = {}
    for fid in factors_df.columns:
        b = fit["betas"].get(fid, 0.0)
        ci_low, ci_high = fit["intervals"].get(fid, (0.0, 0.0))
        pval = fit["p_values"].get(fid, 1.0)
        tstat = fit["t_stats"].get(fid, 0.0)

        # Stability
        r_df = rolling_dict.get(fid, pd.DataFrame())
        stability = compute_driver_stability(r_df["beta"]) if not r_df.empty else 1.0

        # Prior
        expected_prior = get_economic_prior(fid, target_id, target_code)
        prior_aligned = evaluate_prior_alignment(b, expected_prior)

        # Significance: 95% CI does not span zero
        significant = not (ci_low <= 0.0 <= ci_high)

        # History points for rolling charts (last 60 points)
        trail_pts = []
        if not r_df.empty:
            for _, r in r_df.tail(60).iterrows():
                trail_pts.append({
                    "date": str(r["date"]),
                    "beta": round(float(r["beta"]), 4),
                    "ci_lower": round(float(r["ci_lower"]), 4),
                    "ci_upper": round(float(r["ci_upper"]), 4),
                })

        driver_items[fid] = {
            "driver_id": fid,
            "beta": b,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "p_value": pval,
            "t_stat": tstat,
            "stability": round(stability, 2),
            "prior": expected_prior,
            "prior_aligned": prior_aligned,
            "significant": significant,
            "history": trail_pts,
        }

    return {
        "target_id": target_id,
        "target_name": target_name,
        "target_code": target_code,
        "window": window,
        "r2": fit["r2"],
        "regularized": fit["regularized"],
        "n_obs": fit["n_obs"],
        "drivers": driver_items,
    }
