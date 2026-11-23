# pipeline/src/radar/metrics/decomposition.py
"""Move Decomposer metric engine (Section 3 of docs/MASTER_PLAN.md).

Decomposes target country or sector returns into global factor contributions,
an exact FX identity (for USD lens), and an explicit unexplained residual with
honest HAC uncertainty intervals.
"""

from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Any, Literal
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge

LOGGER = logging.getLogger(__name__)


def compute_utc_close_time(
    date: datetime.date,
    iana_tz: str,
    close_time_str: str,
) -> datetime:
    """Compute exact UTC timestamp for a market close on a specific calendar date.

    Handles Daylight Saving Time shifts dynamically via zoneinfo.
    """
    hour, minute = map(int, close_time_str.split(":"))
    local_dt = datetime.combine(date, time(hour, minute), tzinfo=ZoneInfo(iana_tz))
    return local_dt.astimezone(ZoneInfo("UTC"))


def determine_calendar_lag(
    target_id: str,
    factor_id: str,
    calendars_cfg: dict[str, dict[str, str]],
    reference_date: datetime.date | None = None,
) -> int:
    """Determine whether factor returns must be lagged by 1 day relative to target.

    Rule: If the factor closes AFTER the target on date d, the factor move has not
    yet occurred when the target closes. The target can only react to the factor's
    close from date d-1 (lag = 1).
    """
    if reference_date is None:
        reference_date = datetime.now().date()

    target_cal = calendars_cfg.get(target_id)
    factor_cal = calendars_cfg.get(factor_id)

    if not target_cal or not factor_cal:
        return 0

    target_utc = compute_utc_close_time(
        reference_date, target_cal["tz"], target_cal["close_time"]
    )
    factor_utc = compute_utc_close_time(
        reference_date, factor_cal["tz"], factor_cal["close_time"]
    )

    # If factor closes after target, factor on day d is unavailable to target on day d
    if factor_utc > target_utc:
        return 1
    return 0


def align_factor_series(
    factor_series: pd.Series,
    lag: int,
) -> pd.Series:
    """Shift factor returns by calendar lag if non-synchronous."""
    if lag == 0:
        return factor_series
    return factor_series.shift(lag)


def compute_fx_identity(
    r_local: float,
    r_usd: float | None = None,
    r_fx: float | None = None,
) -> tuple[float, float, float]:
    """Compute exact FX identity for USD lens.

    Identity: (1 + r_usd) = (1 + r_local) * (1 + r_fx)

    Parameters
    ----------
    r_local : float
        Cumulative local currency return over horizon.
    r_usd : float, optional
        Cumulative USD return. If provided alongside r_local, r_fx is derived.
    r_fx : float, optional
        Cumulative FX return (local currency vs USD, positive = local strengthened).

    Returns
    -------
    tuple[float, float, float]
        (r_usd, r_local, r_fx) exactly satisfying the identity.
    """
    if r_usd is not None:
        # Derived r_fx from identity: (1 + r_usd) / (1 + r_local) - 1.0
        one_plus_local = 1.0 + r_local
        if abs(one_plus_local) < 1e-9:
            derived_fx = 0.0
        else:
            derived_fx = (1.0 + r_usd) / one_plus_local - 1.0
        return float(r_usd), float(r_local), float(derived_fx)

    if r_fx is not None:
        derived_usd = (1.0 + r_local) * (1.0 + r_fx) - 1.0
        return float(derived_usd), float(r_local), float(r_fx)

    return float(r_local), float(r_local), 0.0


def compute_vif(X: np.ndarray) -> np.ndarray:
    """Compute Variance Inflation Factor (VIF) for each column in design matrix X.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Array of VIF values for each feature.
    """
    n_features = X.shape[1]
    vifs = np.zeros(n_features)
    for i in range(n_features):
        y_i = X[:, i]
        X_i = np.delete(X, i, axis=1)
        if X_i.shape[1] == 0:
            vifs[i] = 1.0
            continue
        X_with_const = sm.add_constant(X_i)
        try:
            model = sm.OLS(y_i, X_with_const).fit()
            r2 = max(0.0, min(model.rsquared, 0.9999))
            vifs[i] = 1.0 / (1.0 - r2)
        except Exception:
            vifs[i] = 1.0
    return vifs


def fit_prior_window_betas(
    y_prior: pd.Series,
    X_prior: pd.DataFrame,
    vif_threshold: float = 5.0,
    ridge_alpha: float = 1.0,
    hac_lags: int = 5,
    estimator: Literal["rolling_ols", "kalman", "auto"] = "auto",
) -> dict[str, Any]:
    """Fit multi-factor regression on prior window with Newey-West HAC errors or Kalman filter.

    If any factor has VIF > vif_threshold, regularizes via Ridge regression.
    Evaluates out-of-sample R^2 between Kalman filter and rolling OLS when estimator='auto'.
    """
    from radar.metrics.kalman import KalmanBetaFilter, compare_kalman_vs_rolling_ols

    # Align and drop NaNs
    df = pd.concat([y_prior.rename("y"), X_prior], axis=1).dropna()
    n_obs = len(df)
    if n_obs < 30:
        raise ValueError(f"Insufficient prior observations: {n_obs} < 30")

    y = df["y"].values
    factor_names = list(X_prior.columns)
    X = df[factor_names].values

    # Check VIF
    vifs = compute_vif(X)
    max_vif = float(np.max(vifs)) if len(vifs) > 0 else 1.0
    is_regularized = max_vif > vif_threshold

    # Estimator selection
    eval_res = compare_kalman_vs_rolling_ols(df["y"], df[factor_names])
    chosen_estimator = eval_res["selected_estimator"] if estimator == "auto" else estimator

    if chosen_estimator == "kalman" and not is_regularized:
        try:
            # Add constant to factors
            X_const = np.column_stack([np.ones(len(X)), X])
            kf = KalmanBetaFilter(init_obs=min(60, n_obs // 2))
            kf_res = kf.fit_filter(y, X_const)
            alpha = float(kf_res["final_beta"][0])
            betas = {name: float(kf_res["final_beta"][i + 1]) for i, name in enumerate(factor_names)}
            intervals = {
                name: (float(kf_res["final_intervals"][i + 1, 0]), float(kf_res["final_intervals"][i + 1, 1]))
                for i, name in enumerate(factor_names)
            }
            # Compute pseudo in-sample R2
            y_pred = X_const @ kf_res["final_beta"]
            res = y - y_pred
            r2 = float(max(0.0, 1.0 - np.sum(res**2) / max(1e-12, np.sum((y - np.mean(y)) ** 2))))
            return {
                "alpha": alpha,
                "betas": betas,
                "intervals": intervals,
                "vifs": {name: float(v) for name, v in zip(factor_names, vifs)},
                "r2_in_sample": r2,
                "regularized": False,
                "n_obs": n_obs,
                "estimator": "kalman",
                "estimator_eval": eval_res,
            }
        except Exception as e:
            LOGGER.warning(f"Kalman fit failed ({e}), falling back to OLS HAC")
            chosen_estimator = "rolling_ols"

    if not is_regularized:
        # OLS with HAC (Newey-West) standard errors
        X_const = sm.add_constant(X)
        try:
            model = sm.OLS(y, X_const).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
            alpha = float(model.params[0])
            betas = {name: float(model.params[i + 1]) for i, name in enumerate(factor_names)}
            bse = model.bse[1:]  # standard errors of slope coefficients
            intervals = {}
            for i, name in enumerate(factor_names):
                se = float(bse[i]) if not np.isnan(bse[i]) else 0.0
                intervals[name] = (betas[name] - 1.96 * se, betas[name] + 1.96 * se)
            r2 = float(max(0.0, model.rsquared))
        except Exception as e:
            LOGGER.warning(f"OLS HAC failed ({e}), falling back to Ridge")
            is_regularized = True

    if is_regularized:
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
        ridge.fit(X, y)
        alpha = float(ridge.intercept_)
        betas = {name: float(coef) for name, coef in zip(factor_names, ridge.coef_)}
        # Approximate standard errors for ridge via residual variance
        y_pred = ridge.predict(X)
        res = y - y_pred
        s2 = np.sum(res**2) / max(1, n_obs - len(factor_names) - 1)
        r2 = float(max(0.0, 1.0 - np.sum(res**2) / max(1e-12, np.sum((y - np.mean(y)) ** 2))))
        intervals = {name: (betas[name] - 1.96 * np.sqrt(s2 / n_obs), betas[name] + 1.96 * np.sqrt(s2 / n_obs)) for name in factor_names}

    return {
        "alpha": alpha,
        "betas": betas,
        "intervals": intervals,
        "vifs": {name: float(v) for name, v in zip(factor_names, vifs)},
        "r2_in_sample": r2,
        "regularized": is_regularized,
        "n_obs": n_obs,
        "estimator": "rolling_ols",
        "estimator_eval": eval_res,
    }


def compute_decomposition(
    target_series: pd.Series,
    factors_df: pd.DataFrame,
    horizon_days: int = 5,
    lens: Literal["local", "usd"] = "usd",
    fx_series: pd.Series | None = None,
    prior_window_days: int = 250,
    unexplained_r2_floor: float = 0.10,
    target_name: str = "Target",
) -> dict[str, Any]:
    """Execute complete Move Decomposition for a single target and horizon.

    Parameters
    ----------
    target_series : pd.Series
        Daily target price or index levels (DateTime index).
    factors_df : pd.DataFrame
        Daily factor series (DateTime index).
    horizon_days : int
        Horizon {1, 5, 21, 63} trading days.
    lens : {'local', 'usd'}
        Return denomination lens.
    fx_series : pd.Series, optional
        Daily currency series (if lens == 'usd').
    prior_window_days : int
        Prior estimation window (default 250 days).
    unexplained_r2_floor : float
        Threshold below which model is tagged 'mostly_unexplained'.
    target_name : str
        Display name of the target entity.

    Returns
    -------
    dict[str, Any]
        Decomposition output conforming to M6 contract.
    """
    # 1. Compute target daily returns
    y_daily = np.log(target_series / target_series.shift(1)).dropna()

    # 2. Identify horizon period
    if len(y_daily) < prior_window_days + horizon_days:
        raise ValueError(
            f"Series length {len(y_daily)} insufficient for prior window {prior_window_days} + horizon {horizon_days}"
        )

    t_end = y_daily.index[-1]
    horizon_indices = y_daily.index[-horizon_days:]
    t_start = horizon_indices[0]

    # Prior window strictly ends BEFORE horizon starts (no lookahead!)
    prior_y = y_daily.loc[: horizon_indices[0]].iloc[:-1].tail(prior_window_days)
    prior_dates = prior_y.index

    # 3. Factor returns on prior window
    prior_X = factors_df.loc[prior_dates].dropna()
    prior_y = prior_y.loc[prior_X.index]

    # 4. Fit prior-window betas
    fit_result = fit_prior_window_betas(prior_y, prior_X)
    betas = fit_result["betas"]
    beta_intervals = fit_result["intervals"]

    # 5. Compute factor moves over horizon
    # Cumulative return of factors over horizon
    horizon_X = factors_df.loc[horizon_indices].dropna()
    cum_factor_moves = horizon_X.sum(axis=0)

    # 6. Target cumulative return over horizon
    cum_target_local = float(y_daily.loc[horizon_indices].sum())

    # 7. Apply FX Identity if USD lens
    fx_contribution = 0.0
    fx_move = 0.0
    if lens == "usd" and fx_series is not None:
        fx_daily = np.log(fx_series / fx_series.shift(1)).dropna()
        cum_fx = float(fx_daily.loc[horizon_indices].sum())
        cum_target_usd, cum_target_local, fx_contribution = compute_fx_identity(
            r_local=cum_target_local, r_fx=cum_fx
        )
        total_return = cum_target_usd
        fx_move = cum_fx
    else:
        total_return = cum_target_local

    # 8. Compute contributions and uncertainty intervals
    contributions = {}
    intervals = {}
    sum_contributions = 0.0

    for factor_name, beta in betas.items():
        if factor_name in cum_factor_moves:
            factor_move = float(cum_factor_moves[factor_name])
            contrib = float(beta * factor_move)
            contributions[factor_name] = contrib
            sum_contributions += contrib

            # Uncertainty interval: beta CI * factor move
            ci_low_beta, ci_high_beta = beta_intervals[factor_name]
            bound_1 = ci_low_beta * factor_move
            bound_2 = ci_high_beta * factor_move
            intervals[factor_name] = (float(min(bound_1, bound_2)), float(max(bound_1, bound_2)))
        else:
            contributions[factor_name] = 0.0
            intervals[factor_name] = (0.0, 0.0)

    if lens == "usd" and fx_series is not None:
        contributions["fx_effect"] = float(fx_contribution)
        intervals["fx_effect"] = (float(fx_contribution), float(fx_contribution))
        sum_contributions += fx_contribution

    # Residual = total - sum of contributions
    residual = float(total_return - sum_contributions)

    # 9. Out-of-sample R-squared evaluation (last 12 months)
    oos_eval_days = min(250, len(y_daily) - prior_window_days - horizon_days)
    if oos_eval_days >= 30:
        # Compare actual daily moves vs predicted from fixed betas
        eval_slice_y = y_daily.iloc[-(oos_eval_days + horizon_days) : -horizon_days]
        eval_slice_X = factors_df.loc[eval_slice_y.index]
        common = eval_slice_y.index.intersection(eval_slice_X.index)
        y_eval = eval_slice_y.loc[common].values
        pred_eval = np.zeros_like(y_eval)
        for fname, b in betas.items():
            if fname in eval_slice_X:
                pred_eval += b * eval_slice_X.loc[common, fname].values
        ss_res = np.sum((y_eval - pred_eval) ** 2)
        ss_tot = np.sum((y_eval - np.mean(y_eval)) ** 2)
        r2_out_sample = float(max(0.0, 1.0 - ss_res / max(1e-12, ss_tot)))
    else:
        r2_out_sample = fit_result["r2_in_sample"]

    is_mostly_unexplained = r2_out_sample < unexplained_r2_floor

    # 10. Generate associative, neutral narrative sentence
    narrative = generate_decomposition_narrative(
        target_name=target_name,
        horizon_days=horizon_days,
        lens=lens,
        total_return=total_return,
        contributions=contributions,
        residual=residual,
        r2_oos=r2_out_sample,
    )

    return {
        "target": target_name,
        "horizon_days": horizon_days,
        "lens": lens,
        "total_return": round(total_return, 4),
        "contributions": {k: round(v, 4) for k, v in contributions.items()},
        "intervals": {k: (round(v[0], 4), round(v[1], 4)) for k, v in intervals.items()},
        "residual": round(residual, 4),
        "betas": {k: round(v, 4) for k, v in betas.items()},
        "r2_in_sample": round(fit_result["r2_in_sample"], 4),
        "r2_out_sample": round(r2_out_sample, 4),
        "mostly_unexplained": is_mostly_unexplained,
        "regularized": fit_result["regularized"],
        "estimator": fit_result.get("estimator", "rolling_ols"),
        "estimator_eval": fit_result.get("estimator_eval"),
        "n_obs": fit_result["n_obs"],
        "window_dates": {
            "prior_start": str(prior_dates[0].date()),
            "prior_end": str(prior_dates[-1].date()),
            "horizon_start": str(t_start.date()),
            "horizon_end": str(t_end.date()),
        },
        "narrative": narrative,
    }


def generate_decomposition_narrative(
    target_name: str,
    horizon_days: int,
    lens: str,
    total_return: float,
    contributions: dict[str, float],
    residual: float,
    r2_oos: float,
) -> str:
    """Generate neutral, strictly associative narrative sentence.

    Template follows Section 3.5 of docs/MASTER_PLAN.md:
    "Over 5 days, India's USD return of -1.4% was associated with: rest-of-world
    equity -0.6%, US dollar -0.3%, oil -0.2%, rupee -0.4%, unexplained +0.1%
    (out-of-sample R-squared 0.41)."
    """
    horizon_label = f"{horizon_days} days" if horizon_days > 1 else "1 day"
    ret_pct = total_return * 100.0
    res_pct = residual * 100.0

    parts = []
    # Sort contributions by absolute magnitude
    sorted_contribs = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    for name, val in sorted_contribs:
        readable = name.replace("_", " ").title()
        val_pct = val * 100.0
        sign = "+" if val_pct >= 0 else ""
        parts.append(f"{readable.lower()} {sign}{val_pct:.1f}%")

    res_sign = "+" if res_pct >= 0 else ""
    parts.append(f"unexplained {res_sign}{res_pct:.1f}%")

    contrib_summary = ", ".join(parts)
    sign_ret = "+" if ret_pct >= 0 else ""

    sentence = (
        f"Over {horizon_label}, {target_name}'s {lens.upper()} return of {sign_ret}{ret_pct:.1f}% "
        f"was associated with: {contrib_summary} (out-of-sample R-squared {r2_oos:.2f})."
    )
    return sentence
