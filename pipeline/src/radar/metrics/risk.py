# pipeline/src/radar/metrics/risk.py
"""Risk Lab quantitative metrics (Section 4.4 of docs/MASTER_PLAN.md).

Implements:
1. Volatility Cone across windows {10, 20, 60, 120, 250} with historical percentiles.
2. EWMA volatility (lambda = 0.94) overlaid with realized volatility.
3. Distribution shape: rolling skewness, excess kurtosis, Jarque-Bera statistic, and QQ plot.
4. Tail risk: Historical vs. Gaussian VaR and Expected Shortfall at 95% and 99%, with Tail Ratio.
5. Autocorrelation Function (ACF) of returns and squared returns (lags 1 to 20) with N_eff confidence bounds.
6. Underwater drawdown series and the top 5 deepest drawdown episodes.
7. PCA factor structure with sign alignment (fixing US loading positive) and rolling PC1 synchronization share.
8. Downside correlation and Stress Correlation Premium (correlation during worst 10% benchmark days).
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from radar.metrics.stats import compute_effective_n, compute_lag1_autocorr

LOGGER = logging.getLogger(__name__)


def compute_volatility_cone(
    prices: pd.Series,
    windows: list[int] | None = None,
    annualization_factor: float = 252.0,
) -> list[dict[str, Any]]:
    """Compute realized volatility distribution across multiple lookback windows."""
    if windows is None:
        windows = [10, 20, 60, 120, 250]

    s = prices.dropna()
    if len(s) < 30:
        return []

    log_rets = np.log(s / s.shift(1)).dropna()
    cone_data: list[dict[str, Any]] = []

    for w in windows:
        if len(log_rets) < w + 10:
            continue
        rolling_vol = (log_rets.rolling(window=w).std(ddof=1) * np.sqrt(annualization_factor)).dropna()
        if rolling_vol.empty:
            continue

        arr = rolling_vol.to_numpy()
        current_val = float(arr[-1])
        pct_rank = float(np.sum(arr <= current_val) / len(arr) * 100.0)

        cone_data.append({
            "window": w,
            "min": round(float(np.min(arr)), 4),
            "p10": round(float(np.percentile(arr, 10)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "max": round(float(np.max(arr)), 4),
            "current": round(current_val, 4),
            "percentile": round(pct_rank, 1),
        })

    return cone_data


def compute_ewma_volatility(
    returns: pd.Series,
    decay: float = 0.94,
    annualization_factor: float = 252.0,
) -> pd.Series:
    """Compute Exponentially Weighted Moving Average (EWMA) volatility."""
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)

    sq_rets = r**2
    ewma_var = sq_rets.ewm(alpha=1.0 - decay, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var * annualization_factor)
    return ewma_vol


def compute_distribution_shape_and_qq(
    returns: pd.Series,
    num_qq_points: int = 50,
) -> dict[str, Any]:
    """Compute skewness, kurtosis, Jarque-Bera test, and QQ plot quantiles."""
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 30:
        return {
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "jb_stat": 0.0,
            "jb_pvalue": 1.0,
            "qq_theoretical": [],
            "qq_empirical": [],
            "mean": 0.0,
            "std": 0.0,
        }

    mean_val = float(np.mean(r))
    std_val = float(np.std(r, ddof=1))
    skew_val = float(sp_stats.skew(r))
    kurt_val = float(sp_stats.kurtosis(r))  # excess kurtosis (Fisher=True by default)

    jb_stat, jb_pval = sp_stats.jarque_bera(r)

    # Standardized returns for QQ plot
    z_scores = (r - mean_val) / std_val if std_val > 0 else r
    quantiles = np.linspace(0.01, 0.99, num_qq_points)
    theoretical_q = sp_stats.norm.ppf(quantiles)
    empirical_q = np.percentile(z_scores, quantiles * 100)

    return {
        "skewness": round(skew_val, 3),
        "excess_kurtosis": round(kurt_val, 3),
        "jb_stat": round(float(jb_stat), 2),
        "jb_pvalue": round(float(jb_pval), 4),
        "qq_theoretical": [round(float(q), 3) for q in theoretical_q],
        "qq_empirical": [round(float(q), 3) for q in empirical_q],
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
    }


def compute_tail_risk(
    returns: pd.Series,
    confidence_levels: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Compute Historical vs. Gaussian VaR and Expected Shortfall with Tail Ratio."""
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    r = returns.dropna().to_numpy()
    if len(r) < 30:
        return []

    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))

    tail_metrics: list[dict[str, Any]] = []

    for alpha in confidence_levels:
        tail_pct = (1.0 - alpha) * 100.0

        # Historical VaR (reported as positive loss percentage)
        hist_var = float(-np.percentile(r, tail_pct))
        losses_beyond_var = r[r <= -hist_var]
        hist_es = float(-np.mean(losses_beyond_var)) if len(losses_beyond_var) > 0 else hist_var

        # Gaussian parametric equivalent
        z_score = float(sp_stats.norm.ppf(1.0 - alpha))
        gauss_var = float(-(mean + z_score * std))
        pdf_z = float(sp_stats.norm.pdf(z_score))
        gauss_es = float(-(mean - std * (pdf_z / (1.0 - alpha))))

        tail_ratio = float(hist_es / gauss_es) if gauss_es > 0 else 1.0

        tail_metrics.append({
            "confidence": alpha,
            "hist_var": round(hist_var, 4),
            "hist_es": round(hist_es, 4),
            "gauss_var": round(gauss_var, 4),
            "gauss_es": round(gauss_es, 4),
            "tail_ratio": round(tail_ratio, 2),
        })

    return tail_metrics


def compute_autocorrelations(
    returns: pd.Series,
    max_lags: int = 20,
) -> dict[str, Any]:
    """Compute ACF of raw returns and squared returns with N_eff confidence bands."""
    r = returns.dropna()
    n_raw = len(r)
    if n_raw < max_lags + 10:
        return {"lags": [], "acf_returns": [], "acf_squared": [], "band": 0.0, "n_eff": float(n_raw)}

    rho1 = compute_lag1_autocorr(r)
    n_eff = compute_effective_n(n_raw, rho1)
    band = 1.96 / np.sqrt(n_eff) if n_eff > 0 else 0.0

    acf_rets: list[float] = []
    acf_sq: list[float] = []
    lags = list(range(1, max_lags + 1))

    sq_r = r**2

    for k in lags:
        c_r = float(r.autocorr(lag=k)) if not np.isnan(r.autocorr(lag=k)) else 0.0
        c_sq = float(sq_r.autocorr(lag=k)) if not np.isnan(sq_r.autocorr(lag=k)) else 0.0
        acf_rets.append(round(c_r, 4))
        acf_sq.append(round(c_sq, 4))

    return {
        "lags": lags,
        "acf_returns": acf_rets,
        "acf_squared": acf_sq,
        "band": round(float(band), 4),
        "n_eff": round(float(n_eff), 1),
    }


def compute_underwater_drawdowns(
    prices: pd.Series,
    top_k: int = 5,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Compute continuous underwater drawdown series and extract top-k deepest drawdown episodes."""
    s = prices.dropna()
    if len(s) < 10:
        return pd.Series(dtype=float), []

    peaks = s.cummax()
    dd_series = (s / peaks) - 1.0
    dd_series = dd_series.clip(upper=0.0)

    episodes: list[dict[str, Any]] = []
    in_dd = False
    peak_date = None
    peak_val = 0.0
    trough_date = None
    trough_val = 0.0

    dates = s.index

    for i in range(len(s)):
        val = float(s.iloc[i])
        dt = dates[i]
        cur_dd = float(dd_series.iloc[i])

        if not in_dd and cur_dd < 0:
            in_dd = True
            peak_date = dates[i - 1] if i > 0 else dt
            peak_val = float(s.iloc[i - 1]) if i > 0 else val
            trough_date = dt
            trough_val = val
        elif in_dd:
            if val < trough_val:
                trough_val = val
                trough_date = dt
            if cur_dd == 0:
                # Recovered
                depth = (trough_val / peak_val) - 1.0 if peak_val > 0 else 0.0
                episodes.append({
                    "peak_date": str(pd.to_datetime(peak_date).date()),
                    "trough_date": str(pd.to_datetime(trough_date).date()),
                    "recovery_date": str(pd.to_datetime(dt).date()),
                    "depth": round(float(depth), 4),
                    "duration_days": int((pd.to_datetime(trough_date) - pd.to_datetime(peak_date)).days),
                    "recovery_days": int((pd.to_datetime(dt) - pd.to_datetime(trough_date)).days),
                    "status": "Recovered",
                })
                in_dd = False

    # If still in drawdown at end of series
    if in_dd and peak_date is not None and trough_date is not None:
        depth = (trough_val / peak_val) - 1.0 if peak_val > 0 else 0.0
        episodes.append({
            "peak_date": str(pd.to_datetime(peak_date).date()),
            "trough_date": str(pd.to_datetime(trough_date).date()),
            "recovery_date": "Active",
            "depth": round(float(depth), 4),
            "duration_days": int((pd.to_datetime(trough_date) - pd.to_datetime(peak_date)).days),
            "recovery_days": None,
            "status": "Active",
        })

    # Sort by depth (ascending, i.e. deepest/most negative first)
    sorted_episodes = sorted(episodes, key=lambda e: e["depth"])[:top_k]

    return dd_series, sorted_episodes


def compute_pca_factor_structure(
    country_returns_df: pd.DataFrame,
    us_col: str = "spx",
    window: int = 250,
) -> dict[str, Any]:
    """Compute PCA on standardized USD returns with sign alignment to US loading."""
    clean_df = country_returns_df.dropna()
    if len(clean_df) < 50 or clean_df.shape[1] < 3:
        return {
            "scree": [],
            "loadings_pc1": {},
            "loadings_pc2": {},
            "pc1_history": [],
        }

    # Latest window for static factor structure
    latest_window = clean_df.tail(window)
    cols = list(latest_window.columns)

    # Standardize
    means = latest_window.mean()
    stds = latest_window.std(ddof=1)
    std_df = (latest_window - means) / stds.replace(0, 1)

    corr_matrix = std_df.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = float(np.sum(eigenvalues))
    scree = [round(float(ev / total_var * 100.0), 2) for ev in eigenvalues[:5]]

    v1 = eigenvectors[:, 0]
    v2 = eigenvectors[:, 1]

    # Sign alignment: ensure US loading is strictly positive
    if us_col in cols:
        us_idx = cols.index(us_col)
        if v1[us_idx] < 0:
            v1 = -v1
        if v2[us_idx] < 0:
            v2 = -v2

    loadings_pc1 = {cols[i]: round(float(v1[i]), 4) for i in range(len(cols))}
    loadings_pc2 = {cols[i]: round(float(v2[i]), 4) for i in range(len(cols))}

    # Rolling PC1 share over time ("Market Synchronization")
    pc1_history: list[dict[str, Any]] = []
    step = 5
    n_total = len(clean_df)
    for end_idx in range(window, n_total + 1, step):
        dt = clean_df.index[end_idx - 1]
        w_df = clean_df.iloc[end_idx - window : end_idx]
        w_corr = w_df.corr().values
        w_eigvals = np.linalg.eigvalsh(w_corr)
        w_pc1_share = float(np.max(w_eigvals) / np.sum(w_eigvals) * 100.0)
        pc1_history.append({
            "date": str(dt.date()) if hasattr(dt, "date") else str(dt)[:10],
            "pc1_share": round(w_pc1_share, 2),
        })

    return {
        "scree": scree,
        "loadings_pc1": loadings_pc1,
        "loadings_pc2": loadings_pc2,
        "pc1_history": pc1_history,
    }


def compute_downside_correlation(
    target_returns: pd.Series,
    benchmark_returns: pd.Series,
    stress_quantile: float = 0.10,
) -> dict[str, Any]:
    """Compute unconditional vs downside correlation and the stress correlation premium."""
    aligned = pd.concat([target_returns, benchmark_returns], axis=1, keys=["target", "bench"]).dropna()
    if len(aligned) < 30:
        return {
            "all_day_corr": 0.0,
            "downside_corr": 0.0,
            "stress_premium": 0.0,
            "stress_threshold": 0.0,
            "n_all": len(aligned),
            "n_stress": 0,
        }

    all_corr = float(aligned["target"].corr(aligned["bench"]))
    threshold = float(np.percentile(aligned["bench"], stress_quantile * 100.0))

    stress_df = aligned[aligned["bench"] <= threshold]
    if len(stress_df) >= 10:
        downside_corr = float(stress_df["target"].corr(stress_df["bench"]))
    else:
        downside_corr = all_corr

    premium = downside_corr - all_corr

    return {
        "all_day_corr": round(all_corr, 4),
        "downside_corr": round(downside_corr, 4),
        "stress_premium": round(premium, 4),
        "stress_threshold": round(threshold, 4),
        "n_all": len(aligned),
        "n_stress": len(stress_df),
    }
