# pipeline/src/radar/metrics/context.py
"""Historical context: distribution histograms, KDE, QQ plot, ECDF, seasonality, and declustered analogs.

Implements Section 4.5 and Section 5.1 (Charts C11, C12, C20, C25, C26) of docs/MASTER_PLAN.md.
Pure functions: vectorized, zero I/O.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def compute_distribution_histogram(s: pd.Series, num_bins: int = 25) -> dict[str, list[float]]:
    """Compute empirical distribution histogram bins and frequencies."""
    clean = s.dropna().to_numpy()
    if len(clean) == 0:
        return {"bins": [], "counts": []}

    counts, bin_edges = np.histogram(clean, bins=num_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        "bins": [round(float(b), 4) for b in bin_centers],
        "counts": [int(c) for c in counts],
    }


def compute_kde_and_normal_overlay(
    returns: pd.Series,
    grid_points: int = 60,
) -> dict[str, list[float]]:
    """Compute Gaussian Kernel Density Estimate (Silverman's rule) and normal overlay.

    Silverman's rule of thumb bandwidth:
        h = 0.9 * min(std, IQR / 1.34) * n^(-1/5)
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 20:
        return {"grid": [], "kde": [], "normal": []}

    mean_val = float(np.mean(r))
    std_val = float(np.std(r, ddof=1))
    if std_val <= 0:
        return {"grid": [], "kde": [], "normal": []}

    iqr_val = float(sp_stats.iqr(r))
    sigma_est = min(std_val, iqr_val / 1.34) if iqr_val > 0 else std_val
    h = 0.9 * sigma_est * (n ** (-0.2))

    # Grid covering 3.5 std on each side
    x_min = mean_val - 3.5 * std_val
    x_max = mean_val + 3.5 * std_val
    grid = np.linspace(x_min, x_max, grid_points)

    # Gaussian KDE evaluation
    # f_hat(x) = (1 / (n * h * sqrt(2*pi))) * sum(exp(-0.5 * ((x - r_i)/h)^2))
    diffs = grid[:, np.newaxis] - r[np.newaxis, :]
    kde_vals = np.mean(np.exp(-0.5 * (diffs / h) ** 2), axis=1) / (h * math.sqrt(2 * math.pi))

    # Normal curve with series' own empirical mean and std
    norm_vals = sp_stats.norm.pdf(grid, loc=mean_val, scale=std_val)

    return {
        "grid": [round(float(x), 5) for x in grid],
        "kde": [round(float(y), 5) for y in kde_vals],
        "normal": [round(float(y), 5) for y in norm_vals],
    }


def compute_qq_quantiles(
    returns: pd.Series,
    num_points: int = 50,
) -> dict[str, list[float]]:
    """Compute theoretical normal quantiles vs standardized empirical quantiles for QQ plot (C12)."""
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 20:
        return {"theoretical": [], "empirical": []}

    mean_val = float(np.mean(r))
    std_val = float(np.std(r, ddof=1))
    if std_val <= 0:
        return {"theoretical": [], "empirical": []}

    # Standardize returns
    z = (r - mean_val) / std_val

    # Quantile grid avoiding extreme asymptotes at 0 and 1
    quantiles = np.linspace(0.01, 0.99, num_points)
    theoretical = sp_stats.norm.ppf(quantiles)
    empirical = np.percentile(z, quantiles * 100.0)

    return {
        "theoretical": [round(float(q), 3) for q in theoretical],
        "empirical": [round(float(q), 3) for q in empirical],
    }


def compute_ecdf(
    returns: pd.Series,
    num_points: int = 100,
) -> dict[str, list[float]]:
    """Compute empirical cumulative distribution function (C20) downsampled to fixed points."""
    r = np.sort(returns.dropna().to_numpy())
    n = len(r)
    if n < 10:
        return {"x": [], "y": []}

    indices = np.linspace(0, n - 1, min(n, num_points), dtype=int)
    sampled_x = r[indices]
    sampled_y = (indices + 1) / float(n)

    return {
        "x": [round(float(x), 5) for x in sampled_x],
        "y": [round(float(y), 4) for y in sampled_y],
    }


def compute_calendar_heatmap(prices_or_returns: pd.Series, is_prices: bool = True) -> list[dict[str, Any]]:
    """Compute monthly cumulative returns grouped by year and month (C25).

    Flagged with mandatory statistical warning: 'may reflect chance; sample size too small for statistical significance'.
    """
    s = prices_or_returns.dropna()
    if len(s) < 30:
        return []

    df = pd.DataFrame({"val": s})
    df["year"] = df.index.year
    df["month"] = df.index.month

    records: list[dict[str, Any]] = []

    if is_prices:
        # For price series, compute monthly return as pct change from first to last in month
        grouped = df.groupby(["year", "month"])["val"]
        first_vals = grouped.first()
        last_vals = grouped.last()
        monthly_rets = (last_vals / first_vals - 1.0).reset_index()
        for _, row in monthly_rets.iterrows():
            val = float(row["val"])
            if not np.isnan(val) and not np.isinf(val):
                records.append({
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "return": round(val, 4),
                })
    else:
        # For difference series (rates, spreads), compute total monthly change
        grouped = df.groupby(["year", "month"])["val"]
        first_vals = grouped.first()
        last_vals = grouped.last()
        monthly_diffs = (last_vals - first_vals).reset_index()
        for _, row in monthly_diffs.iterrows():
            val = float(row["val"])
            if not np.isnan(val) and not np.isinf(val):
                records.append({
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "return": round(val, 4),
                })

    return records


def compute_declustered_analogs(
    s: pd.Series,
    target_idx: int = -1,
    top_k: int = 5,
    min_gap_days: int = 20,
) -> list[dict[str, Any]]:
    """Find top_k closest historical readings enforcing minimum date gap to decluster episodes."""
    clean = s.dropna()
    if len(clean) < top_k:
        return []

    target_val = float(clean.iloc[target_idx])
    target_date = clean.index[target_idx]

    # Exclude recent window around target date
    history = clean[clean.index < target_date - pd.Timedelta(days=min_gap_days)]
    if history.empty:
        return []

    abs_diff = (history - target_val).abs()
    sorted_history = abs_diff.sort_values()

    selected_analogs: list[dict[str, Any]] = []

    for dt, diff in sorted_history.items():
        is_too_close = False
        for a in selected_analogs:
            gap = abs((pd.to_datetime(dt) - pd.to_datetime(a["date"])).days)
            if gap < min_gap_days:
                is_too_close = True
                break

        if not is_too_close:
            val = float(history.loc[dt])
            # Find forward 5D and 20D outcomes
            idx_loc = clean.index.get_loc(dt)
            fwd_5d = None
            fwd_20d = None
            if idx_loc + 5 < len(clean):
                fwd_5d = round(float(clean.iloc[idx_loc + 5] / clean.iloc[idx_loc] - 1.0), 4)
            if idx_loc + 20 < len(clean):
                fwd_20d = round(float(clean.iloc[idx_loc + 20] / clean.iloc[idx_loc] - 1.0), 4)

            selected_analogs.append({
                "date": dt.strftime("%Y-%m-%d"),
                "value": round(val, 2),
                "distance": round(float(diff), 4),
                "fwd_5d": fwd_5d,
                "fwd_20d": fwd_20d,
            })
            if len(selected_analogs) >= top_k:
                break

    return selected_analogs
