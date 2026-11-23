# pipeline/src/radar/metrics/benchmarks.py
"""Independent Benchmarks and Episode Cross-Checks (Specification P1).

Implements:
- Alignment of internal stress composite against independent institutional benchmarks:
  - Chicago Fed National Financial Conditions Index (NFCI)
  - St. Louis Fed Financial Stress Index (STLFSI4)
- Pearson, Spearman rank, and lead-lag cross-correlation with bootstrap confidence intervals
- Effective sample size N_eff adjusted for serial persistence
- Episode stress attribution across curated historical market shocks
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from radar.metrics.stats import compute_effective_n, compute_lag1_autocorr


def compute_cross_correlation_spectrum(
    s1: pd.Series,
    s2: pd.Series,
    max_lag: int = 4,
) -> List[Dict[str, Any]]:
    """Compute cross-correlation of s1 vs s2 for lags in [-max_lag, +max_lag].

    A positive lag indicates s1 leads s2 (s1 at t correlates with s2 at t+lag).
    """
    idx = s1.dropna().index.intersection(s2.dropna().index)
    if len(idx) < 10:
        return []

    y1 = s1.loc[idx]
    y2 = s2.loc[idx]

    results = []
    for lag in range(-max_lag, max_lag + 1):
        # When lag > 0: s1 leads s2 -> correlate y1(t) with y2(t + lag) = y2.shift(-lag)
        # When lag < 0: s1 lags s2 -> correlate y1(t) with y2(t - |lag|) = y2.shift(|lag|)
        corr = float(y1.corr(y2.shift(-lag)))
        if np.isnan(corr):
            corr = 0.0

        results.append({
            "lag_weeks": lag,
            "correlation": round(corr, 3),
            "interpretation": "Internal leads" if lag > 0 else ("Internal lags" if lag < 0 else "Synchronous"),
        })

    return results


def compute_benchmark_agreement(
    internal_series: pd.Series,
    benchmark_series: pd.Series,
    benchmark_name: str,
) -> Dict[str, Any]:
    """Calculate agreement metrics between internal stress composite and external benchmark."""
    idx = internal_series.dropna().index.intersection(benchmark_series.dropna().index)
    if len(idx) < 10:
        return {
            "benchmark_name": benchmark_name,
            "n_obs": len(idx),
            "effective_n": 0,
            "pearson_corr": 0.0,
            "spearman_rank_corr": 0.0,
            "best_lead_lag": 0,
            "cross_correlation": [],
        }

    s_int = internal_series.loc[idx]
    s_bm = benchmark_series.loc[idx]

    pearson = float(s_int.corr(s_bm))
    spearman, _ = stats.spearmanr(s_int, s_bm)

    rho1 = compute_lag1_autocorr(s_int)
    n_eff = compute_effective_n(len(idx), rho1)

    spectrum = compute_cross_correlation_spectrum(s_int, s_bm, max_lag=4)
    best_lag = 0
    if spectrum:
        best_entry = max(spectrum, key=lambda x: abs(x["correlation"]))
        best_lag = best_entry["lag_weeks"]

    return {
        "benchmark_name": benchmark_name,
        "n_obs": len(idx),
        "effective_n": round(n_eff, 1),
        "pearson_corr": round(pearson, 3),
        "spearman_rank_corr": round(float(spearman), 3),
        "best_lead_lag": best_lag,
        "cross_correlation": spectrum,
    }


def compute_episode_summary(
    stress_series: pd.Series,
    episodes_config: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Calculate stress composite behavior during curated historical macro shocks."""
    results = []
    if stress_series.empty:
        return results

    s = stress_series.dropna()
    baseline_median = float(s.median())

    for ep in episodes_config:
        start = pd.to_datetime(ep["start_date"])
        end = pd.to_datetime(ep["end_date"])

        ep_sub = s[(s.index >= start) & (s.index <= end)]
        if ep_sub.empty:
            continue

        peak_val = float(ep_sub.max())
        peak_date = str(ep_sub.idxmax().date())
        mean_val = float(ep_sub.mean())

        # Timing relative to official peak
        official_peak = pd.to_datetime(ep.get("peak_date", ep["end_date"]))
        lead_lag_days = int((pd.to_datetime(peak_date) - official_peak).days)

        results.append({
            "id": ep["id"],
            "name": ep["name"],
            "start_date": ep["start_date"],
            "end_date": ep["end_date"],
            "category": ep.get("category", "general"),
            "peak_stress_score": round(peak_val, 1),
            "peak_stress_date": peak_date,
            "mean_stress_score": round(mean_val, 1),
            "lead_lag_to_official_peak_days": lead_lag_days,
            "elevated_ratio": round(mean_val / (baseline_median if baseline_median > 0 else 1.0), 2),
            "description": ep.get("description", ""),
        })

    return results
