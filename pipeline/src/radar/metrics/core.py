"""Core statistical primitives for MacroMonitor.

Implements strictly backward-looking / expanding-window calculations:
- Log returns (for price series) and first differences (for rates/yields/vol indices)
- Annualized realized volatility over window w (sample std * sqrt(252))
- Rolling drawdown P_t / max(P_{t-w+1..t}) - 1
- Expanding percentile rank against strictly prior observations [1..t-1]
- Expanding Z-score using Welford's online mean & variance over [1..t-1]
- Lag-1 autocorrelation rho_1 and effective sample size N_eff = N * (1 - rho_1) / (1 + rho_1)
- OLS 60-day trend t-statistic of log-price (or raw rate) on time
"""

from __future__ import annotations

import bisect
import math


def compute_returns(values: list[float | None], is_rate: bool = False) -> list[float | None]:
    """Compute daily log returns for prices or first differences for rate/yield series."""
    n = len(values)
    out: list[float | None] = [None] * n
    for i in range(1, n):
        cur = values[i]
        prev = values[i - 1]
        if cur is None or prev is None:
            continue
        if is_rate:
            out[i] = cur - prev
        else:
            if cur > 0.0 and prev > 0.0:
                out[i] = math.log(cur / prev)
    return out


def compute_period_return(values: list[float | None], window: int, is_rate: bool = False) -> list[float | None]:
    """Compute trailing `window`-day return (simple pct or rate change)."""
    n = len(values)
    out: list[float | None] = [None] * n
    for i in range(window, n):
        cur = values[i]
        prev = values[i - window]
        if cur is None or prev is None:
            continue
        if is_rate:
            out[i] = cur - prev
        else:
            if prev > 0.0:
                out[i] = cur / prev - 1.0
    return out


def compute_realized_vol_series(
    returns: list[float | None],
    window: int,
    annualize_days: int = 252,
) -> list[float | None]:
    """Compute rolling annualized sample standard deviation over `window` observations."""
    n = len(returns)
    out: list[float | None] = [None] * n
    if window < 2:
        return out
    sqrt_ann = math.sqrt(annualize_days)
    for i in range(window - 1, n):
        slice_vals = [x for x in returns[i - window + 1 : i + 1] if x is not None]
        if len(slice_vals) < max(2, math.ceil(window * 0.9)):
            continue
        mean_v = sum(slice_vals) / len(slice_vals)
        var_v = sum((x - mean_v) ** 2 for x in slice_vals) / (len(slice_vals) - 1)
        out[i] = math.sqrt(max(0.0, var_v)) * sqrt_ann
    return out


def compute_drawdown(values: list[float | None], window: int = 252) -> list[float | None]:
    """Compute trailing `window`-day rolling drawdown P_t / max(P_{t-w+1..t}) - 1."""
    n = len(values)
    out: list[float | None] = [None] * n
    for i in range(n):
        cur = values[i]
        if cur is None or cur <= 0.0:
            continue
        start = max(0, i - window + 1)
        win_vals = [x for x in values[start : i + 1] if x is not None and x > 0.0]
        if not win_vals:
            continue
        peak = max(win_vals)
        out[i] = (cur / peak) - 1.0
    return out


def compute_expanding_percentile_series(values: list[float | None], min_obs: int = 750) -> list[float | None]:
    """Compute expanding percentile [0, 100] comparing x_t strictly against prior x_{1..t-1}."""
    n = len(values)
    out: list[float | None] = [None] * n
    sorted_hist: list[float] = []
    for val_idx in range(n):
        val = values[val_idx]
        if val is None or not math.isfinite(val):
            continue
        if len(sorted_hist) >= min_obs:
            left = bisect.bisect_left(sorted_hist, val)
            right = bisect.bisect_right(sorted_hist, val)
            equal_cnt = right - left
            out[val_idx] = 100.0 * (left + 0.5 * equal_cnt) / len(sorted_hist)
        bisect.insort(sorted_hist, val)
    return out


def compute_expanding_zscore_series(values: list[float | None], min_obs: int = 750) -> list[float | None]:
    """Compute expanding Z-score comparing x_t strictly against prior mean & std over [1..t-1]."""
    n = len(values)
    out: list[float | None] = [None] * n
    count = 0
    mean = 0.0
    m2 = 0.0
    for i, val in enumerate(values):
        if val is None or not math.isfinite(val):
            continue
        if count >= min_obs and count >= 2:
            var = m2 / (count - 1)
            std = math.sqrt(max(0.0, var))
            out[i] = 0.0 if std <= 1e-12 else (val - mean) / std
        count += 1
        delta = val - mean
        mean += delta / count
        delta2 = val - mean
        m2 += delta * delta2
    return out


def compute_lag1_autocorr(values: list[float | None]) -> float:
    """Compute sample lag-1 autocorrelation rho_1 across consecutive non-null pairs."""
    pairs = [
        (values[i - 1], values[i])
        for i in range(1, len(values))
        if values[i - 1] is not None and values[i] is not None
    ]
    if len(pairs) < 5:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs)
    den_y = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(den_x * den_y)
    if den <= 1e-12:
        return 0.0
    return max(-0.99, min(0.99, num / den))


def compute_effective_n(n_obs: int, rho1: float) -> int:
    """Compute effective sample size N_eff = N * (1 - rho1) / (1 + rho1)."""
    if n_obs <= 1:
        return max(0, n_obs)
    if rho1 <= 0.0:
        return n_obs
    eff = n_obs * (1.0 - rho1) / (1.0 + rho1)
    return max(1, int(round(eff)))


def compute_trend_tstat_series(
    values: list[float | None],
    window: int = 60,
    is_rate: bool = False,
) -> list[float | None]:
    """Compute rolling OLS slope t-statistic of ln(P_t) (or rate) on time over `window` days."""
    n = len(values)
    out: list[float | None] = [None] * n
    if window < 5:
        return out
    x_bar = (window - 1) / 2.0
    sxx = sum((k - x_bar) ** 2 for k in range(window))
    for i in range(window - 1, n):
        raw_win = values[i - window + 1 : i + 1]
        if any(v is None for v in raw_win):
            continue
        if not is_rate:
            if any(v <= 0.0 for v in raw_win if v is not None):
                continue
            ys = [math.log(v) for v in raw_win if v is not None]
        else:
            ys = [float(v) for v in raw_win if v is not None]
        y_bar = sum(ys) / window
        sxy = sum((k - x_bar) * (y - y_bar) for k, y in enumerate(ys))
        beta = sxy / sxx
        rss = sum((y - (y_bar + beta * (k - x_bar))) ** 2 for k, y in enumerate(ys))
        se_sq = (rss / (window - 2)) / sxx
        if se_sq <= 1e-18:
            if abs(beta) <= 1e-12:
                out[i] = 0.0
            else:
                out[i] = 25.0 if beta > 0 else -25.0
        else:
            t_val = beta / math.sqrt(se_sq)
            out[i] = max(-25.0, min(25.0, t_val))
    return out
