# pipeline/src/radar/metrics/cycles.py
"""4-Quadrant Growth-Inflation Economic Machine & Macro Cycle Clock.

Implements:
1. Market-Implied Growth vs Inflation 4-Quadrant Regime Clock:
   - Growth Impulse: equally-weighted standardized 60D changes in Copper/Gold,
     Cyclicals/Defensives (XLY/XLP), 10Y-2Y Curve Slope, and inverted HY Spread.
   - Inflation Impulse: equally-weighted standardized 60D changes in 10Y Breakeven
     Inflation (T10YIE), Brent Crude, and Gold.
   - Maps monthly/weekly trail across 4 Quadrants:
       Recovery    (Growth > 0, Inflation <= 0)
       Overheat    (Growth > 0, Inflation > 0)
       Stagflation (Growth <= 0, Inflation > 0)
       Reflation   (Growth <= 0, Inflation <= 0)
2. Empirical Quadrant Asset Returns: actual annualized return of SPX, Nifty 50,
   Gold, Brent, and 10Y Yield changes inside each quadrant across history.
3. Macro Cycle Clock: Curve Slope (10Y-2Y), Credit Spread (HY OAS), Real 10Y Yield
   (DFII10), and Short Funding Rate (SOFR).
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def classify_quadrant(growth_z: float, inflation_z: float) -> str:
    if growth_z >= 0 and inflation_z < 0:
        return "Recovery"
    if growth_z >= 0 and inflation_z >= 0:
        return "Overheat"
    if growth_z < 0 and inflation_z >= 0:
        return "Stagflation"
    return "Reflation"


def _zscore_60d_impulse(s: pd.Series, log_ret: bool = True) -> pd.Series:
    s_clean = s.dropna()
    if len(s_clean) < 120:
        return pd.Series(dtype=float)
    if log_ret:
        s_clean = s_clean[s_clean > 0]
        chg = np.log(s_clean / s_clean.shift(60)).dropna()
    else:
        chg = s_clean.diff(60).dropna()
    exp_mean = chg.expanding(min_periods=60).mean().shift(1)
    exp_std = chg.expanding(min_periods=60).std().shift(1).replace(0.0, np.nan)
    return ((chg - exp_mean) / exp_std).dropna().clip(-4.0, 4.0)


def compute_macro_cycles(all_points: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Compute 4-Quadrant Growth/Inflation cycle and empirical quadrant returns."""
    # Growth components
    g_parts: dict[str, pd.Series] = {}
    if "copper" in all_points and "gold" in all_points:
        cu = all_points["copper"]["value"].dropna()
        au = all_points["gold"]["value"].dropna()
        df_cg = pd.concat([cu, au], axis=1, keys=["cu", "au"]).dropna()
        df_cg = df_cg[df_cg["au"] > 0]
        if not df_cg.empty:
            g_parts["cu_au"] = _zscore_60d_impulse(df_cg["cu"] / df_cg["au"], log_ret=True)
    if "xly" in all_points and "xlp" in all_points:
        xly = all_points["xly"]["value"].dropna()
        xlp = all_points["xlp"]["value"].dropna()
        df_cd = pd.concat([xly, xlp], axis=1, keys=["xly", "xlp"]).dropna()
        df_cd = df_cd[df_cd["xlp"] > 0]
        if not df_cd.empty:
            g_parts["xly_xlp"] = _zscore_60d_impulse(df_cd["xly"] / df_cd["xlp"], log_ret=True)
    if "us2s10s" in all_points and not all_points["us2s10s"].empty:
        g_parts["curve"] = _zscore_60d_impulse(all_points["us2s10s"]["value"], log_ret=False)
    if "hy_spread" in all_points and not all_points["hy_spread"].empty:
        # Tightening credit spread = positive growth impulse
        g_parts["hy_inv"] = -1.0 * _zscore_60d_impulse(all_points["hy_spread"]["value"], log_ret=False)

    # Inflation components
    i_parts: dict[str, pd.Series] = {}
    if "us10y_breakeven" in all_points and not all_points["us10y_breakeven"].empty:
        i_parts["breakeven"] = _zscore_60d_impulse(all_points["us10y_breakeven"]["value"], log_ret=False)
    if "brent" in all_points and not all_points["brent"].empty:
        i_parts["brent"] = _zscore_60d_impulse(all_points["brent"]["value"], log_ret=True)
    if "gold" in all_points and not all_points["gold"].empty:
        i_parts["gold"] = _zscore_60d_impulse(all_points["gold"]["value"], log_ret=True)

    g_df = pd.DataFrame(g_parts).dropna(how="all")
    i_df = pd.DataFrame(i_parts).dropna(how="all")

    growth_impulse = g_df.mean(axis=1).dropna()
    inflation_impulse = i_df.mean(axis=1).dropna()

    regime_df = pd.concat([growth_impulse, inflation_impulse], axis=1, keys=["growth", "inflation"]).dropna()
    if regime_df.empty:
        return {
            "current_quadrant": "Unknown",
            "current_growth_z": 0.0,
            "current_inflation_z": 0.0,
            "trail": [],
            "quadrant_returns": [],
            "gauges": [],
        }

    regime_df["quadrant"] = [
        classify_quadrant(float(r.growth), float(r.inflation)) for r in regime_df.itertuples()
    ]

    # Monthly trail over last 24 months
    monthly = regime_df.resample("ME").last().dropna().tail(24)
    trail = [
        {
            "date": str(dt.date())[:7],
            "growth_z": round(float(row.growth), 2),
            "inflation_z": round(float(row.inflation), 2),
            "quadrant": str(row.quadrant),
        }
        for dt, row in monthly.iterrows()
    ]

    # Empirical annualized return inside each quadrant
    asset_keys = [
        ("spx", "S&P 500 (US)", False),
        ("nifty50", "Nifty 50 (India)", False),
        ("gold", "Gold", False),
        ("brent", "Brent Crude", False),
        ("us10y", "US 10Y Yield (bps/yr)", True),
    ]
    ret_cols: dict[str, pd.Series] = {}
    for aid, _, is_diff in asset_keys:
        df_a = all_points.get(aid)
        if df_a is not None and not df_a.empty:
            s_a = df_a["value"].dropna()
            if is_diff:
                ret_cols[aid] = s_a.diff() * 100.0
            else:
                s_a = s_a[s_a > 0]
                ret_cols[aid] = np.log(s_a / s_a.shift(1))

    merged = pd.concat([regime_df[["quadrant"]], pd.DataFrame(ret_cols)], axis=1).dropna(subset=["quadrant"])
    quadrant_returns: list[dict[str, Any]] = []
    for q in ["Recovery", "Overheat", "Stagflation", "Reflation"]:
        sub = merged[merged["quadrant"] == q]
        n_days = len(sub)
        stats: dict[str, float | None] = {}
        for aid, _, is_diff in asset_keys:
            if aid in sub.columns and sub[aid].notna().sum() > 20:
                ann = float(sub[aid].mean()) * 252.0
                stats[aid] = round(ann if is_diff else ann * 100.0, 1)
            else:
                stats[aid] = None
        quadrant_returns.append({
            "quadrant": q,
            "days": n_days,
            "share_pct": round((n_days / len(merged)) * 100.0, 1) if len(merged) > 0 else 0.0,
            "returns": stats,
        })

    # Cycle & Debt Pressure Gauges
    gauges: list[dict[str, Any]] = []
    gauge_specs = [
        ("us2s10s", "10Y - 2Y Yield Curve", "pct", "Steepening (>0) signals expansion / easing; inverted (<0) signals tight policy"),
        ("hy_spread", "US High Yield OAS", "pct", "Credit stress indicator; >5.0% indicates acute tightening"),
        ("us10y_real", "US 10Y Real Yield (TIPS)", "pct", "Real cost of capital; >2.0% tightens financial conditions"),
        ("us10y_breakeven", "10Y Breakeven Inflation", "pct", "Market-implied long-term CPI expectations"),
    ]
    for gid, gname, unit, desc in gauge_specs:
        df_g = all_points.get(gid)
        if df_g is not None and not df_g.empty:
            s_g = df_g["value"].dropna()
            if len(s_g) > 10:
                curr = float(s_g.iloc[-1])
                pct = float((s_g <= curr).mean() * 100.0)
                gauges.append({
                    "id": gid,
                    "name": gname,
                    "value": round(curr, 2),
                    "unit": unit,
                    "percentile": round(pct, 1),
                    "note": desc,
                })

    curr_row = regime_df.iloc[-1]
    return {
        "current_quadrant": str(curr_row["quadrant"]),
        "current_growth_z": round(float(curr_row["growth"]), 2),
        "current_inflation_z": round(float(curr_row["inflation"]), 2),
        "trail": trail,
        "quadrant_returns": quadrant_returns,
        "gauges": gauges,
    }
