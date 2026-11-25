# pipeline/src/radar/metrics/ratios.py
"""Institutional Cross-Asset Ratios & Fair-Value Valuation Models.

Implements daily cross-asset relative-value diagnostics:
1. 9 Cross-Asset Ratios (Gold/Silver, Copper/Gold, Cyclicals/Defensives, Breadth, etc.)
   with 1D/20D changes, expanding Z-scores, and historical percentiles.
2. Gold Fair-Value Model: expanding OLS of log(Gold) ~ US 10Y Real Yield + log(DXY),
   isolating the sovereign/geopolitical valuation residual.
3. Copper/Gold vs US 10Y Growth Divergence: standardized spread between industrial
   metals and Treasury yields.
4. Credit vs Equity Volatility Dislocation: HY OAS 250D Z-score minus VIX 250D Z-score.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

from radar.metrics.stats import compute_expanding_percentile, compute_expanding_zscore

LOGGER = logging.getLogger(__name__)


RATIO_SPECS = [
    {
        "id": "gold_silver",
        "name": "Gold / Silver Ratio",
        "num": "gold",
        "den": "silver",
        "op": "div",
        "category": "Metals & Safe Haven",
        "signal_high": "Monetary stress / silver lagging",
        "signal_low": "Industrial reflation / silver outperforming",
    },
    {
        "id": "copper_gold",
        "name": "Copper / Gold Ratio (x1000)",
        "num": "copper",
        "den": "gold",
        "op": "div_1000",
        "category": "Growth vs Safety",
        "signal_high": "Global industrial expansion",
        "signal_low": "Growth slowdown / safe-haven bid",
    },
    {
        "id": "xly_xlp",
        "name": "Cyclicals / Defensives (XLY / XLP)",
        "num": "xly",
        "den": "xlp",
        "op": "div",
        "category": "Equity Internals",
        "signal_high": "Risk-on consumer expansion",
        "signal_low": "Defensive rotation",
    },
    {
        "id": "rsp_spx",
        "name": "US Breadth (Equal-Wt RSP / SPX x100)",
        "num": "rsp",
        "den": "spx",
        "op": "div_100",
        "category": "Equity Internals",
        "signal_high": "Broad market participation",
        "signal_low": "Narrow mega-cap concentration",
    },
    {
        "id": "rut_ndx",
        "name": "Small Caps / Tech Duration (RUT / NDX)",
        "num": "rut",
        "den": "ndx",
        "op": "div",
        "category": "Equity Style",
        "signal_high": "Domestic cyclical / reflation leadership",
        "signal_low": "Long-duration mega-cap dominance",
    },
    {
        "id": "brent_wti",
        "name": "Brent - WTI Crude Spread ($/bbl)",
        "num": "brent",
        "den": "wti",
        "op": "sub",
        "category": "Energy & Geopolitics",
        "signal_high": "International / Middle East supply premium",
        "signal_low": "US domestic bottleneck / balanced seaborne",
    },
    {
        "id": "xle_wti",
        "name": "Energy Equities / Spot WTI (XLE / WTI)",
        "num": "xle",
        "den": "wti",
        "op": "div",
        "category": "Energy & Geopolitics",
        "signal_high": "Equities pricing stronger oil / margins",
        "signal_low": "Equities lagging spot crude",
    },
    {
        "id": "spx_eem",
        "name": "US vs Emerging Markets (SPX / EEM)",
        "num": "spx",
        "den": "eem",
        "op": "div",
        "category": "Global Allocation",
        "signal_high": "US exceptionalism / strong USD",
        "signal_low": "EM rotation / weak USD",
    },
    {
        "id": "niftybank_fmcg",
        "name": "India Cyclicals / Defensives (Bank / FMCG)",
        "num": "niftybank",
        "den": "niftyfmcg",
        "op": "div",
        "category": "India Internals",
        "signal_high": "Indian credit & capex leadership",
        "signal_low": "Defensive staples preference",
    },
]


def _build_ratio_series(s_num: pd.Series, s_den: pd.Series, op: str) -> pd.Series:
    aligned = pd.concat([s_num, s_den], axis=1, keys=["num", "den"]).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    if op == "sub":
        return (aligned["num"] - aligned["den"]).dropna()
    valid = aligned[aligned["den"] > 0]
    if valid.empty:
        return pd.Series(dtype=float)
    scale = 1000.0 if op == "div_1000" else 100.0 if op == "div_100" else 1.0
    return (valid["num"] / valid["den"] * scale).dropna()


def compute_ratios_and_valuations(all_points: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Compute cross-asset ratios, Gold fair-value residual, Cu/Au vs 10Y, and Credit-Vol spread."""
    ratios_out: list[dict[str, Any]] = []

    for spec in RATIO_SPECS:
        df_n = all_points.get(spec["num"])
        df_d = all_points.get(spec["den"])
        if df_n is None or df_n.empty or df_d is None or df_d.empty:
            continue

        series = _build_ratio_series(df_n["value"].dropna(), df_d["value"].dropna(), spec["op"])
        if len(series) < 60:
            continue

        z_s = compute_expanding_zscore(series, min_obs=60)
        p_s = compute_expanding_percentile(series, min_obs=60)

        curr = float(series.iloc[-1])
        prev_1d = float(series.iloc[-2]) if len(series) >= 2 else curr
        prev_20d = float(series.iloc[-21]) if len(series) >= 21 else curr

        if spec["op"] == "sub":
            chg_1d = curr - prev_1d
            chg_20d = curr - prev_20d
        else:
            chg_1d = ((curr / prev_1d) - 1.0) * 100.0 if prev_1d != 0 else 0.0
            chg_20d = ((curr / prev_20d) - 1.0) * 100.0 if prev_20d != 0 else 0.0

        z_val = float(z_s.iloc[-1]) if not np.isnan(z_s.iloc[-1]) else 0.0
        pct_val = float(p_s.iloc[-1]) if not np.isnan(p_s.iloc[-1]) else 50.0

        tail_252 = series.tail(252)
        ratios_out.append({
            "id": spec["id"],
            "name": spec["name"],
            "category": spec["category"],
            "op": spec["op"],
            "current": round(curr, 3),
            "chg_1d": round(chg_1d, 2),
            "chg_20d": round(chg_20d, 2),
            "z_score": round(z_val, 2),
            "percentile": round(pct_val, 1),
            "interpretation": spec["signal_high"] if z_val >= 0 else spec["signal_low"],
            "history_dates": [str(dt.date()) if hasattr(dt, "date") else str(dt)[:10] for dt in tail_252.index],
            "history_values": [round(float(x), 3) for x in tail_252.values],
        })

    # 2. Gold Fair-Value Model: log(Gold) ~ US10Y_Real + log(DXY)
    gold_val_model: dict[str, Any] = {}
    if all(k in all_points and not all_points[k].empty for k in ("gold", "us10y_real", "dxy")):
        g = all_points["gold"]["value"].dropna()
        ry = all_points["us10y_real"]["value"].dropna()
        dxy = all_points["dxy"]["value"].dropna()
        df_g = pd.concat([g, ry, dxy], axis=1, keys=["gold", "real_10y", "dxy"]).dropna()
        df_g = df_g[(df_g["gold"] > 0) & (df_g["dxy"] > 0)]
        if len(df_g) >= 250:
            # Fit on full history prior to last 252d (or full sample OLS) for structural fair value
            y = np.log(df_g["gold"].to_numpy())
            X = sm.add_constant(np.column_stack([df_g["real_10y"].to_numpy(), np.log(df_g["dxy"].to_numpy())]))
            ols = sm.OLS(y, X).fit()
            fitted = np.exp(ols.predict(X))
            df_g["fair_value"] = fitted
            df_g["gap_pct"] = ((df_g["gold"] / df_g["fair_value"]) - 1.0) * 100.0
            gap_std = float(df_g["gap_pct"].std()) or 1.0
            curr_gap = float(df_g["gap_pct"].iloc[-1])
            tail_g = df_g.tail(504)
            gold_val_model = {
                "r2": round(float(ols.rsquared), 3),
                "beta_real_10y": round(float(ols.params[1]), 4),
                "beta_log_dxy": round(float(ols.params[2]), 4),
                "current_actual": round(float(df_g["gold"].iloc[-1]), 2),
                "current_fair_value": round(float(df_g["fair_value"].iloc[-1]), 2),
                "valuation_gap_pct": round(curr_gap, 2),
                "valuation_z": round(curr_gap / gap_std, 2),
                "dates": [str(dt.date()) if hasattr(dt, "date") else str(dt)[:10] for dt in tail_g.index],
                "actual": [round(float(x), 2) for x in tail_g["gold"].values],
                "fair_value": [round(float(x), 2) for x in tail_g["fair_value"].values],
            }

    # 3. Copper/Gold vs US 10Y Yield Divergence
    cu_au_vs_10y: dict[str, Any] = {}
    if all(k in all_points and not all_points[k].empty for k in ("copper", "gold", "us10y")):
        cu_au = _build_ratio_series(
            all_points["copper"]["value"].dropna(),
            all_points["gold"]["value"].dropna(),
            "div_1000",
        )
        y10 = all_points["us10y"]["value"].dropna()
        df_c = pd.concat([cu_au, y10], axis=1, keys=["cu_au", "us10y"]).dropna().tail(504)
        if len(df_c) >= 60:
            z_cu = (df_c["cu_au"] - df_c["cu_au"].mean()) / (df_c["cu_au"].std() or 1.0)
            z_y = (df_c["us10y"] - df_c["us10y"].mean()) / (df_c["us10y"].std() or 1.0)
            spread_z = z_y - z_cu
            cu_au_vs_10y = {
                "current_cu_au": round(float(df_c["cu_au"].iloc[-1]), 3),
                "current_us10y": round(float(df_c["us10y"].iloc[-1]), 2),
                "divergence_z": round(float(spread_z.iloc[-1]), 2),
                "dates": [str(dt.date()) if hasattr(dt, "date") else str(dt)[:10] for dt in df_c.index],
                "cu_au_z": [round(float(x), 2) for x in z_cu.values],
                "us10y_z": [round(float(x), 2) for x in z_y.values],
            }

    # 4. Credit (HY OAS) vs Equity Vol (VIX) Dislocation
    credit_vs_vol: dict[str, Any] = {}
    if all(k in all_points and not all_points[k].empty for k in ("hy_spread", "vix")):
        hy = all_points["hy_spread"]["value"].dropna()
        vix = all_points["vix"]["value"].dropna()
        df_cv = pd.concat([hy, vix], axis=1, keys=["hy", "vix"]).dropna().tail(504)
        if len(df_cv) >= 60:
            z_hy = (df_cv["hy"] - df_cv["hy"].mean()) / (df_cv["hy"].std() or 1.0)
            z_vix = (df_cv["vix"] - df_cv["vix"].mean()) / (df_cv["vix"].std() or 1.0)
            disloc = z_hy - z_vix
            credit_vs_vol = {
                "current_hy": round(float(df_cv["hy"].iloc[-1]), 2),
                "current_vix": round(float(df_cv["vix"].iloc[-1]), 2),
                "dislocation_z": round(float(disloc.iloc[-1]), 2),
                "dates": [str(dt.date()) if hasattr(dt, "date") else str(dt)[:10] for dt in df_cv.index],
                "hy_z": [round(float(x), 2) for x in z_hy.values],
                "vix_z": [round(float(x), 2) for x in z_vix.values],
                "dislocation": [round(float(x), 2) for x in disloc.values],
            }

    return {
        "ratios": ratios_out,
        "gold_valuation": gold_val_model,
        "cu_au_vs_10y": cu_au_vs_10y,
        "credit_vs_vol": credit_vs_vol,
    }
