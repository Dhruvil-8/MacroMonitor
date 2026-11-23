"""India Flows and Positioning Metrics Module.

Section 8.9:
- Delivery percentage = delivered quantity / traded quantity, aggregated for index constituents and per stock
- F&O: Open Interest (OI) buildup classification across four regimes:
    1. Long Buildup: Price UP, OI UP
    2. Short Buildup: Price DOWN, OI UP
    3. Short Covering: Price UP, OI DOWN
    4. Long Unwinding: Price DOWN, OI DOWN
- Put-Call Ratio (PCR) = Total Put OI / Total Call OI
- FII/DII Net Flows: Daily net flow (INR Crores), 5-day and 20-day rolling sums with expanding percentiles
- India VIX historical percentile ribbon
"""

from typing import Any
import numpy as np
import pandas as pd

from radar.metrics.stats import compute_expanding_percentile


def classify_oi_buildup(price_change_pct: float, oi_change_pct: float) -> str:
    """Classify derivative open interest buildup into four classical regimes.

    Args:
        price_change_pct: Percentage change in underlying price.
        oi_change_pct: Percentage change in open interest.

    Returns:
        One of 'Long Buildup', 'Short Buildup', 'Short Covering', 'Long Unwinding', or 'Neutral'.
    """
    if price_change_pct > 0 and oi_change_pct > 0:
        return "Long Buildup"
    elif price_change_pct < 0 and oi_change_pct > 0:
        return "Short Buildup"
    elif price_change_pct > 0 and oi_change_pct < 0:
        return "Short Covering"
    elif price_change_pct < 0 and oi_change_pct < 0:
        return "Long Unwinding"
    return "Neutral"


def compute_fii_dii_rolling_flows(
    fii_daily: pd.Series,
    dii_daily: pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling cumulative sums and expanding percentiles for FII and DII cash flows.

    Args:
        fii_daily: Daily FII net purchases in INR Crores.
        dii_daily: Daily DII net purchases in INR Crores.
        windows: Rolling sum horizons in days (default [5, 20]).

    Returns:
        DataFrame with rolling sums and percentiles.
    """
    if windows is None:
        windows = [5, 20]

    common_idx = fii_daily.index.intersection(dii_daily.index).sort_values()
    df = pd.DataFrame(
        {
            "fii_daily": fii_daily.loc[common_idx],
            "dii_daily": dii_daily.loc[common_idx],
        },
        index=common_idx,
    )

    for w in windows:
        df[f"fii_{w}d"] = df["fii_daily"].rolling(window=w, min_periods=1).sum()
        df[f"dii_{w}d"] = df["dii_daily"].rolling(window=w, min_periods=1).sum()

    # Expanding percentiles on 20D cumulative flow
    if "fii_20d" in df.columns:
        df["fii_20d_pct"] = compute_expanding_percentile(df["fii_20d"], min_obs=max(20, min(len(df) // 2, 60)))
    if "dii_20d" in df.columns:
        df["dii_20d_pct"] = compute_expanding_percentile(df["dii_20d"], min_obs=max(20, min(len(df) // 2, 60)))

    return df


def compute_put_call_ratio(put_oi: float, call_oi: float) -> float:
    """Compute Put-Call Ratio (PCR) from total option open interest.

    Args:
        put_oi: Total put option open interest.
        call_oi: Total call option open interest.

    Returns:
        PCR ratio rounded to 2 decimal places.
    """
    if call_oi <= 0:
        return 0.0
    return round(float(put_oi / call_oi), 2)


def compute_delivery_metrics(
    delivered_qty: pd.Series,
    traded_qty: pd.Series,
) -> dict[str, float]:
    """Compute volume-weighted delivery percentage for a basket or single stock.

    Args:
        delivered_qty: Delivered share quantity.
        traded_qty: Total traded share quantity.

    Returns:
        Dict with 'delivery_pct'.
    """
    tot_del = delivered_qty.sum()
    tot_traded = traded_qty.sum()
    if tot_traded <= 0:
        return {"delivery_pct": 0.0}

    pct = (tot_del / tot_traded) * 100.0
    return {"delivery_pct": round(float(pct), 2)}
