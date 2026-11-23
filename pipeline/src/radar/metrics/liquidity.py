"""Liquidity and Funding Metrics Module.

Section 8.8:
- Net liquidity = Fed total assets (WALCL) - Treasury General Account (WTREGEN) - Overnight Reverse Repo (RRPONTSYD)
- Units: WALCL (usd_mn -> usd_bn via / 1000), WTREGEN (usd_mn -> usd_bn via / 1000), RRPONTSYD (usd_bn)
- Level, 4-week (20 business days) and 13-week (65 business days) changes with expanding percentiles
- Overlay against S&P 500 with rolling 60-day correlation
- Stablecoin total supply and 30-day percentage change
"""

from typing import Any
import numpy as np
import pandas as pd

from radar.metrics.stats import compute_expanding_percentile


def compute_net_liquidity(
    fed_assets: pd.Series,
    tga: pd.Series,
    rrp: pd.Series,
    max_ffill_days: int = 7,
) -> pd.Series:
    """Compute US Net Liquidity in Billions of USD.

    Formula:
        Net Liquidity (USD B) = (WALCL / 1000) - (WTREGEN / 1000) - RRPONTSYD

    Args:
        fed_assets: WALCL from FRED in Millions USD.
        tga: WTREGEN from FRED in Millions USD.
        rrp: RRPONTSYD from FRED/NY Fed in Billions USD.
        max_ffill_days: Maximum forward-fill days to bridge weekly release gaps.

    Returns:
        Series of Net Liquidity in Billions of USD.
    """
    # 1. Normalize units to Billions of USD
    fed_bn = fed_assets / 1000.0
    tga_bn = tga / 1000.0
    rrp_bn = rrp.copy()

    # 2. Build aligned index across all three series
    combined_index = fed_bn.index.union(tga_bn.index).union(rrp_bn.index).sort_values()

    # Reindex and forward fill weekly points up to max_ffill_days
    fed_aligned = fed_bn.reindex(combined_index).ffill(limit=max_ffill_days)
    tga_aligned = tga_bn.reindex(combined_index).ffill(limit=max_ffill_days)
    rrp_aligned = rrp_bn.reindex(combined_index).ffill(limit=max_ffill_days)

    # 3. Compute Net Liquidity
    net_liq = fed_aligned - tga_aligned - rrp_aligned
    return net_liq.dropna()


def compute_liquidity_changes(
    net_liquidity: pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling changes (e.g. 4W = 20D, 13W = 65D) and expanding percentiles.

    Args:
        net_liquidity: Series of net liquidity in USD Billions.
        windows: List of day horizons (default [20, 65]).

    Returns:
        DataFrame with columns 'change_<w>' and 'pct_<w>'.
    """
    if windows is None:
        windows = [20, 65]

    df = pd.DataFrame(index=net_liquidity.index)
    for w in windows:
        chg = net_liquidity.diff(w)
        pct = compute_expanding_percentile(chg, min_obs=max(30, w))
        df[f"change_{w}"] = chg
        df[f"pct_{w}"] = pct

    return df


def compute_liquidity_equity_overlay(
    net_liquidity: pd.Series,
    spx: pd.Series,
    corr_window: int = 60,
) -> pd.DataFrame:
    """Align Net Liquidity and S&P 500, and compute rolling correlation.

    Args:
        net_liquidity: Series of net liquidity in USD Billions.
        spx: Series of S&P 500 price levels.
        corr_window: Window for rolling Pearson correlation.

    Returns:
        DataFrame with columns 'net_liquidity', 'spx', and f'rolling_corr_{corr_window}'.
    """
    common_idx = net_liquidity.index.intersection(spx.index).sort_values()
    df = pd.DataFrame(
        {
            "net_liquidity": net_liquidity.loc[common_idx],
            "spx": spx.loc[common_idx],
        },
        index=common_idx,
    )

    # Compute rolling correlation between Net Liquidity and SPX
    corr = df["net_liquidity"].rolling(window=corr_window, min_periods=corr_window // 2).corr(df["spx"])
    df[f"rolling_corr_{corr_window}"] = corr.clip(-1.0, 1.0)
    return df


def compute_stablecoin_metrics(
    stablecoin_supply: pd.Series,
    window: int = 30,
) -> dict[str, float]:
    """Compute stablecoin total circulating supply in USD Billions and window % change.

    Args:
        stablecoin_supply: Series of total stablecoin supply (in Billions USD).
        window: Change window in days (default 30).

    Returns:
        Dict with 'latest_supply' and f'change_{window}d_pct'.
    """
    clean_s = stablecoin_supply.dropna()
    if clean_s.empty:
        return {"latest_supply": 0.0, f"change_{window}d_pct": 0.0}

    latest = float(clean_s.iloc[-1])
    if len(clean_s) > window:
        past = float(clean_s.iloc[-window - 1])
        pct_chg = ((latest - past) / past) * 100.0 if past > 0 else 0.0
    else:
        pct_chg = 0.0

    return {
        "latest_supply": round(latest, 2),
        f"change_{window}d_pct": round(pct_chg, 2),
    }
