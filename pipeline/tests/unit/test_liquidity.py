import numpy as np
import pandas as pd
import pytest

from radar.metrics.liquidity import (
    compute_net_liquidity,
    compute_liquidity_changes,
    compute_liquidity_equity_overlay,
    compute_stablecoin_metrics,
)


def test_compute_net_liquidity_hand_checked_dates():
    """Verify net liquidity calculation against 3 hand-checked date fixtures."""
    # Date 1: 2024-01-03
    # WALCL = 7,686,674 M, WTREGEN = 759,045 M, RRP = 701.82 B
    # Expected: 7686.674 - 759.045 - 701.82 = 6225.809 B
    walcl_1 = 7686674.0
    tga_1 = 759045.0
    rrp_1 = 701.82
    expected_1 = (walcl_1 / 1000.0) - (tga_1 / 1000.0) - rrp_1
    assert round(expected_1, 2) == 6225.81

    # Date 2: 2025-01-08
    walcl_2 = 6980120.0
    tga_2 = 720500.0
    rrp_2 = 185.30
    expected_2 = (walcl_2 / 1000.0) - (tga_2 / 1000.0) - rrp_2
    assert round(expected_2, 2) == 6074.32

    # Date 3: 2026-11-18
    walcl_3 = 6741452.0
    tga_3 = 916613.0
    rrp_3 = 0.40
    expected_3 = (walcl_3 / 1000.0) - (tga_3 / 1000.0) - rrp_3
    assert round(expected_3, 2) == 5824.44

    dates = pd.to_datetime(["2024-01-03", "2025-01-08", "2026-11-18"])
    s_fed = pd.Series([walcl_1, walcl_2, walcl_3], index=dates)
    s_tga = pd.Series([tga_1, tga_2, tga_3], index=dates)
    s_rrp = pd.Series([rrp_1, rrp_2, rrp_3], index=dates)

    net_liq = compute_net_liquidity(s_fed, s_tga, s_rrp)
    assert round(net_liq.loc[dates[0]], 2) == 6225.81
    assert round(net_liq.loc[dates[1]], 2) == 6074.32
    assert round(net_liq.loc[dates[2]], 2) == 5824.44


def test_liquidity_alignment_and_changes():
    """Verify forward-fill frequency alignment and 4W / 13W change computations."""
    # Create 100 business days of synthetic data
    dates = pd.bdate_range(start="2025-01-01", periods=100)
    # Weekly releases on Wednesdays
    s_fed = pd.Series(7000000.0, index=dates).where(dates.weekday == 2).dropna()
    s_tga = pd.Series(800000.0, index=dates).where(dates.weekday == 2).dropna()
    # Daily RRP
    s_rrp = pd.Series(np.linspace(200.0, 100.0, len(dates)), index=dates)

    net_liq = compute_net_liquidity(s_fed, s_tga, s_rrp)
    assert len(net_liq) == 100
    # Net liquidity = 7000 - 800 - rrp = 6200 - rrp
    assert np.isclose(net_liq.iloc[0], 6200.0 - 200.0)
    assert np.isclose(net_liq.iloc[-1], 6200.0 - 100.0)

    changes = compute_liquidity_changes(net_liq, windows=[20, 65])
    assert "change_20" in changes
    assert "change_65" in changes
    assert "pct_20" in changes
    assert "pct_65" in changes
    assert pd.isna(changes["change_20"].iloc[0])
    assert not pd.isna(changes["change_20"].iloc[25])
    assert 0.0 <= changes["pct_20"].iloc[-1] <= 100.0


def test_liquidity_equity_overlay():
    """Verify rolling correlation between Net Liquidity and S&P 500."""
    dates = pd.bdate_range(start="2025-01-01", periods=80)
    net_liq = pd.Series(np.linspace(6000.0, 6500.0, 80), index=dates)
    # Perfectly co-trending equity index
    spx = pd.Series(np.linspace(5000.0, 5500.0, 80), index=dates)

    df_overlay = compute_liquidity_equity_overlay(net_liq, spx, corr_window=60)
    assert "net_liquidity" in df_overlay.columns
    assert "spx" in df_overlay.columns
    assert "rolling_corr_60" in df_overlay.columns

    # Perfectly correlated series should yield ~1.0 correlation
    valid_corr = df_overlay["rolling_corr_60"].dropna()
    assert len(valid_corr) > 0
    assert np.all(valid_corr >= -1.0) and np.all(valid_corr <= 1.0)
    assert np.isclose(valid_corr.iloc[-1], 1.0, atol=1e-2)


def test_compute_stablecoin_metrics():
    """Verify stablecoin supply tracking and 30-day percentage changes."""
    dates = pd.date_range("2026-01-01", periods=31, freq="D")
    supply = pd.Series(np.linspace(150.0, 165.0, 31), index=dates)  # $150B to $165B (10% growth over 30 days)

    res = compute_stablecoin_metrics(supply, window=30)
    assert res["latest_supply"] == 165.0
    assert np.isclose(res["change_30d_pct"], 10.0, atol=1e-1)
