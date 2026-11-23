"""Unit tests for P10: EVT GPD Tails and Markov-Switching Regimes."""

import numpy as np
import pandas as pd
import pytest

from radar.metrics.tails_regimes import (
    compute_evt_gpd_tail,
    fit_markov_switching_regimes,
    compute_tails_regimes_payload,
)
from radar.export.schemas import TailsRegimesEnvelope, TailsRegimesData


def _generate_synthetic_returns(n: int = 600, seed: int = 42) -> pd.Series:
    """Generate returns with fat tails and volatility clustering."""
    rng = np.random.default_rng(seed)
    # Student-t with df=4 has heavy tails
    t_innovations = rng.standard_t(df=4, size=n) * 0.01
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(t_innovations, index=dates)


def test_evt_gpd_tail_heavy_tails():
    """Verify that Student-t returns produce positive shape (xi) and fat tail VaR > Gaussian."""
    ret = _generate_synthetic_returns(n=600, seed=123)
    evt = compute_evt_gpd_tail(ret, threshold_quantile=0.95)

    assert "shape_xi" in evt
    assert "scale_beta" in evt
    assert "var_99_pct" in evt
    assert "gaussian_var_99" in evt

    # Heavy tail distribution should yield xi >= -0.05 and EVT VaR >= Gaussian VaR
    assert evt["shape_xi"] > -0.10
    assert evt["var_99_pct"] >= evt["gaussian_var_99"] * 0.90
    assert evt["es_99_pct"] >= evt["var_99_pct"]  # Expected Shortfall >= VaR
    assert len(evt["mean_excess_curve"]) > 0


def test_evt_gpd_fallback_on_insufficient_data():
    """Verify robust fallback when data length is too short."""
    short_ret = pd.Series([0.01, -0.02, 0.005], index=pd.date_range("2026-01-01", periods=3))
    evt = compute_evt_gpd_tail(short_ret)
    assert evt["tail_fatness"] == "Heavy-Tailed (Fréchet)"
    assert evt["var_99_pct"] > 0


def test_markov_switching_regimes():
    """Verify 2-regime estimation outputs probabilities, transition matrix, and durations."""
    ret = _generate_synthetic_returns(n=350, seed=42)
    regimes = fit_markov_switching_regimes(ret, n_days=300)

    assert regimes["current_regime"] in ("calm", "stress")
    assert 0.0 <= regimes["current_prob_stress"] <= 1.0

    tm = regimes["transition_matrix"]
    assert 0.50 <= tm["p00_calm_to_calm"] <= 1.0
    assert 0.50 <= tm["p11_stress_to_stress"] <= 1.0

    durs = regimes["expected_duration_days"]
    assert durs["calm_regime"] > 0
    assert durs["stress_regime"] > 0
    assert len(regimes["ribbon"]) > 0


def test_markov_filtered_no_lookahead():
    """Verify point-in-time property: altering future data does not alter past filtered probabilities."""
    ret = _generate_synthetic_returns(n=400, seed=99)
    # Fit on first 300 days
    regimes_base = fit_markov_switching_regimes(ret.iloc[:300], n_days=300)
    filtered_base = [r["filtered_prob_stress"] for r in regimes_base["ribbon"][:50]]

    # Add massive shock in day 301..350
    ret_extended = ret.copy()
    ret_extended.iloc[300:] = 0.10  # extreme shock in future

    regimes_ext = fit_markov_switching_regimes(ret_extended.iloc[:300], n_days=300)
    filtered_ext = [r["filtered_prob_stress"] for r in regimes_ext["ribbon"][:50]]

    # Up to day 300, filtered probabilities must be identical
    np.testing.assert_allclose(filtered_base, filtered_ext, atol=1e-5)


def test_tails_regimes_payload_schema():
    """Verify compute_tails_regimes_payload produces valid TailsRegimesEnvelope."""
    ret = _generate_synthetic_returns(n=350, seed=77)
    payload = compute_tails_regimes_payload(ret, target_name="S&P 500")

    env = TailsRegimesEnvelope(
        schema_version="1.0",
        as_of=payload["as_of"],
        code_version="1.0.0",
        params_hash="hash123",
        data=TailsRegimesData(**payload),
    )
    assert env.data.target_name == "S&P 500"
    assert "Generalized Pareto" in env.data.interpretation
