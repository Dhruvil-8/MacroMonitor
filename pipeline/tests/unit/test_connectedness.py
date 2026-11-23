"""Unit tests for P7: Diebold-Yilmaz GFEVD Connectedness Framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.metrics.connectedness import (
    fit_var1,
    compute_gfevd_matrix,
    compute_connectedness_metrics,
    compute_connectedness_payload,
)


def test_var1_parameter_recovery():
    """Verify VAR(1) recovers coefficient matrix in simulation."""
    np.random.seed(42)
    T = 800
    N = 3
    # True Phi
    Phi_true = np.array([
        [0.4, 0.1, 0.0],
        [0.2, 0.3, 0.1],
        [0.0, 0.1, 0.5],
    ])
    Sigma_true = np.diag([0.01, 0.01, 0.01])

    Y = np.zeros((T, N))
    for t in range(1, T):
        eps = np.random.multivariate_normal([0, 0, 0], Sigma_true)
        Y[t] = Y[t - 1] @ Phi_true + eps

    Phi_est, Sigma_est = fit_var1(Y)
    np.testing.assert_allclose(Phi_est, Phi_true, atol=0.12)


def test_gfevd_normalization_and_order_invariance():
    """Verify Pesaran-Shin GFEVD rows sum to 100% and total connectedness is invariant to variable ordering."""
    np.random.seed(123)
    Phi = np.array([
        [0.3, 0.2],
        [0.1, 0.4],
    ])
    Sigma = np.array([
        [0.02, 0.005],
        [0.005, 0.015],
    ])

    theta1 = compute_gfevd_matrix(Phi, Sigma, H=10)
    # Check that rows sum to 100%
    np.testing.assert_allclose(theta1.sum(axis=1), [100.0, 100.0], atol=1e-5)

    # Permute variable order: swap 0 and 1
    P = np.array([[0, 1], [1, 0]])
    Phi_perm = P @ Phi @ P.T
    Sigma_perm = P @ Sigma @ P.T

    theta2 = compute_gfevd_matrix(Phi_perm, Sigma_perm, H=10)
    # The diagonal elements should swap, and total off-diagonal connectedness should be identical
    metrics1 = compute_connectedness_metrics(theta1, ["A", "B"])
    metrics2 = compute_connectedness_metrics(theta2, ["B", "A"])

    assert abs(metrics1["tci"] - metrics2["tci"]) < 1e-4


def test_compute_connectedness_payload():
    dates = pd.date_range("2024-01-01", periods=150, freq="W-FRI")
    df = pd.DataFrame(
        {
            "US": np.random.normal(0, 0.02, size=150),
            "India": np.random.normal(0, 0.025, size=150),
            "Germany": np.random.normal(0, 0.022, size=150),
            "UK": np.random.normal(0, 0.02, size=150),
        },
        index=dates,
    )

    payload = compute_connectedness_payload(df, H=10, window_weeks=52)
    assert payload["tci"] > 0
    assert len(payload["countries"]) == 4
    assert len(payload["matrix"]) == 4
    assert len(payload["tci_history"]) > 0
    # Every row in matrix must sum to 100
    for row in payload["matrix"]:
        row_sum = sum(row[c] for c in payload["labels"])
        assert abs(row_sum - 100.0) < 0.1
