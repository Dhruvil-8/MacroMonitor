"""Unit tests for Ledoit-Wolf Mahalanobis analogues, declustering, look-ahead safety, and walk-forward skill."""

from __future__ import annotations

import math
import random
import unittest

from radar.metrics.analogues import (
    decluster_matches,
    evaluate_walk_forward_series,
    invert_matrix,
    ledoit_wolf_cov,
    mahalanobis_distance,
)


class TestAnalogues(unittest.TestCase):
    def test_mahalanobis_known_answer_diagonal(self) -> None:
        cov = [[4.0, 0.0], [0.0, 1.0]]
        inv = invert_matrix(cov)
        d = mahalanobis_distance([2.0, 1.0], [0.0, 0.0], inv)
        # sqrt(2^2/4 + 1^2/1) = sqrt(2)
        self.assertAlmostEqual(d, math.sqrt(2.0), places=4)

    def test_ledoit_wolf_shrinks_singular_covariance(self) -> None:
        # Rank-1 collinear sample -> invertible after shrinkage
        X = [[float(i), float(i)] for i in range(1, 20)]
        sigma, delta = ledoit_wolf_cov(X)
        self.assertGreaterEqual(delta, 0.0)
        self.assertLessEqual(delta, 1.0)
        inv = invert_matrix(sigma)
        self.assertTrue(math.isfinite(inv[0][0]))

    def test_decluster_property_enforces_min_gap(self) -> None:
        candidates = [(0.1, 100), (0.12, 105), (0.15, 118), (0.20, 121), (0.25, 200)]
        chosen = decluster_matches(candidates, top_k=5, gap_days=20)
        indices = [idx for _, idx in chosen]
        self.assertEqual(indices, [100, 121, 200])

    def test_walk_forward_planted_signal_vs_noise(self) -> None:
        # 1. Pure i.i.d. Gaussian noise -> should show NO reliable improvement
        rng = random.Random(7)
        n = 900
        dates = [f"2020-{(i // 21) % 12 + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]
        vecs_noise = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(n)]
        vals_noise = [100.0]
        for _ in range(1, n):
            vals_noise.append(vals_noise[-1] * math.exp(rng.gauss(0.0002, 0.01)))
        res_noise = evaluate_walk_forward_series(
            dates=dates,
            vectors=vecs_noise,
            target_values=vals_noise,
            target_is_rate=False,
            step_days=21,
            horizon_days=21,
            top_k=10,
            gap_days=10,
            min_prior_vectors=120,
        )
        self.assertEqual(res_noise["verdict_class"], "no_reliable_improvement")

        # 2. Planted deterministic cycle signal where feature vector cleanly forecasts 21D forward direction
        vecs_sig = [[math.sin(i * 0.15), math.cos(i * 0.15)] for i in range(n)]
        vals_sig = [100.0 * math.exp(0.05 * math.sin((i - 21) * 0.15) + 0.0003 * i) for i in range(n)]
        res_sig = evaluate_walk_forward_series(
            dates=dates,
            vectors=vecs_sig,
            target_values=vals_sig,
            target_is_rate=False,
            step_days=15,
            horizon_days=21,
            top_k=10,
            gap_days=10,
            min_prior_vectors=120,
        )
        self.assertGreater(res_sig["mae_skill"], 0.15)
        self.assertGreater(res_sig["spearman_rho"], 0.4)
        self.assertEqual(res_sig["verdict_class"], "improvement")


if __name__ == "__main__":
    unittest.main()
