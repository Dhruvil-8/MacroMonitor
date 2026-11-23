"""Historical Analogue Engine & Out-of-Sample Walk-Forward Evaluation (M20–M21).

Formulas & Assumptions:
- Strictly expanding prior standardization (`min_expanding_obs = 500`).
- Ledoit-Wolf analytic shrinkage covariance toward scaled identity target:
  Sigma_hat = (1 - delta) * S + delta * F, with F = (tr(S)/P) * I.
- Mahalanobis distance D(x_t, x_s) = sqrt((x_t - x_s)^T Sigma_hat^{-1} (x_t - x_s)).
- Effective independent features N_indep = (sum lambda_i)^2 / sum(lambda_i^2)
  from eigenvalues of the feature correlation matrix.
- 20-day declustered top-K match selection + seeded block bootstrap CI vs base rate.
- Walk-forward evaluation reports honest skill verdict: "improvement" iff
  `n_eval >= 36` and `mae_skill > 0.02`, otherwise historical context only.
"""

from __future__ import annotations

import math
import random
from typing import Any

from radar.metrics.core import compute_expanding_zscore_series


def _mean_cols(X: list[list[float]]) -> list[float]:
    p = len(X[0])
    n = len(X)
    return [sum(row[j] for row in X) / n for j in range(p)]


def ledoit_wolf_cov(X: list[list[float]]) -> tuple[list[list[float]], float]:
    """Compute Ledoit-Wolf shrinkage covariance matrix toward scaled identity."""
    n = len(X)
    p = len(X[0])
    if n < 3:
        ident = [[1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
        return ident, 1.0

    means = _mean_cols(X)
    Y = [[row[j] - means[j] for j in range(p)] for row in X]

    S = [[0.0] * p for _ in range(p)]
    for row in Y:
        for i in range(p):
            ri = row[i]
            for j in range(p):
                S[i][j] += ri * row[j]
    inv_n = 1.0 / n
    for i in range(p):
        for j in range(p):
            S[i][j] *= inv_n

    mu = sum(S[i][i] for i in range(p)) / p
    F = [[mu if i == j else 0.0 for j in range(p)] for i in range(p)]

    s_frob2 = sum(S[i][j] ** 2 for i in range(p) for j in range(p))
    gamma = sum((S[i][j] - F[i][j]) ** 2 for i in range(p) for j in range(p))

    phi_sum = 0.0
    for row in Y:
        norm2 = sum(v * v for v in row)
        quad = 0.0
        for i in range(p):
            s_row = sum(S[i][j] * row[j] for j in range(p))
            quad += row[i] * s_row
        phi_sum += max(0.0, norm2 * norm2 - 2.0 * quad + s_frob2)
    phi = phi_sum / (n * n)

    delta = 1.0 if gamma <= 1e-12 else min(1.0, max(0.0, phi / gamma))
    sigma = [
        [(1.0 - delta) * S[i][j] + delta * F[i][j] for j in range(p)]
        for i in range(p)
    ]
    return sigma, round(delta, 4)


def invert_matrix(mat: list[list[float]]) -> list[list[float]]:
    """Invert a small symmetric positive-definite matrix using Gauss-Jordan with ridge fallback."""
    p = len(mat)
    # Add tiny ridge for numerical safety
    aug = [
        [mat[i][j] + (1e-6 if i == j else 0.0) for j in range(p)]
        + [1.0 if i == j else 0.0 for j in range(p)]
        for i in range(p)
    ]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            aug[piv][col] = 1e-6
        if piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        div = aug[col][col]
        for j in range(2 * p):
            aug[col][j] /= div
        for r in range(p):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) > 1e-15:
                for j in range(2 * p):
                    aug[r][j] -= factor * aug[col][j]
    return [row[p:] for row in aug]


def symmetric_eigenvalues(mat: list[list[float]], max_iter: int = 80) -> list[float]:
    """Compute eigenvalues of a small symmetric matrix using cyclic Jacobi rotations."""
    p = len(mat)
    A = [[float(mat[i][j]) for j in range(p)] for i in range(p)]
    for _ in range(max_iter):
        best_val = 0.0
        pi, pj = 0, 1
        for i in range(p):
            for j in range(i + 1, p):
                if abs(A[i][j]) > best_val:
                    best_val = abs(A[i][j])
                    pi, pj = i, j
        if best_val < 1e-10:
            break
        app = A[pi][pi]
        aqq = A[pj][pj]
        apq = A[pi][pj]
        if abs(aqq - app) < 1e-12:
            theta = math.pi / 4.0
        else:
            theta = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c = math.cos(theta)
        s = math.sin(theta)
        for k in range(p):
            if k != pi and k != pj:
                aki = A[k][pi]
                akj = A[k][pj]
                A[k][pi] = c * aki - s * akj
                A[pi][k] = A[k][pi]
                A[k][pj] = s * aki + c * akj
                A[pj][k] = A[k][pj]
        A[pi][pi] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        A[pj][pj] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        A[pi][pj] = 0.0
        A[pj][pi] = 0.0
    return [max(0.0, A[i][i]) for i in range(p)]


def compute_n_indep(sigma: list[list[float]]) -> float:
    """Compute effective independent dimensions (sum lambda_i)^2 / sum lambda_i^2."""
    p = len(sigma)
    if p <= 1:
        return float(p)
    corr = [[0.0] * p for _ in range(p)]
    stds = [math.sqrt(max(1e-12, sigma[i][i])) for i in range(p)]
    for i in range(p):
        for j in range(p):
            corr[i][j] = sigma[i][j] / (stds[i] * stds[j])
    eigs = symmetric_eigenvalues(corr)
    s1 = sum(eigs)
    s2 = sum(e * e for e in eigs)
    if s2 <= 1e-12:
        return float(p)
    return round((s1 * s1) / s2, 2)


def mahalanobis_distance(x: list[float], y: list[float], inv_cov: list[list[float]]) -> float:
    diff = [a - b for a, b in zip(x, y)]
    quad = 0.0
    for i, di in enumerate(diff):
        row_dot = sum(inv_cov[i][j] * diff[j] for j in range(len(diff)))
        quad += di * row_dot
    return math.sqrt(max(0.0, quad))


def decluster_matches(
    candidates: list[tuple[float, int]],
    top_k: int = 15,
    gap_days: int = 20,
) -> list[tuple[float, int]]:
    """Greedily pick closest matches separated by at least `gap_days` indices."""
    ordered = sorted(candidates, key=lambda x: x[0])
    chosen: list[tuple[float, int]] = []
    for dist, idx in ordered:
        if all(abs(idx - prev_idx) >= gap_days for _, prev_idx in chosen):
            chosen.append((dist, idx))
            if len(chosen) >= top_k:
                break
    return chosen


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _forward_return(values: list[float | None], idx: int, h: int, is_rate: bool) -> float | None:
    if idx + h >= len(values):
        return None
    v0 = values[idx]
    v1 = values[idx + h]
    if v0 is None or v1 is None:
        return None
    if is_rate:
        return v1 - v0
    if v0 <= 0.0:
        return None
    return 100.0 * (v1 / v0 - 1.0)


def _forward_max_drawdown(values: list[float | None], idx: int, h: int, is_rate: bool) -> float | None:
    if is_rate or idx + h >= len(values):
        return None
    v0 = values[idx]
    if v0 is None or v0 <= 0.0:
        return None
    seg = [values[k] for k in range(idx, idx + h + 1) if values[k] is not None and values[k] > 0.0]
    if len(seg) < 2:
        return None
    peak = seg[0]
    worst = 0.0
    for v in seg:
        if v > peak:
            peak = v
        dd = 100.0 * (v / peak - 1.0)
        if dd < worst:
            worst = dd
    return worst


def _bootstrap_diff_ci(
    cond: list[float],
    base: list[float],
    iterations: int = 400,
    seed: int = 42,
) -> tuple[float, float]:
    if len(cond) < 3 or len(base) < 3:
        diff = (_quantile(sorted(cond), 0.5) - _quantile(sorted(base), 0.5)) if cond and base else 0.0
        return round(diff, 3), round(diff, 3)
    rng = random.Random(seed)
    diffs: list[float] = []
    nc, nb = len(cond), len(base)
    for _ in range(iterations):
        sc = sorted(cond[rng.randrange(0, nc)] for _ in range(nc))
        sb = sorted(base[rng.randrange(0, nb)] for _ in range(nb))
        diffs.append(_quantile(sc, 0.5) - _quantile(sb, 0.5))
    diffs.sort()
    return round(_quantile(diffs, 0.05), 3), round(_quantile(diffs, 0.95), 3)


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 5:
        return 0.0

    def rank(arr: list[float]) -> list[float]:
        indexed = sorted(enumerate(arr), key=lambda x: x[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_r = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_r
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    den = math.sqrt(dx * dy)
    return 0.0 if den <= 1e-12 else num / den


def _extract_preset_vectors(
    preset: dict[str, Any],
    asset_metrics: dict[str, dict[str, Any]],
    min_obs: int,
) -> tuple[list[dict[str, Any]], list[list[float] | None]]:
    metric_map = {
        "mom_20": "ret_20d",
        "vol20": "vol20",
        "drawdown_252d": "drawdown",
        "trend_tstat": "trend_tstat",
    }
    features = preset.get("features", [])
    dates = asset_metrics["SPX"]["dates"]
    z_cols: list[list[float | None]] = []
    for f in features:
        am = asset_metrics.get(str(f["asset"]))
        key = metric_map.get(str(f["metric"]), "ret_20d")
        if not am or key not in am:
            z_cols.append([None] * len(dates))
        else:
            z_cols.append(compute_expanding_zscore_series(am[key], min_obs=min_obs))

    vectors: list[list[float] | None] = []
    for i in range(len(dates)):
        row = [col[i] for col in z_cols]
        if any(v is None or not math.isfinite(v) for v in row):
            vectors.append(None)
        else:
            vectors.append([float(v) for v in row])
    return features, vectors


def evaluate_walk_forward_series(
    dates: list[str],
    vectors: list[list[float] | None],
    target_values: list[float | None],
    target_is_rate: bool,
    step_days: int = 21,
    horizon_days: int = 63,
    top_k: int = 15,
    gap_days: int = 20,
    min_prior_vectors: int = 252,
    skill_threshold: float = 0.02,
    min_eval_points: int = 36,
) -> dict[str, Any]:
    """Run strict out-of-sample walk-forward test for one preset vector sequence."""
    valid_indices = [i for i, v in enumerate(vectors) if v is not None]
    if len(valid_indices) < min_prior_vectors + horizon_days + 10:
        return {
            "n_eval": 0,
            "spearman_rho": 0.0,
            "iqr_coverage": 0.0,
            "mae_cond": None,
            "mae_base": None,
            "mae_skill": 0.0,
            "verdict_class": "no_reliable_improvement",
            "verdict_text": "Historical context only. Out-of-sample evaluation shows no reliable improvement over the base rate.",
            "subperiods": [],
        }

    first_valid = valid_indices[min_prior_vectors]
    last_eval = len(dates) - horizon_days - 1
    eval_points: list[dict[str, Any]] = []

    for t in range(first_valid, last_eval + 1, step_days):
        xt = vectors[t]
        if xt is None:
            continue
        actual = _forward_return(target_values, t, horizon_days, target_is_rate)
        if actual is None:
            continue
        hist_rows = [vectors[s] for s in range(0, t) if vectors[s] is not None]
        if len(hist_rows) < min_prior_vectors:
            continue
        cov, _ = ledoit_wolf_cov(hist_rows)
        inv_cov = invert_matrix(cov)

        candidates: list[tuple[float, int]] = []
        base_outcomes: list[float] = []
        for s in range(0, t - horizon_days):
            xs = vectors[s]
            if xs is None:
                continue
            r_s = _forward_return(target_values, s, horizon_days, target_is_rate)
            if r_s is None:
                continue
            base_outcomes.append(r_s)
            candidates.append((mahalanobis_distance(xt, xs, inv_cov), s))

        chosen = decluster_matches(candidates, top_k=top_k, gap_days=gap_days)
        cond_outcomes = sorted(
            r for r in (_forward_return(target_values, idx, horizon_days, target_is_rate) for _, idx in chosen) if r is not None
        )
        if len(cond_outcomes) < 5 or len(base_outcomes) < 20:
            continue
        c_p25 = _quantile(cond_outcomes, 0.25)
        c_med = _quantile(cond_outcomes, 0.50)
        c_p75 = _quantile(cond_outcomes, 0.75)
        b_med = _quantile(sorted(base_outcomes), 0.50)
        eval_points.append(
            {
                "date": dates[t],
                "pred_cond": c_med,
                "pred_base": b_med,
                "actual": actual,
                "in_iqr": c_p25 <= actual <= c_p75,
            }
        )

    n_eval = len(eval_points)
    if n_eval == 0:
        return {
            "n_eval": 0,
            "spearman_rho": 0.0,
            "iqr_coverage": 0.0,
            "mae_cond": None,
            "mae_base": None,
            "mae_skill": 0.0,
            "verdict_class": "no_reliable_improvement",
            "verdict_text": "Historical context only. Out-of-sample evaluation shows no reliable improvement over the base rate.",
            "subperiods": [],
        }

    preds = [p["pred_cond"] for p in eval_points]
    acts = [p["actual"] for p in eval_points]
    mae_cond = sum(abs(p["pred_cond"] - p["actual"]) for p in eval_points) / n_eval
    mae_base = sum(abs(p["pred_base"] - p["actual"]) for p in eval_points) / n_eval
    mae_skill = (1.0 - mae_cond / mae_base) if mae_base > 1e-12 else 0.0
    iqr_cov = sum(1 for p in eval_points if p["in_iqr"]) / n_eval
    rho = _spearman_rho(preds, acts)

    improved = n_eval >= min_eval_points and mae_skill > skill_threshold
    verdict_class = "improvement" if improved else "no_reliable_improvement"
    verdict_text = (
        "Out-of-sample evaluation shows modest error reduction over the unconditional base rate; descriptive historical context only."
        if improved
        else "Historical context only. Out-of-sample evaluation shows no reliable improvement over the base rate."
    )

    buckets = [
        ("2010–2015", "2010-01-01", "2015-12-31"),
        ("2016–2019", "2016-01-01", "2019-12-31"),
        ("2020–2022", "2020-01-01", "2022-12-31"),
        ("2023+", "2023-01-01", "2099-12-31"),
    ]
    subperiods: list[dict[str, Any]] = []
    for label, s_dt, e_dt in buckets:
        sub = [p for p in eval_points if s_dt <= p["date"] <= e_dt]
        if len(sub) < 4:
            continue
        mc = sum(abs(p["pred_cond"] - p["actual"]) for p in sub) / len(sub)
        mb = sum(abs(p["pred_base"] - p["actual"]) for p in sub) / len(sub)
        subperiods.append(
            {
                "label": label,
                "n_eval": len(sub),
                "spearman_rho": round(_spearman_rho([p["pred_cond"] for p in sub], [p["actual"] for p in sub]), 3),
                "iqr_coverage": round(sum(1 for p in sub if p["in_iqr"]) / len(sub), 3),
                "mae_skill": round((1.0 - mc / mb) if mb > 1e-12 else 0.0, 4),
            }
        )

    return {
        "n_eval": n_eval,
        "spearman_rho": round(rho, 4),
        "iqr_coverage": round(iqr_cov, 4),
        "mae_cond": round(mae_cond, 4),
        "mae_base": round(mae_base, 4),
        "mae_skill": round(mae_skill, 4),
        "verdict_class": verdict_class,
        "verdict_text": verdict_text,
        "subperiods": subperiods,
    }


def build_analogues_payloads(
    analogues_cfg: dict[str, Any],
    asset_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute analogue matches, outcome fans, base rate comparisons, and walk-forward scores."""
    eng = analogues_cfg.get("engine", {})
    min_obs = int(eng.get("min_expanding_obs", 500))
    gap_days = int(eng.get("decluster_gap_days", 20))
    top_k = int(eng.get("top_k", 15))
    horizons = {str(k): int(v) for k, v in eng.get("horizons_days", {"1M": 21, "3M": 63, "6M": 126}).items()}
    max_h = max(horizons.values())
    targets = [str(x) for x in eng.get("target_assets", ["SPX", "NDX", "US10Y", "DXY", "GOLD", "NIFTY50", "HYG"])]
    seed = int(eng.get("bootstrap_seed", 42))
    iters = int(eng.get("bootstrap_iterations", 400))
    wf_step = int(eng.get("walk_forward_step_days", 21))
    min_eval = int(eng.get("min_eval_points", 36))
    skill_thr = float(eng.get("skill_threshold", 0.02))

    dates = asset_metrics["SPX"]["dates"]
    presets_summary: list[dict[str, Any]] = []
    by_preset: dict[str, dict[str, Any]] = {}
    eval_rows: list[dict[str, Any]] = []

    for preset in analogues_cfg.get("presets", []):
        pid = str(preset["id"])
        features, vectors = _extract_preset_vectors(preset, asset_metrics, min_obs=min_obs)

        # Latest valid query index
        t = next((i for i in range(len(vectors) - 1, -1, -1) if vectors[i] is not None), None)
        if t is None:
            continue
        xt = vectors[t]
        assert xt is not None

        hist_rows = [vectors[s] for s in range(0, t) if vectors[s] is not None]
        cov, shrink_delta = ledoit_wolf_cov(hist_rows)
        inv_cov = invert_matrix(cov)
        n_indep = compute_n_indep(cov)

        candidates: list[tuple[float, int]] = []
        eligible_indices: list[int] = []
        for s in range(0, t - max_h):
            xs = vectors[s]
            if xs is None:
                continue
            eligible_indices.append(s)
            candidates.append((mahalanobis_distance(xt, xs, inv_cov), s))

        chosen = decluster_matches(candidates, top_k=top_k, gap_days=gap_days)
        eval_target = "NIFTY50" if pid == "india_flow_stress" else "SPX"
        wf_eval = evaluate_walk_forward_series(
            dates=dates,
            vectors=vectors,
            target_values=asset_metrics[eval_target]["values"],
            target_is_rate=bool(asset_metrics[eval_target]["is_rate"]),
            step_days=wf_step,
            horizon_days=horizons.get("3M", 63),
            top_k=top_k,
            gap_days=gap_days,
            min_prior_vectors=252,
            skill_threshold=skill_thr,
            min_eval_points=min_eval,
        )

        # Feature vector summary for Today + Top Matches
        matches_out: list[dict[str, Any]] = []
        for rank, (dist, idx) in enumerate(chosen, start=1):
            xs = vectors[idx] or []
            spx_3m = _forward_return(asset_metrics["SPX"]["values"], idx, horizons.get("3M", 63), False)
            us10y_3m = _forward_return(asset_metrics["US10Y"]["values"], idx, horizons.get("3M", 63), True) if "US10Y" in asset_metrics else None
            dd_3m = _forward_max_drawdown(asset_metrics["SPX"]["values"], idx, horizons.get("3M", 63), False)
            matches_out.append(
                {
                    "rank": rank,
                    "date": dates[idx],
                    "distance": round(dist, 3),
                    "similarity_score": round(100.0 / (1.0 + dist), 1),
                    "features": [round(v, 3) for v in xs],
                    "spx_fwd_3m": round(spx_3m, 2) if spx_3m is not None else None,
                    "us10y_fwd_3m": round(us10y_3m, 3) if us10y_3m is not None else None,
                    "spx_mdd_3m": round(dd_3m, 2) if dd_3m is not None else None,
                }
            )

        # Outcome Fans across targets and horizons
        outcome_fans: list[dict[str, Any]] = []
        for tgt in targets:
            am = asset_metrics.get(tgt)
            if not am:
                continue
            vals = am["values"]
            is_rate = bool(am["is_rate"])
            unit = "pp" if is_rate else "%"
            for h_lbl, h_days in horizons.items():
                cond = [
                    r for r in (_forward_return(vals, idx, h_days, is_rate) for _, idx in chosen) if r is not None
                ]
                base = [
                    r for r in (_forward_return(vals, idx, h_days, is_rate) for idx in eligible_indices) if r is not None
                ]
                if len(cond) < 3 or len(base) < 10:
                    continue
                sc = sorted(cond)
                sb = sorted(base)
                mdds = sorted(
                    m for m in (_forward_max_drawdown(vals, idx, h_days, is_rate) for _, idx in chosen) if m is not None
                )
                ci_lo, ci_hi = _bootstrap_diff_ci(cond, base, iterations=iters, seed=seed)
                outcome_fans.append(
                    {
                        "asset": tgt,
                        "unit": unit,
                        "horizon": h_lbl,
                        "horizon_days": h_days,
                        "n_matches": len(cond),
                        "n_base": len(base),
                        "cond_p10": round(_quantile(sc, 0.10), 2),
                        "cond_p25": round(_quantile(sc, 0.25), 2),
                        "cond_p50": round(_quantile(sc, 0.50), 2),
                        "cond_p75": round(_quantile(sc, 0.75), 2),
                        "cond_p90": round(_quantile(sc, 0.90), 2),
                        "base_p25": round(_quantile(sb, 0.25), 2),
                        "base_p50": round(_quantile(sb, 0.50), 2),
                        "base_p75": round(_quantile(sb, 0.75), 2),
                        "diff_median": round(_quantile(sc, 0.50) - _quantile(sb, 0.50), 2),
                        "diff_ci90_low": ci_lo,
                        "diff_ci90_high": ci_hi,
                        "cond_mdd_p50": round(_quantile(mdds, 0.50), 2) if mdds else None,
                        "cond_mdd_p90": round(_quantile(mdds, 0.10), 2) if mdds else None,
                    }
                )

        preset_doc = {
            "preset_id": pid,
            "name": preset["name"],
            "rationale": preset["rationale"],
            "query_date": dates[t],
            "n_features": len(features),
            "n_indep": n_indep,
            "shrinkage_delta": shrink_delta,
            "decluster_gap_days": gap_days,
            "feature_labels": [str(f["label"]) for f in features],
            "today_vector": [round(v, 3) for v in xt],
            "matches": matches_out,
            "outcome_fans": outcome_fans,
            "evaluation": {
                "target_asset": eval_target,
                **wf_eval,
            },
        }
        by_preset[pid] = preset_doc
        presets_summary.append(
            {
                "id": pid,
                "name": preset["name"],
                "rationale": preset["rationale"],
                "query_date": dates[t],
                "n_features": len(features),
                "n_indep": n_indep,
                "top_match_date": matches_out[0]["date"] if matches_out else None,
                "top_match_distance": matches_out[0]["distance"] if matches_out else None,
                "verdict_class": wf_eval["verdict_class"],
                "mae_skill": wf_eval["mae_skill"],
            }
        )
        eval_rows.append(
            {
                "preset_id": pid,
                "name": preset["name"],
                "target_asset": eval_target,
                **wf_eval,
            }
        )

    return {
        "presets_index": {"presets": presets_summary},
        "by_preset": by_preset,
        "evaluation": {"skill_threshold": skill_thr, "min_eval_points": min_eval, "presets": eval_rows},
    }
