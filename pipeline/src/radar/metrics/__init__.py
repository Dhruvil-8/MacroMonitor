# pipeline/src/radar/metrics/__init__.py
"""Quantitative metrics package."""

from radar.metrics.correlation import (
    compute_correlation_shift,
    compute_heatmap_matrix,
    compute_rolling_correlation,
)
from radar.metrics.divergence import (
    compute_divergence_series,
    detect_divergences,
)
from radar.metrics.drawdown import compute_drawdown
from radar.metrics.returns_vol import (
    compute_realized_volatility,
    compute_returns,
    compute_vol_trend,
)
from radar.metrics.rotation import (
    classify_rotation_phase,
    compute_ratio_momentum,
    compute_relative_strength_ratio,
)
from radar.metrics.state_map import (
    classify_quadrant,
    compute_trend_tstat,
)
from radar.metrics.stats import (
    compute_effective_n,
    compute_expanding_percentile,
    compute_expanding_zscore,
    compute_lag1_autocorr,
)

__all__ = [
    "compute_returns",
    "compute_realized_volatility",
    "compute_vol_trend",
    "compute_drawdown",
    "compute_lag1_autocorr",
    "compute_effective_n",
    "compute_expanding_zscore",
    "compute_expanding_percentile",
    "compute_trend_tstat",
    "classify_quadrant",
    "compute_divergence_series",
    "detect_divergences",
    "compute_rolling_correlation",
    "compute_correlation_shift",
    "compute_heatmap_matrix",
    "compute_relative_strength_ratio",
    "compute_ratio_momentum",
    "classify_rotation_phase",
]
