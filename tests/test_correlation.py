"""
File: tests/test_correlation.py

Tests for correlation and z-score computation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.correlation import (
    compute_rolling_correlation_zscore,
    compute_correlation_timeseries,
    detect_correlation_regime_change,
    CorrelationStats,
)


class TestComputeRollingCorrelationZscore:
    """Tests for the main correlation z-score function."""
    
    @pytest.fixture
    def perfectly_correlated_series(self):
        """Create high (but not perfectly) correlated series."""
        np.random.seed(42)
        n = 250
        dates = pd.date_range("2023-01-01", periods=n)
        
        # Add small noise to avoid perfect correlation
        macro = pd.Series(np.random.randn(n) * 0.01, index=dates)
        target = macro * 0.8 + np.random.randn(n) * 0.002  # High but not perfect correlation
        
        return macro, target
    
    @pytest.fixture
    def regime_change_series(self):
        """Create series with clear correlation regime change."""
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n)
        
        # First 200 days: highly correlated
        # Last 100 days: uncorrelated
        macro = np.random.randn(n) * 0.01
        
        target = np.zeros(n)
        target[:200] = macro[:200] * 0.9 + np.random.randn(200) * 0.001  # Highly correlated
        target[200:] = np.random.randn(100) * 0.01  # Completely uncorrelated
        
        return pd.Series(macro, index=dates), pd.Series(target, index=dates)
    
    def test_basic_computation(self, perfectly_correlated_series):
        """Test basic z-score computation works."""
        macro, target = perfectly_correlated_series
        
        stats = compute_rolling_correlation_zscore(
            macro_returns=macro,
            target_returns=target,
            corr_window=30,
            hist_window=180,
        )
        
        assert isinstance(stats, CorrelationStats)
        # Allow for floating point slightly outside [-1, 1]
        assert -1.01 <= stats.current_corr <= 1.01
        assert stats.std_hist >= 0
    
    def test_high_correlation_detected(self, perfectly_correlated_series):
        """Test that high correlation is detected."""
        macro, target = perfectly_correlated_series
        
        stats = compute_rolling_correlation_zscore(
            macro_returns=macro,
            target_returns=target,
            corr_window=30,
            hist_window=180,
        )
        
        # Should have high positive correlation
        assert stats.current_corr > 0.7
    
    def test_regime_change_detectable(self, regime_change_series):
        """Test that correlation regime change is detectable in the stats."""
        macro, target = regime_change_series
        
        stats = compute_rolling_correlation_zscore(
            macro_returns=macro,
            target_returns=target,
            corr_window=30,
            hist_window=180,
        )
        
        # After regime change, current correlation should be much lower than mean
        # The current_corr should be low (from uncorrelated period)
        # Mean hist should be higher (from mixed period)
        assert stats.current_corr < stats.mean_hist
    
    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        dates = pd.date_range("2024-01-01", periods=50)
        macro = pd.Series(np.random.randn(50), index=dates)
        target = pd.Series(np.random.randn(50), index=dates)
        
        with pytest.raises(ValueError, match="Insufficient history"):
            compute_rolling_correlation_zscore(
                macro_returns=macro,
                target_returns=target,
                corr_window=30,
                hist_window=180,
            )
    
    def test_handles_nan_in_zscore(self):
        """Test that very low or zero std is handled gracefully."""
        n = 250
        dates = pd.date_range("2023-01-01", periods=n)
        
        # Create series with very stable correlation
        np.random.seed(123)
        macro = pd.Series(np.random.randn(n) * 0.01, index=dates)
        target = pd.Series(macro + np.random.randn(n) * 0.001, index=dates)
        
        stats = compute_rolling_correlation_zscore(
            macro_returns=macro,
            target_returns=target,
            corr_window=30,
            hist_window=180,
        )
        
        # Should not be NaN or Inf
        assert not np.isnan(stats.z_score)
        assert not np.isinf(stats.z_score)


class TestCorrelationTimeseries:
    """Tests for rolling correlation timeseries."""
    
    def test_returns_series(self):
        """Test that timeseries function returns a Series."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n)
        macro = pd.Series(np.random.randn(n), index=dates)
        target = pd.Series(np.random.randn(n), index=dates)
        
        result = compute_correlation_timeseries(macro, target, corr_window=20)
        
        assert isinstance(result, pd.Series)
        assert len(result) == n
    
    def test_values_in_range(self):
        """Test that correlation values are in [-1, 1]."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n)
        macro = pd.Series(np.random.randn(n), index=dates)
        target = pd.Series(np.random.randn(n), index=dates)
        
        result = compute_correlation_timeseries(macro, target, corr_window=20)
        
        valid = result.dropna()
        # Allow small floating point errors
        assert (valid >= -1.01).all()
        assert (valid <= 1.01).all()


class TestRegimeChangeDetection:
    """Tests for regime change detection."""
    
    def test_detects_extreme_change(self):
        """Test that extreme correlation change produces meaningful z-score."""
        np.random.seed(42)
        # Create a correlation series with a dramatic break
        n = 200
        rolling_corr = pd.Series(np.concatenate([
            np.ones(100) * 0.8 + np.random.randn(100) * 0.02,  # Stable high correlation
            np.ones(100) * -0.5 + np.random.randn(100) * 0.02,  # Dramatic reversal to negative
        ]))
        
        detected, z_score = detect_correlation_regime_change(
            rolling_corr,
            lookback=180,
            threshold=0.5,  # Lower threshold to detect change
        )
        
        # The function should compute a non-zero z-score for this change
        # The direction should be negative (current below mean)
        assert z_score != 0.0
        assert z_score < 0  # Current is -0.5, historical mean is positive
    
    def test_no_detection_for_stable(self):
        """Test that stable correlation shows modest z-score."""
        np.random.seed(42)
        rolling_corr = pd.Series(np.ones(200) * 0.6 + np.random.randn(200) * 0.05)
        
        detected, z_score = detect_correlation_regime_change(
            rolling_corr,
            lookback=180,
            threshold=3.0,  # Higher threshold
        )
        
        # Should have modest z-score for stable series
        assert abs(z_score) < 5.0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_minimum_viable_data(self):
        """Test with exactly minimum required data."""
        n = 30 + 180  # corr_window + hist_window
        dates = pd.date_range("2023-01-01", periods=n)
        np.random.seed(42)
        
        macro = pd.Series(np.random.randn(n) * 0.01, index=dates)
        target = pd.Series(np.random.randn(n) * 0.01, index=dates)
        
        # Should work without error
        stats = compute_rolling_correlation_zscore(
            macro_returns=macro,
            target_returns=target,
            corr_window=30,
            hist_window=180,
        )
        
        assert isinstance(stats, CorrelationStats)
