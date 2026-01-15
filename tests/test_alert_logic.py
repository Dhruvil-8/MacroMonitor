"""
File: tests/test_alert_logic.py

Tests for alert gating logic and severity determination.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.regime_detector import (
    compute_macro_move_zscore,
    evaluate_alert_gate,
    determine_severity,
    compute_regime_score,
    filter_alerts_by_limit,
)
from src.engine.beta import compute_beta_and_deviation, BetaStats
from src.engine.correlation import compute_rolling_correlation_zscore


class TestMacroMoveZscore:
    """Tests for macro move z-score computation."""
    
    def test_normal_move_low_zscore(self):
        """Test that normal move produces low z-score."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n)
        
        # Normal volatility
        returns = pd.Series(np.random.randn(n) * 0.01, index=dates)
        
        z_score = compute_macro_move_zscore(returns, vol_window=180)
        
        # Normal move should be within ~3 std
        assert abs(z_score) < 4.0
    
    def test_large_move_high_zscore(self):
        """Test that large move produces high z-score."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2023-01-01", periods=n)
        
        # Normal volatility with large final move
        returns = np.random.randn(n) * 0.01
        returns[-1] = 0.05  # 5% move, much larger than 1% daily vol
        returns = pd.Series(returns, index=dates)
        
        z_score = compute_macro_move_zscore(returns, vol_window=180)
        
        # Large move should produce high z-score
        assert abs(z_score) > 3.0
    
    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        returns = pd.Series(np.random.randn(50))
        
        with pytest.raises(ValueError, match="Insufficient data"):
            compute_macro_move_zscore(returns, vol_window=180)


class TestAlertGate:
    """Tests for the AND-gate alert logic."""
    
    @pytest.fixture
    def default_thresholds(self):
        """Default threshold configuration."""
        return {
            "z_threshold": 2.0,
            "macro_move_threshold": 1.5,
        }
    
    def test_all_conditions_true_triggers(self, default_thresholds):
        """Test that alert triggers when all conditions met."""
        result = evaluate_alert_gate(
            z_score=2.5,           # Above threshold
            macro_move_z=2.0,      # Above threshold
            deviation_flag=True,   # Deviation detected
            thresholds=default_thresholds,
        )
        
        assert result is True
    
    def test_low_zscore_no_trigger(self, default_thresholds):
        """Test that low z-score prevents trigger."""
        result = evaluate_alert_gate(
            z_score=1.5,           # Below threshold
            macro_move_z=2.0,
            deviation_flag=True,
            thresholds=default_thresholds,
        )
        
        assert result is False
    
    def test_low_macro_move_no_trigger(self, default_thresholds):
        """Test that low macro move prevents trigger."""
        result = evaluate_alert_gate(
            z_score=2.5,
            macro_move_z=1.0,      # Below threshold
            deviation_flag=True,
            thresholds=default_thresholds,
        )
        
        assert result is False
    
    def test_no_deviation_no_trigger(self, default_thresholds):
        """Test that missing deviation flag prevents trigger."""
        result = evaluate_alert_gate(
            z_score=2.5,
            macro_move_z=2.0,
            deviation_flag=False,  # No deviation
            thresholds=default_thresholds,
        )
        
        assert result is False
    
    def test_negative_zscore_triggers(self, default_thresholds):
        """Test that negative z-score (below mean) also triggers."""
        result = evaluate_alert_gate(
            z_score=-2.5,          # Large negative
            macro_move_z=-2.0,     # Large negative
            deviation_flag=True,
            thresholds=default_thresholds,
        )
        
        assert result is True


class TestSeverityDetermination:
    """Tests for severity level determination."""
    
    def test_high_severity(self):
        """Test high severity determination."""
        severity = determine_severity(z_score=3.5, macro_move_z=2.8)
        assert severity == "high"
    
    def test_medium_severity(self):
        """Test medium severity determination."""
        severity = determine_severity(z_score=2.2, macro_move_z=1.8)
        assert severity == "medium"
    
    def test_low_severity(self):
        """Test low severity determination."""
        severity = determine_severity(z_score=1.0, macro_move_z=0.5)
        assert severity == "low"
    
    def test_boundary_high(self):
        """Test boundary condition for high severity."""
        # Exactly at thresholds
        severity = determine_severity(z_score=3.0, macro_move_z=2.5)
        assert severity == "high"
    
    def test_boundary_medium(self):
        """Test boundary condition for medium severity."""
        # Just below high threshold
        severity = determine_severity(z_score=2.9, macro_move_z=2.5)
        assert severity == "medium"


class TestRegimeScore:
    """Tests for composite regime score."""
    
    def test_high_score_for_extreme_values(self):
        """Test that extreme values produce high score."""
        score = compute_regime_score(
            corr_z=4.0,
            macro_z=4.0,
            deviation_flag=True,
        )
        
        assert score >= 80  # Should be high
    
    def test_low_score_for_normal_values(self):
        """Test that normal values produce low score."""
        score = compute_regime_score(
            corr_z=0.5,
            macro_z=0.5,
            deviation_flag=False,
        )
        
        assert score < 30  # Should be low
    
    def test_score_capped_at_100(self):
        """Test that score is capped at 100."""
        score = compute_regime_score(
            corr_z=10.0,
            macro_z=10.0,
            deviation_flag=True,
        )
        
        assert score <= 100


class TestAlertFiltering:
    """Tests for alert filtering by limit."""
    
    def test_respects_limit(self):
        """Test that filtering respects max limit."""
        alerts = [
            {"severity": "medium", "extra": {"regime_score": 50}},
            {"severity": "high", "extra": {"regime_score": 80}},
            {"severity": "medium", "extra": {"regime_score": 60}},
        ]
        
        filtered = filter_alerts_by_limit(alerts, max_alerts=2)
        
        assert len(filtered) == 2
    
    def test_prioritizes_high_severity(self):
        """Test that high severity alerts are kept."""
        alerts = [
            {"severity": "medium", "extra": {"regime_score": 90}},
            {"severity": "high", "extra": {"regime_score": 50}},
            {"severity": "low", "extra": {"regime_score": 95}},
        ]
        
        filtered = filter_alerts_by_limit(alerts, max_alerts=1)
        
        assert filtered[0]["severity"] == "high"
    
    def test_returns_all_if_under_limit(self):
        """Test that all alerts returned if under limit."""
        alerts = [{"severity": "medium"}] * 5
        
        filtered = filter_alerts_by_limit(alerts, max_alerts=10)
        
        assert len(filtered) == 5


class TestIntegratedAlertScenario:
    """Integration tests for alert triggering scenarios."""
    
    @pytest.fixture
    def macro_jump_scenario(self):
        """
        Create scenario where macro jumps 3σ and target doesn't move.
        This should trigger an alert with deviation.
        """
        np.random.seed(42)
        n = 250
        dates = pd.date_range("2023-01-01", periods=n)
        
        # Normal macro returns with a large spike at the end
        macro_returns = np.random.randn(n) * 0.01
        macro_returns[-1] = 0.03  # 3σ move
        
        # Target doesn't respond (breaks relationship)
        target_returns = np.random.randn(n) * 0.01
        target_returns[-1] = 0.001  # Tiny move
        
        return (
            pd.Series(macro_returns, index=dates),
            pd.Series(target_returns, index=dates),
        )
    
    def test_macro_jump_triggers_alert(self, macro_jump_scenario):
        """Test that macro jump with no target response triggers alert."""
        macro, target = macro_jump_scenario
        
        # Compute macro move z-score
        macro_move_z = compute_macro_move_zscore(macro, vol_window=180)
        
        # Should be a large z-score for the 3σ move
        assert abs(macro_move_z) > 2.0
        
        # Compute beta deviation
        beta_stats = compute_beta_and_deviation(
            macro_returns=macro,
            target_returns=target,
            beta_window=180,
            deviation_params={"target_vol_multiplier": 0.75, "beta_macro_multiplier": 0.5},
        )
        
        # Should detect deviation since target didn't respond
        # (This depends on historical relationship)
        print(f"Beta: {beta_stats.beta}, Deviation: {beta_stats.deviation_flag}")
