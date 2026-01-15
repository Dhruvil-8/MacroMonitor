"""
File: tests/test_returns.py

Tests for return computation functions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.returns import compute_log_returns, compute_simple_returns


class TestComputeLogReturns:
    """Tests for log return computation."""
    
    def test_simple_log_returns(self):
        """Test log returns match expected formula."""
        prices = pd.DataFrame(
            {"asset": [100.0, 101.0, 102.0, 100.0]},
            index=pd.date_range("2024-01-01", periods=4),
        )
        
        returns = compute_log_returns(prices)
        
        # Manual calculation
        expected = np.log(np.array([101, 102, 100]) / np.array([100, 101, 102]))
        
        np.testing.assert_array_almost_equal(
            returns["asset"].values,
            expected,
            decimal=10,
        )
    
    def test_log_returns_formula(self):
        """Verify r_t = ln(price_t / price_{t-1})."""
        prices = pd.DataFrame(
            {"asset": [100.0, 110.0, 99.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        
        returns = compute_log_returns(prices)
        
        # Day 1: ln(110/100) = ln(1.1)
        assert np.isclose(returns["asset"].iloc[0], np.log(1.1))
        
        # Day 2: ln(99/110)
        assert np.isclose(returns["asset"].iloc[1], np.log(99/110))
    
    def test_empty_prices_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        prices = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty"):
            compute_log_returns(prices)
    
    def test_all_nan_column_raises_error(self):
        """Test that entirely NaN column raises ValueError."""
        prices = pd.DataFrame(
            {"asset": [np.nan, np.nan, np.nan]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        
        with pytest.raises(ValueError, match="empty columns"):
            compute_log_returns(prices)
    
    def test_partial_nan_handled(self):
        """Test that partial NaN values are handled gracefully."""
        prices = pd.DataFrame(
            {"asset": [100.0, np.nan, 102.0, 103.0]},
            index=pd.date_range("2024-01-01", periods=4),
        )
        
        returns = compute_log_returns(prices)
        
        # Should have some valid returns
        assert not returns["asset"].dropna().empty
    
    def test_output_dtype(self):
        """Test that output is float64."""
        prices = pd.DataFrame(
            {"asset": [100, 101, 102]},  # integers
            index=pd.date_range("2024-01-01", periods=3),
        )
        
        returns = compute_log_returns(prices)
        
        assert returns["asset"].dtype == np.float64
    
    def test_index_preserved(self):
        """Test that DatetimeIndex is preserved."""
        dates = pd.date_range("2024-01-01", periods=4)
        prices = pd.DataFrame(
            {"asset": [100.0, 101.0, 102.0, 103.0]},
            index=dates,
        )
        
        returns = compute_log_returns(prices)
        
        # Returns should have same type of index (minus first row)
        assert isinstance(returns.index, pd.DatetimeIndex)
    
    def test_multiple_columns(self):
        """Test with multiple assets."""
        prices = pd.DataFrame(
            {
                "asset_a": [100.0, 101.0, 102.0],
                "asset_b": [50.0, 51.0, 52.0],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )
        
        returns = compute_log_returns(prices)
        
        assert "asset_a" in returns.columns
        assert "asset_b" in returns.columns
        assert len(returns) == 2  # First row dropped


class TestComputeSimpleReturns:
    """Tests for simple return computation."""
    
    def test_simple_returns_formula(self):
        """Test simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}."""
        prices = pd.DataFrame(
            {"asset": [100.0, 110.0, 99.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        
        returns = compute_simple_returns(prices)
        
        # Day 1: (110-100)/100 = 0.1
        assert np.isclose(returns["asset"].iloc[0], 0.1)
        
        # Day 2: (99-110)/110 = -0.1
        assert np.isclose(returns["asset"].iloc[1], -0.1, atol=0.001)
    
    def test_log_vs_simple_for_small_moves(self):
        """For small moves, log and simple returns should be similar."""
        prices = pd.DataFrame(
            {"asset": [100.0, 100.5, 101.0, 100.8]},
            index=pd.date_range("2024-01-01", periods=4),
        )
        
        log_ret = compute_log_returns(prices)
        simple_ret = compute_simple_returns(prices)
        
        # Should be very close for small percentage moves
        np.testing.assert_array_almost_equal(
            log_ret["asset"].values,
            simple_ret["asset"].values,
            decimal=2,
        )
