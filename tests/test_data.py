"""
File: tests/test_data.py

Tests for data engine functionality.

Tests:
- YAML loading
- Universe flattening
- Price store operations (with synthetic data)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.data_engine import (
    load_yaml,
    flatten_universe,
    prepare_analysis_ready_data,
    DataEngineError,
)


class TestLoadYaml:
    """Tests for load_yaml function."""
    
    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid YAML file."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnested:\n  inner: 123")
        
        result = load_yaml(yaml_file)
        
        assert result == {"key": "value", "nested": {"inner": 123}}
    
    def test_load_empty_yaml(self, tmp_path):
        """Test loading empty YAML returns empty dict."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        
        result = load_yaml(yaml_file)
        
        assert result == {}
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nonexistent.yaml")
    
    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises ValueError."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("key: [unclosed")
        
        with pytest.raises(ValueError, match="YAML parse error"):
            load_yaml(yaml_file)


class TestFlattenUniverse:
    """Tests for flatten_universe function."""
    
    def test_flatten_simple_universe(self):
        """Test flattening a simple universe."""
        universe = {
            "group1": {"asset_a": "TICK-A", "asset_b": "TICK-B"},
            "group2": {"asset_c": "TICK-C"},
        }
        
        result = flatten_universe(universe)
        
        assert result == {
            "asset_a": "TICK-A",
            "asset_b": "TICK-B",
            "asset_c": "TICK-C",
        }
    
    def test_flatten_empty_universe(self):
        """Test flattening empty universe."""
        result = flatten_universe({})
        assert result == {}
    
    def test_flatten_with_non_dict_group(self):
        """Test that non-dict groups are skipped."""
        universe = {
            "valid_group": {"asset": "TICK"},
            "invalid_group": "not a dict",
        }
        
        result = flatten_universe(universe)
        
        assert result == {"asset": "TICK"}
    
    def test_flatten_detects_duplicates(self):
        """Test that duplicate names raise error."""
        universe = {
            "group1": {"duplicate": "TICK-1"},
            "group2": {"duplicate": "TICK-2"},
        }
        
        with pytest.raises(DataEngineError, match="Duplicate asset name"):
            flatten_universe(universe)


class TestPrepareAnalysisReadyData:
    """Tests for prepare_analysis_ready_data function."""
    
    def test_rename_columns(self):
        """Test that ticker columns are renamed to semantic names."""
        prices = pd.DataFrame(
            {"TICK-A": [100, 101], "TICK-B": [200, 202]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        asset_map = {"asset_a": "TICK-A", "asset_b": "TICK-B"}
        
        result = prepare_analysis_ready_data(prices, asset_map)
        
        assert "asset_a" in result.columns
        assert "asset_b" in result.columns
        assert "TICK-A" not in result.columns
    
    def test_dtype_is_float64(self):
        """Test that output dtype is float64."""
        prices = pd.DataFrame(
            {"TICK": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        asset_map = {"asset": "TICK"}
        
        result = prepare_analysis_ready_data(prices, asset_map)
        
        assert result["asset"].dtype == np.float64
    
    def test_handles_missing_tickers(self):
        """Test graceful handling of missing tickers."""
        prices = pd.DataFrame(
            {"TICK-A": [100, 101]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        # asset_map references a ticker not in prices
        asset_map = {"asset_a": "TICK-A", "asset_b": "TICK-B"}
        
        # Should not raise, just warn
        result = prepare_analysis_ready_data(prices, asset_map)
        
        assert "asset_a" in result.columns


class TestSyntheticPriceData:
    """Tests using synthetic price data to avoid network calls."""
    
    @pytest.fixture
    def synthetic_prices(self):
        """Create synthetic price data."""
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        np.random.seed(42)
        
        # Random walk prices
        price_a = 100 * np.exp(np.cumsum(np.random.randn(365) * 0.01))
        price_b = 50 * np.exp(np.cumsum(np.random.randn(365) * 0.015))
        
        return pd.DataFrame(
            {"TICK-A": price_a, "TICK-B": price_b},
            index=dates,
        )
    
    def test_synthetic_prices_shape(self, synthetic_prices):
        """Test synthetic data has expected shape."""
        assert len(synthetic_prices) == 365
        assert len(synthetic_prices.columns) == 2
    
    def test_synthetic_prices_no_gaps(self, synthetic_prices):
        """Test synthetic data has no gaps."""
        # Should have data for all dates
        date_range = pd.date_range(
            synthetic_prices.index.min(),
            synthetic_prices.index.max(),
            freq="D",
        )
        assert len(synthetic_prices) == len(date_range)
    
    def test_missing_values_within_limit(self, synthetic_prices):
        """Test that missing values are within acceptable limit."""
        # For synthetic data, we expect 0% missing
        missing_pct = synthetic_prices.isnull().sum().sum() / synthetic_prices.size
        assert missing_pct <= 0.05  # Less than 5%
