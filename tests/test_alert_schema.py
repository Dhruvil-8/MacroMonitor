"""
File: tests/test_alert_schema.py

Tests for alert JSON schema validation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerts.formatter import (
    Alert,
    ALERT_SCHEMA,
    format_alert_json,
    validate_alert,
    format_alert_text,
)

import jsonschema


class TestAlertDataclass:
    """Tests for Alert dataclass."""
    
    def test_create_alert(self):
        """Test creating a basic alert."""
        alert = Alert(
            date="2024-01-15",
            macro="DXY",
            target="SP500",
            rationale="USD strength impacts equities",
            mean_corr=-0.5,
            curr_corr=-0.1,
            corr_z_score=2.5,
            beta=-0.3,
            macro_move_pct=1.5,
            target_move_pct=0.1,
            deviation=True,
            severity="high",
            extra={"test": True},
        )
        
        assert alert.date == "2024-01-15"
        assert alert.severity == "high"
    
    def test_to_dict(self):
        """Test converting alert to dictionary."""
        alert = Alert(
            date="2024-01-15",
            macro="DXY",
            target="SP500",
            rationale="Test",
            mean_corr=0.5,
            curr_corr=0.3,
            corr_z_score=2.0,
            beta=0.8,
            macro_move_pct=1.0,
            target_move_pct=0.8,
            deviation=False,
            severity="medium",
        )
        
        data = alert.to_dict()
        
        assert isinstance(data, dict)
        assert data["date"] == "2024-01-15"
        assert data["macro"] == "DXY"


class TestAlertJsonValidation:
    """Tests for JSON schema validation."""
    
    @pytest.fixture
    def valid_alert(self):
        """Create a valid alert."""
        return Alert(
            date="2024-01-15",
            macro="Brent",
            target="Europe",
            rationale="Energy price shock impacts European net importers",
            mean_corr=-0.62,
            curr_corr=-0.07,
            corr_z_score=2.45,
            beta=-0.42,
            macro_move_pct=-2.2,
            target_move_pct=0.12,
            deviation=True,
            severity="medium",
            extra={"hist_corr_window": 180, "corr_window": 30},
        )
    
    def test_valid_alert_passes_schema(self, valid_alert):
        """Test that valid alert passes schema validation."""
        data = format_alert_json(valid_alert)
        
        # Should not raise
        jsonschema.validate(instance=data, schema=ALERT_SCHEMA)
    
    def test_validate_alert_function(self, valid_alert):
        """Test validate_alert helper function."""
        data = valid_alert.to_dict()
        
        assert validate_alert(data) is True
    
    def test_invalid_severity_fails(self):
        """Test that invalid severity fails validation."""
        data = {
            "date": "2024-01-15",
            "macro": "DXY",
            "target": "SP500",
            "mean_corr": 0.5,
            "curr_corr": 0.3,
            "corr_z_score": 2.0,
            "beta": 0.8,
            "macro_move_pct": 1.0,
            "target_move_pct": 0.8,
            "deviation": False,
            "severity": "critical",  # Invalid!
        }
        
        assert validate_alert(data) is False
    
    def test_missing_required_field_fails(self):
        """Test that missing required field fails validation."""
        data = {
            "date": "2024-01-15",
            "macro": "DXY",
            # Missing "target" and other required fields
        }
        
        assert validate_alert(data) is False
    
    def test_extra_fields_allowed(self, valid_alert):
        """Test that extra object can have any fields."""
        alert = Alert(
            date="2024-01-15",
            macro="DXY",
            target="SP500",
            rationale="Test",
            mean_corr=0.5,
            curr_corr=0.3,
            corr_z_score=2.0,
            beta=0.8,
            macro_move_pct=1.0,
            target_move_pct=0.8,
            deviation=False,
            severity="medium",
            extra={
                "custom_field": 123,
                "nested": {"deeply": "nested"},
            },
        )
        
        data = format_alert_json(alert)
        
        assert validate_alert(data) is True


class TestAlertTextFormatting:
    """Tests for human-readable text formatting."""
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample alert for text formatting."""
        return Alert(
            date="2024-01-15",
            macro="Brent",
            target="Europe",
            rationale="Energy price shock impacts European net importers",
            mean_corr=-0.62,
            curr_corr=-0.07,
            corr_z_score=2.45,
            beta=-0.42,
            macro_move_pct=-2.2,
            target_move_pct=0.12,
            deviation=True,
            severity="high",
        )
    
    def test_text_contains_date(self, sample_alert):
        """Test that text includes the date."""
        text = format_alert_text(sample_alert)
        assert "2024-01-15" in text
    
    def test_text_contains_pair(self, sample_alert):
        """Test that text includes macro-target pair."""
        text = format_alert_text(sample_alert)
        assert "Brent" in text
        assert "Europe" in text
    
    def test_text_contains_metrics(self, sample_alert):
        """Test that text includes key metrics."""
        text = format_alert_text(sample_alert)
        assert "-2.20%" in text or "-2.2" in text  # Macro move
        assert "2.45" in text  # Z-score
    
    def test_text_contains_severity(self, sample_alert):
        """Test that text indicates severity."""
        text = format_alert_text(sample_alert)
        assert "HIGH" in text.upper()
    
    def test_text_is_multiline(self, sample_alert):
        """Test that text is properly formatted multiline."""
        text = format_alert_text(sample_alert)
        lines = text.strip().split("\n")
        assert len(lines) > 5  # Should have multiple lines


class TestAlertSchemaConformance:
    """Tests verifying schema matches spec exactly."""
    
    def test_schema_has_required_fields(self):
        """Test that schema requires all spec fields."""
        required = ALERT_SCHEMA["required"]
        
        spec_required = [
            "date", "macro", "target", "mean_corr", "curr_corr",
            "corr_z_score", "beta", "macro_move_pct", "target_move_pct",
            "deviation", "severity"
        ]
        
        for field in spec_required:
            assert field in required, f"Missing required field: {field}"
    
    def test_schema_severity_enum(self):
        """Test that severity uses correct enum values."""
        severity_schema = ALERT_SCHEMA["properties"]["severity"]
        
        assert severity_schema["enum"] == ["low", "medium", "high"]
    
    def test_schema_date_format(self):
        """Test that date field has date format."""
        date_schema = ALERT_SCHEMA["properties"]["date"]
        
        assert date_schema["type"] == "string"
        assert date_schema["format"] == "date"
