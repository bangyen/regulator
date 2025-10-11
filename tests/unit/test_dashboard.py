"""Unit tests for dashboard functionality."""

from typing import Any, Dict

import pytest

from dashboard.main import calculate_metrics, extract_time_series


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Create sample experiment data for testing."""
    return {
        "steps": [
            {
                "step": 1,
                "market_price": 10.5,
                "regulator_flags": {
                    "flagged": False,
                    "risk_score": 0.2,
                    "fines_applied": [0.0, 0.0],
                },
            },
            {
                "step": 2,
                "market_price": 12.0,
                "regulator_flags": {
                    "flagged": True,
                    "risk_score": 0.8,
                    "fines_applied": [10.0, 5.0],
                },
            },
            {
                "step": 3,
                "market_price": 11.5,
                "regulator_flags": {
                    "flagged": False,
                    "risk_score": 0.3,
                    "fines_applied": [0.0, 0.0],
                },
            },
        ]
    }


def test_calculate_metrics(sample_data: Dict[str, Any]) -> None:
    """Test metric calculation from experiment data."""
    metrics = calculate_metrics(sample_data)

    assert metrics["total_steps"] == 3
    assert metrics["avg_price"] == pytest.approx(11.33, abs=0.01)
    assert metrics["total_violations"] == 1
    assert metrics["total_fines"] == 15.0
    assert metrics["avg_risk_score"] == pytest.approx(0.43, abs=0.01)
    assert metrics["current_risk"] == 0.3


def test_calculate_metrics_empty_data() -> None:
    """Test metric calculation with empty data."""
    metrics = calculate_metrics({"steps": []})

    assert metrics == {}


def test_extract_time_series(sample_data: Dict[str, Any]) -> None:
    """Test time series extraction from experiment data."""
    time_series = extract_time_series(sample_data)

    assert "prices" in time_series
    assert "violations" in time_series
    assert "risk_scores" in time_series
    assert "fines" in time_series

    assert len(time_series["prices"]) == 3
    assert time_series["prices"][0] == {"x": 1, "y": 10.5}
    assert time_series["prices"][1] == {"x": 2, "y": 12.0}

    assert time_series["violations"][1] == {"x": 2, "y": 1}
    assert time_series["violations"][0] == {"x": 1, "y": 0}

    assert time_series["fines"][1] == {"x": 2, "y": 15.0}


def test_extract_time_series_empty_data() -> None:
    """Test time series extraction with empty data."""
    time_series = extract_time_series({"steps": []})

    assert time_series["prices"] == []
    assert time_series["violations"] == []
    assert time_series["risk_scores"] == []
    assert time_series["fines"] == []
