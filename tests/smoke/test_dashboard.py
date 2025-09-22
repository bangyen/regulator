"""
Dashboard import and basic functionality tests for CI.

These tests verify that the dashboard can be imported and basic
functions work without running the full Streamlit application.
"""

import pytest
import json
import tempfile
from pathlib import Path


def test_dashboard_imports():
    """Test that all dashboard modules can be imported."""
    from dashboard.app import (
        load_episode_data,
        calculate_surplus,
        create_price_trajectory_plot,
        create_regulator_flags_plot,
        create_market_overview_plot,
        create_enhanced_regulator_plot,
        display_episode_summary,
        main,
    )

    # Test that all functions are callable
    assert callable(load_episode_data)
    assert callable(calculate_surplus)
    assert callable(create_price_trajectory_plot)
    assert callable(create_regulator_flags_plot)
    assert callable(create_market_overview_plot)
    assert callable(create_enhanced_regulator_plot)
    assert callable(display_episode_summary)
    assert callable(main)


def test_surplus_calculation():
    """Test surplus calculation with various inputs."""
    from dashboard.app import calculate_surplus

    # Test basic calculation
    consumer_surplus, producer_surplus = calculate_surplus(
        prices=[10.0, 12.0, 15.0],
        market_price=12.0,
        total_demand=50.0,
        demand_intercept=100.0,
        demand_slope=-1.0,
        marginal_cost=5.0,
    )

    assert isinstance(consumer_surplus, (int, float))
    assert isinstance(producer_surplus, (int, float))
    assert consumer_surplus >= 0
    # Producer surplus can be negative if selling below marginal cost

    # Test with individual quantities
    consumer_surplus2, producer_surplus2 = calculate_surplus(
        prices=[10.0, 12.0, 15.0],
        market_price=12.0,
        total_demand=50.0,
        individual_quantities=[20.0, 20.0, 10.0],
    )

    assert isinstance(consumer_surplus2, (int, float))
    assert isinstance(producer_surplus2, (int, float))


def test_episode_data_loading():
    """Test episode data loading with mock data."""
    from dashboard.app import load_episode_data

    # Create temporary JSONL file with mock data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write episode header
        header = {
            "type": "episode_header",
            "episode_id": 0,
            "n_firms": 2,
            "max_steps": 5,
            "agent_types": ["random", "random"],
            "seed": 42,
        }
        f.write(json.dumps(header) + "\n")

        # Write step data
        for step in range(3):
            step_data = {
                "type": "step",
                "step": step,
                "prices": [10.0 + step, 12.0 + step],
                "market_price": 11.0 + step,
                "total_demand": 50.0 - step * 5,
                "rewards": [5.0 + step, 6.0 + step],
                "done": step == 2,
            }
            f.write(json.dumps(step_data) + "\n")

        # Write episode summary
        summary = {
            "type": "episode_summary",
            "episode_id": 0,
            "total_reward": 100.0,
            "final_step": 3,
        }
        f.write(json.dumps(summary) + "\n")

        temp_file = Path(f.name)

    try:
        # Test loading
        episode_data = load_episode_data(temp_file)

        assert episode_data["header"] is not None
        assert episode_data["header"]["episode_id"] == 0
        assert len(episode_data["steps"]) == 3
        assert episode_data["summary"] is not None
        assert episode_data["summary"]["total_reward"] == 100.0

    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()


def test_plot_creation():
    """Test that plot creation functions work with mock data."""
    from dashboard.app import (
        create_price_trajectory_plot,
        create_regulator_flags_plot,
        create_market_overview_plot,
        create_enhanced_regulator_plot,
    )

    # Create mock step data
    steps = [
        {
            "step": 0,
            "prices": [10.0, 12.0],
            "market_price": 11.0,
            "total_demand": 50.0,
            "demand_shock": 0.0,
        },
        {
            "step": 1,
            "prices": [11.0, 13.0],
            "market_price": 12.0,
            "total_demand": 45.0,
            "demand_shock": 0.1,
        },
        {
            "step": 2,
            "prices": [12.0, 14.0],
            "market_price": 13.0,
            "total_demand": 40.0,
            "demand_shock": -0.1,
        },
    ]

    # Test price trajectory plot
    price_fig = create_price_trajectory_plot(steps)
    assert price_fig is not None
    assert hasattr(price_fig, "data")
    assert len(price_fig.data) > 0

    # Test regulator flags plot (should handle no regulator data gracefully)
    flags_fig = create_regulator_flags_plot(steps)
    assert flags_fig is not None
    assert hasattr(flags_fig, "data")

    # Test market overview plot
    market_fig = create_market_overview_plot(steps)
    assert market_fig is not None
    assert hasattr(market_fig, "data")
    assert len(market_fig.data) > 0

    # Test enhanced regulator plot (should handle no regulator data gracefully)
    enhanced_fig = create_enhanced_regulator_plot(steps)
    assert enhanced_fig is not None
    assert hasattr(enhanced_fig, "data")


def test_plot_creation_with_regulator_data():
    """Test plot creation with regulator monitoring data."""
    from dashboard.app import create_enhanced_regulator_plot

    # Create mock step data with regulator flags
    steps_with_regulator = [
        {
            "step": 0,
            "prices": [10.0, 12.0],
            "market_price": 11.0,
            "regulator_flags": {
                "fines_applied": [0.0, 0.0],
                "risk_score": 0.2,
                "market_volatility": 0.1,
                "parallel_violation": False,
                "structural_break_violation": False,
            },
        },
        {
            "step": 1,
            "prices": [11.0, 13.0],
            "market_price": 12.0,
            "regulator_flags": {
                "fines_applied": [5.0, 0.0],
                "risk_score": 0.5,
                "market_volatility": 0.2,
                "parallel_violation": True,
                "structural_break_violation": False,
            },
        },
    ]

    # Test enhanced regulator plot with data
    enhanced_fig = create_enhanced_regulator_plot(steps_with_regulator)
    assert enhanced_fig is not None
    assert hasattr(enhanced_fig, "data")
    assert len(enhanced_fig.data) > 0


def test_empty_data_handling():
    """Test that functions handle empty data gracefully."""
    from dashboard.app import (
        create_price_trajectory_plot,
        create_regulator_flags_plot,
        create_market_overview_plot,
        create_enhanced_regulator_plot,
    )

    # Test with empty steps
    empty_fig = create_price_trajectory_plot([])
    assert empty_fig is not None
    assert hasattr(empty_fig, "data")

    empty_flags_fig = create_regulator_flags_plot([])
    assert empty_flags_fig is not None

    empty_market_fig = create_market_overview_plot([])
    assert empty_market_fig is not None

    empty_enhanced_fig = create_enhanced_regulator_plot([])
    assert empty_enhanced_fig is not None


def test_streamlit_imports():
    """Test that Streamlit can be imported (without running the app)."""
    try:
        import streamlit as st

        assert st is not None
    except ImportError:
        pytest.skip("Streamlit not available")

    try:
        import plotly.graph_objects as go

        assert go is not None
    except ImportError:
        pytest.skip("Plotly not available")


def test_dashboard_dependencies():
    """Test that all dashboard dependencies can be imported."""
    try:
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go

        # Test basic functionality
        arr = np.array([1, 2, 3])
        assert len(arr) == 3

        df = pd.DataFrame({"a": [1, 2, 3]})
        assert len(df) == 3

        fig = go.Figure()
        assert fig is not None

    except ImportError as e:
        pytest.skip(f"Required dependency not available: {e}")


if __name__ == "__main__":
    # Run dashboard tests directly
    pytest.main([__file__, "-v"])
