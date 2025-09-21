"""
Tests for the Streamlit dashboard application.

This module tests the dashboard functionality including:
- Episode data loading
- Plot generation
- Data validation
- Application startup
"""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import plotly.graph_objects as go
import pytest

# Import dashboard functions
from dashboard.app import (
    calculate_surplus,
    create_price_trajectory_plot,
    create_enhanced_regulator_plot,
    create_market_overview_plot,
    load_episode_data,
)


class TestDashboardFunctions:
    """Test class for dashboard utility functions."""

    def test_load_episode_data_valid_file(self) -> None:
        """Test loading episode data from a valid JSONL file."""
        # Create sample episode data
        sample_data = [
            {
                "type": "episode_header",
                "episode_id": "test_episode",
                "start_time": "2025-01-01T00:00:00",
                "n_firms": 2,
                "n_steps": 3,
                "agent_types": ["random", "titfortat"],
                "environment_params": {"price_min": 1.0, "price_max": 100.0},
            },
            {
                "type": "step",
                "step": 1,
                "timestamp": "2025-01-01T00:00:01",
                "prices": [50.0, 60.0],
                "profits": [100.0, 120.0],
                "demand_shock": 0.0,
                "market_price": 55.0,
                "total_demand": 45.0,
            },
            {
                "type": "step",
                "step": 2,
                "timestamp": "2025-01-01T00:00:02",
                "prices": [55.0, 65.0],
                "profits": [110.0, 130.0],
                "demand_shock": 1.0,
                "market_price": 60.0,
                "total_demand": 40.0,
            },
            {
                "type": "episode_summary",
                "episode_id": "test_episode",
                "end_time": "2025-01-01T00:00:03",
                "duration_seconds": 2.0,
                "total_steps": 2,
                "total_reward": 360.0,
                "final_prices": [55.0, 65.0],
                "final_profits": [110.0, 130.0],
            },
        ]

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for data in sample_data:
                f.write(json.dumps(data) + "\n")
            temp_file = Path(f.name)

        try:
            # Load episode data
            episode_data = load_episode_data(temp_file)

            # Verify structure
            assert episode_data["header"] is not None
            assert len(episode_data["steps"]) == 2
            assert episode_data["summary"] is not None

            # Verify header data
            header = episode_data["header"]
            assert header["episode_id"] == "test_episode"
            assert header["n_firms"] == 2
            assert header["agent_types"] == ["random", "titfortat"]

            # Verify step data
            steps = episode_data["steps"]
            assert steps[0]["step"] == 1
            assert steps[0]["prices"] == [50.0, 60.0]
            assert steps[1]["step"] == 2
            assert steps[1]["market_price"] == 60.0

            # Verify summary data
            summary = episode_data["summary"]
            assert summary["total_steps"] == 2
            assert summary["total_reward"] == 360.0

        finally:
            # Clean up
            temp_file.unlink()

    def test_load_episode_data_nonexistent_file(self) -> None:
        """Test loading episode data from a nonexistent file."""
        nonexistent_file = Path("nonexistent_file.jsonl")

        with pytest.raises(FileNotFoundError):
            load_episode_data(nonexistent_file)

    def test_calculate_surplus(self) -> None:
        """Test surplus calculation function."""
        prices = [50.0, 60.0]
        market_price = 55.0
        total_demand = 45.0
        demand_intercept = 100.0
        demand_slope = -1.0

        consumer_surplus, producer_surplus = calculate_surplus(
            prices, market_price, total_demand, demand_intercept, demand_slope
        )

        # Verify types
        assert isinstance(consumer_surplus, float)
        assert isinstance(producer_surplus, float)

        # Verify non-negative values
        assert consumer_surplus >= 0.0
        assert producer_surplus >= 0.0

        # Verify reasonable values (consumer surplus should be positive for reasonable prices)
        assert consumer_surplus > 0.0
        assert producer_surplus > 0.0

    def test_create_price_trajectory_plot(self) -> None:
        """Test price trajectory plot creation."""
        steps = [
            {"step": 1, "prices": [50.0, 60.0], "market_price": 55.0},
            {"step": 2, "prices": [55.0, 65.0], "market_price": 60.0},
        ]

        fig = create_price_trajectory_plot(steps)

        # Verify plot object
        assert isinstance(fig, go.Figure)

        # Verify traces (2 firms + 1 market price = 3 traces)
        assert len(fig.data) == 3

        # Verify plot title
        assert "Price Trajectories" in fig.layout.title.text

    def test_create_price_trajectory_plot_empty(self) -> None:
        """Test price trajectory plot creation with empty data."""
        fig = create_price_trajectory_plot([])

        # Verify empty plot
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_create_enhanced_regulator_plot(self) -> None:
        """Test enhanced regulator plot creation."""
        steps = [
            {
                "step": 1,
                "regulator_flags": {
                    "parallel_pricing": False,
                    "structural_break": True,
                    "fines": [0.0, 50.0],
                },
            },
            {
                "step": 2,
                "regulator_flags": {
                    "parallel_pricing": True,
                    "structural_break": False,
                    "fines": [25.0, 0.0],
                },
            },
        ]

        fig = create_enhanced_regulator_plot(steps)

        # Verify plot object
        assert isinstance(fig, go.Figure)

        # Enhanced plot shows multiple traces
        assert len(fig.data) >= 1

        # Verify plot title
        assert "Enhanced Regulator Monitoring" in fig.layout.title.text

    def test_create_market_overview_plot(self) -> None:
        """Test market overview plot creation."""
        steps = [
            {
                "step": 1,
                "prices": [50.0, 60.0],
                "market_price": 55.0,
                "total_demand": 45.0,
            },
            {
                "step": 2,
                "prices": [55.0, 65.0],
                "market_price": 60.0,
                "total_demand": 40.0,
            },
        ]

        fig = create_market_overview_plot(steps)

        # Verify plot object
        assert isinstance(fig, go.Figure)

        # Verify traces (market price, demand, volatility, HHI, demand shock = 5 traces)
        assert len(fig.data) == 5

        # Verify plot title
        assert "Market Overview" in fig.layout.title.text

    def test_create_market_overview_plot_empty(self) -> None:
        """Test market overview plot creation with empty data."""
        steps = []

        fig = create_market_overview_plot(steps)

        # Verify plot object
        assert isinstance(fig, go.Figure)

        # Verify empty plot has no traces
        assert len(fig.data) == 0

        # Verify plot title (empty plot may not have title)
        if fig.layout.title and fig.layout.title.text:
            assert "Market Overview" in fig.layout.title.text


class TestDashboardIntegration:
    """Integration tests for dashboard functionality."""

    def test_dashboard_app_imports(self) -> None:
        """Test that dashboard app can be imported without errors."""
        try:
            import dashboard.app as app

            assert hasattr(app, "main")
            assert hasattr(app, "load_episode_data")
            assert hasattr(app, "create_price_trajectory_plot")
        except ImportError as e:
            pytest.fail(f"Failed to import dashboard app: {e}")

    @patch("streamlit.set_page_config")
    @patch("streamlit.title")
    @patch("streamlit.sidebar")
    @patch("streamlit.error")
    def test_dashboard_app_runs(
        self, mock_error: Any, mock_sidebar: Any, mock_title: Any, mock_config: Any
    ) -> None:
        """Test that dashboard app can run without crashing."""
        # Mock streamlit components
        mock_sidebar.header.return_value = None
        mock_sidebar.selectbox.return_value = None
        mock_title.return_value = None
        mock_config.return_value = None
        mock_error.return_value = None

        # Mock Path operations
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False

            try:
                import dashboard.app as app

                # This should not crash
                app.main()
            except Exception as e:
                pytest.fail(f"Dashboard app crashed: {e}")

    def test_sample_log_produces_correct_plots(self) -> None:
        """Test that a sample log file produces the correct number of timesteps and plot objects."""
        # Create a more comprehensive sample episode
        sample_data = [
            {
                "type": "episode_header",
                "episode_id": "integration_test",
                "start_time": "2025-01-01T00:00:00",
                "n_firms": 2,
                "n_steps": 5,
                "agent_types": ["random", "titfortat"],
                "environment_params": {
                    "price_min": 1.0,
                    "price_max": 100.0,
                    "marginal_cost": 10.0,
                    "demand_intercept": 100.0,
                    "demand_slope": -1.0,
                },
            }
        ]

        # Add 5 steps with varied data
        for i in range(5):
            step_data = {
                "type": "step",
                "step": i + 1,
                "timestamp": f"2025-01-01T00:00:{i+1:02d}",
                "prices": [50.0 + i * 5, 60.0 + i * 3],
                "profits": [100.0 + i * 10, 120.0 + i * 8],
                "demand_shock": i * 0.5,
                "market_price": 55.0 + i * 4,
                "total_demand": 45.0 - i * 2,
                "regulator_flags": {
                    "parallel_pricing": i % 2 == 0,
                    "structural_break": i % 3 == 0,
                    "fines": [i * 5.0, i * 3.0],
                },
            }
            sample_data.append(step_data)

        # Add summary
        sample_data.append(
            {
                "type": "episode_summary",
                "episode_id": "integration_test",
                "end_time": "2025-01-01T00:00:06",
                "duration_seconds": 5.0,
                "total_steps": 5,
                "total_reward": 1000.0,
                "final_prices": [70.0, 72.0],
                "final_profits": [140.0, 152.0],
            }
        )

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for data in sample_data:
                f.write(json.dumps(data) + "\n")
            temp_file = Path(f.name)

        try:
            # Load episode data
            episode_data = load_episode_data(temp_file)

            # Verify correct number of timesteps
            assert len(episode_data["steps"]) == 5

            # Test all plot functions
            price_fig = create_price_trajectory_plot(episode_data["steps"])
            assert isinstance(price_fig, go.Figure)
            assert len(price_fig.data) == 3  # 2 firms + market price

            flags_fig = create_enhanced_regulator_plot(episode_data["steps"])
            assert isinstance(flags_fig, go.Figure)
            # Enhanced plot shows multiple traces
            assert len(flags_fig.data) >= 1

            market_fig = create_market_overview_plot(episode_data["steps"])
            assert isinstance(market_fig, go.Figure)
            assert (
                len(market_fig.data) == 5
            )  # Market price, demand, volatility, HHI, demand shock

            # Verify plot data contains expected number of points
            for fig in [price_fig, flags_fig, market_fig]:
                for trace in fig.data:
                    assert len(trace.x) == 5  # 5 timesteps
                    assert len(trace.y) == 5  # 5 timesteps

        finally:
            # Clean up
            temp_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
