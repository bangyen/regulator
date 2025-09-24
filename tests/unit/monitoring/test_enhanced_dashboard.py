"""
Tests for the EnhancedMonitoringDashboard class.

This module tests the enhanced monitoring dashboard functionality including
data loading, visualization creation, and report generation.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from src.monitoring.enhanced_dashboard import EnhancedMonitoringDashboard


class TestEnhancedMonitoringDashboard:
    """Test the EnhancedMonitoringDashboard class."""

    @pytest.fixture
    def dashboard(self):
        """Create a basic dashboard for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard = EnhancedMonitoringDashboard(log_dir=temp_dir)
            yield dashboard

    @pytest.fixture
    def sample_episode_data(self):
        """Create sample episode data for testing."""
        return {
            "steps": [
                {
                    "step": 1,
                    "prices": [50.0, 52.0, 48.0],
                    "profits": [1000.0, 1100.0, 900.0],
                    "market_price": 50.0,
                    "total_demand": 100.0,
                    "regulator_flags": {
                        "parallel_violation": False,
                        "structural_break_violation": False,
                        "fines_applied": [0.0, 0.0, 0.0],
                        "market_volatility": 0.1,
                        "violation_severities": [],
                        "penalty_multipliers": [],
                    },
                },
                {
                    "step": 2,
                    "prices": [51.0, 53.0, 49.0],
                    "profits": [1050.0, 1150.0, 950.0],
                    "market_price": 51.0,
                    "total_demand": 95.0,
                    "regulator_flags": {
                        "parallel_violation": True,
                        "structural_break_violation": False,
                        "fines_applied": [25.0, 25.0, 25.0],
                        "market_volatility": 0.2,
                        "violation_severities": ["severe", "severe", "severe"],
                        "penalty_multipliers": [1.0, 1.0, 1.0],
                    },
                },
            ]
        }

    def test_initialization(self, dashboard):
        """Test dashboard initialization."""
        assert dashboard.log_dir is not None
        assert dashboard.colors is not None
        assert "market_volatility" in dashboard.colors
        assert "penalties" in dashboard.colors
        assert "severe_violations" in dashboard.colors
        assert "moderate_violations" in dashboard.colors
        assert "minor_violations" in dashboard.colors

    def test_initialization_custom_log_dir(self):
        """Test dashboard initialization with custom log directory."""
        custom_dir = "/custom/log/dir"
        dashboard = EnhancedMonitoringDashboard(log_dir=custom_dir)
        assert dashboard.log_dir == Path(custom_dir)

    def test_load_episode_data_success(self, dashboard, sample_episode_data):
        """Test successful episode data loading."""
        # Create a test episode file
        episode_file = "test_episode.jsonl"
        episode_path = dashboard.log_dir / episode_file

        with open(episode_path, "w") as f:
            for step in sample_episode_data["steps"]:
                f.write(json.dumps({"type": "step", **step}) + "\n")

        # Load the data
        loaded_data = dashboard.load_episode_data(episode_file)

        assert "steps" in loaded_data
        assert len(loaded_data["steps"]) == 2
        assert loaded_data["steps"][0]["step"] == 1
        assert loaded_data["steps"][1]["step"] == 2

    def test_load_episode_data_file_not_found(self, dashboard):
        """Test episode data loading when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Episode file not found"):
            dashboard.load_episode_data("nonexistent.jsonl")

    def test_load_episode_data_empty_file(self, dashboard):
        """Test episode data loading with empty file."""
        episode_file = "empty_episode.jsonl"
        episode_path = dashboard.log_dir / episode_file

        with open(episode_path, "w"):
            pass  # Empty file

        loaded_data = dashboard.load_episode_data(episode_file)

        assert "steps" in loaded_data
        assert len(loaded_data["steps"]) == 0

    def test_load_episode_data_invalid_json(self, dashboard):
        """Test episode data loading with invalid JSON."""
        episode_file = "invalid_episode.jsonl"
        episode_path = dashboard.log_dir / episode_file

        with open(episode_path, "w") as f:
            f.write("invalid json line\n")
            f.write('{"type": "step", "step": 1, "prices": [50.0]}\n')

        # Should handle invalid JSON gracefully
        loaded_data = dashboard.load_episode_data(episode_file)

        assert "steps" in loaded_data
        assert len(loaded_data["steps"]) == 1  # Only valid line

    def test_load_episode_data_filter_non_step_entries(self, dashboard):
        """Test that only step entries are loaded."""
        episode_file = "mixed_episode.jsonl"
        episode_path = dashboard.log_dir / episode_file

        with open(episode_path, "w") as f:
            f.write('{"type": "step", "step": 1, "prices": [50.0]}\n')
            f.write('{"type": "other", "data": "ignored"}\n')
            f.write('{"type": "step", "step": 2, "prices": [51.0]}\n')

        loaded_data = dashboard.load_episode_data(episode_file)

        assert len(loaded_data["steps"]) == 2
        assert loaded_data["steps"][0]["step"] == 1
        assert loaded_data["steps"][1]["step"] == 2

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.suptitle")
    def test_create_comprehensive_dashboard_success(
        self,
        mock_suptitle,
        mock_savefig,
        mock_tight_layout,
        mock_subplots,
        dashboard,
        sample_episode_data,
    ):
        """Test successful comprehensive dashboard creation."""
        # Setup mocks
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()], [Mock(), Mock()]])

        # Mock the bar method to return an iterable
        for row in mock_axes:
            for ax in row:
                mock_bar = Mock()
                mock_bar.get_height.return_value = 1.0
                mock_bar.get_x.return_value = 0.0
                mock_bar.get_width.return_value = 0.8
                ax.bar.return_value = [
                    mock_bar,
                    mock_bar,
                    mock_bar,
                    mock_bar,
                ]  # Mock bars

        mock_subplots.return_value = (mock_fig, mock_axes)

        # Create test episode files
        episode_files = ["episode1.jsonl", "episode2.jsonl"]
        for episode_file in episode_files:
            episode_path = dashboard.log_dir / episode_file
            with open(episode_path, "w") as f:
                for step in sample_episode_data["steps"]:
                    f.write(json.dumps({"type": "step", **step}) + "\n")

        # Create dashboard
        dashboard.create_comprehensive_dashboard(episode_files)

        # Verify calls
        mock_subplots.assert_called_once_with(3, 2, figsize=(20, 15))
        mock_tight_layout.assert_called_once()
        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    def test_create_comprehensive_dashboard_no_data(self, mock_subplots, dashboard):
        """Test dashboard creation with no episode data."""
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()], [Mock(), Mock()]])
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Create dashboard with non-existent files
        episode_files = ["nonexistent1.jsonl", "nonexistent2.jsonl"]

        with patch("builtins.print") as mock_print:
            dashboard.create_comprehensive_dashboard(episode_files)

            # Should print warning about no data
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any("No episode data found!" in call for call in print_calls)

    @patch("matplotlib.pyplot.subplots")
    def test_create_comprehensive_dashboard_file_not_found(
        self, mock_subplots, dashboard, sample_episode_data
    ):
        """Test dashboard creation with some files not found."""
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()], [Mock(), Mock()]])

        # Mock the bar method to return an iterable
        for row in mock_axes:
            for ax in row:
                mock_bar = Mock()
                mock_bar.get_height.return_value = 1.0
                mock_bar.get_x.return_value = 0.0
                mock_bar.get_width.return_value = 0.8
                ax.bar.return_value = [
                    mock_bar,
                    mock_bar,
                    mock_bar,
                    mock_bar,
                ]  # Mock bars

        mock_subplots.return_value = (mock_fig, mock_axes)

        # Create one valid episode file
        episode_file = "valid_episode.jsonl"
        episode_path = dashboard.log_dir / episode_file
        with open(episode_path, "w") as f:
            for step in sample_episode_data["steps"]:
                f.write(json.dumps({"type": "step", **step}) + "\n")

        episode_files = [episode_file, "nonexistent.jsonl"]

        with patch("builtins.print") as mock_print:
            dashboard.create_comprehensive_dashboard(episode_files)

            # Should print warning about missing file
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any("Warning: Could not load" in call for call in print_calls)

    def test_plot_fines_over_time(self, dashboard, sample_episode_data):
        """Test fines over time plotting."""
        mock_ax = Mock()
        episode_data = {"test_episode.jsonl": sample_episode_data}

        dashboard._plot_fines_over_time(mock_ax, episode_data)

        # Verify ax methods were called
        mock_ax.set_title.assert_called_once_with("Fines Over Time", fontweight="bold")
        mock_ax.set_xlabel.assert_called_once_with("Step")
        mock_ax.set_ylabel.assert_called_once_with("Total Fines")
        mock_ax.grid.assert_called_once_with(True, alpha=0.3)
        mock_ax.legend.assert_called_once()

    def test_plot_fines_over_time_empty_data(self, dashboard):
        """Test fines over time plotting with empty data."""
        mock_ax = Mock()
        episode_data = {"empty_episode.jsonl": {"steps": []}}

        dashboard._plot_fines_over_time(mock_ax, episode_data)

        # Should handle empty data gracefully
        mock_ax.set_title.assert_called_once()

    def test_plot_market_volatility(self, dashboard, sample_episode_data):
        """Test market volatility plotting."""
        mock_ax = Mock()
        episode_data = {"test_episode.jsonl": sample_episode_data}

        dashboard._plot_market_volatility(mock_ax, episode_data)

        # Verify ax methods were called
        mock_ax.set_title.assert_called_once_with(
            "Market Volatility Analysis", fontweight="bold"
        )
        mock_ax.set_xlabel.assert_called_once_with("Step")
        mock_ax.set_ylabel.assert_called_once_with("Market Volatility")
        mock_ax.grid.assert_called_once_with(True, alpha=0.3)
        mock_ax.legend.assert_called_once()
        mock_ax.axhline.assert_called_once()  # Threshold line

    def test_plot_penalty_analysis(self, dashboard, sample_episode_data):
        """Test penalty analysis plotting."""
        mock_ax = Mock()
        episode_data = {"test_episode.jsonl": sample_episode_data}

        dashboard._plot_penalty_analysis(mock_ax, episode_data)

        # Verify ax methods were called
        mock_ax.set_title.assert_called_once_with(
            "Penalty Analysis and Cumulative Fines", fontweight="bold"
        )
        mock_ax.set_xlabel.assert_called_once_with("Step")
        mock_ax.set_ylabel.assert_called_once_with("Cumulative Fines")
        mock_ax.grid.assert_called_once_with(True, alpha=0.3)
        mock_ax.legend.assert_called_once()

    def test_plot_violation_severity(self, dashboard, sample_episode_data):
        """Test violation severity plotting."""
        mock_ax = Mock()
        mock_bar = Mock()
        mock_bar.get_height.return_value = 1.0
        mock_bar.get_x.return_value = 0.0
        mock_bar.get_width.return_value = 0.8
        mock_ax.bar.return_value = [mock_bar, mock_bar, mock_bar, mock_bar]  # Mock bars
        episode_data = {"test_episode.jsonl": sample_episode_data}

        dashboard._plot_violation_severity(mock_ax, episode_data)

        # Verify ax methods were called
        mock_ax.set_title.assert_called_once_with(
            "Violation Severity Distribution", fontweight="bold"
        )
        mock_ax.set_xlabel.assert_called_once_with("Violation Severity")
        mock_ax.set_ylabel.assert_called_once_with("Count")
        mock_ax.grid.assert_called_once_with(True, alpha=0.3)
        mock_ax.bar.assert_called_once()
        mock_ax.text.assert_called()  # Value labels on bars

    def test_plot_violation_severity_no_violations(self, dashboard):
        """Test violation severity plotting with no violations."""
        mock_ax = Mock()
        mock_bar = Mock()
        mock_bar.get_height.return_value = 1.0
        mock_bar.get_x.return_value = 0.0
        mock_bar.get_width.return_value = 0.8
        mock_ax.bar.return_value = [mock_bar, mock_bar, mock_bar, mock_bar]  # Mock bars
        episode_data = {
            "no_violations.jsonl": {
                "steps": [{"regulator_flags": {"violation_severities": []}}]
            }
        }

        dashboard._plot_violation_severity(mock_ax, episode_data)

        # Should handle no violations gracefully
        mock_ax.set_title.assert_called_once()

    @patch("pandas.DataFrame")
    def test_plot_monitoring_summary(
        self, mock_dataframe, dashboard, sample_episode_data
    ):
        """Test monitoring summary plotting."""
        mock_ax = Mock()
        mock_df = Mock()
        mock_dataframe.return_value = mock_df
        episode_data = {"test_episode.jsonl": sample_episode_data}

        dashboard._plot_monitoring_summary(mock_ax, episode_data)

        # Verify ax methods were called
        mock_ax.set_title.assert_called_once_with(
            "Monitoring Summary Statistics", fontweight="bold"
        )
        mock_ax.axis.assert_called()
        mock_dataframe.assert_called_once()

    def test_plot_monitoring_summary_empty_data(self, dashboard):
        """Test monitoring summary plotting with empty data."""
        mock_ax = Mock()
        episode_data = {"empty_episode.jsonl": {"steps": []}}

        dashboard._plot_monitoring_summary(mock_ax, episode_data)

        # Should handle empty data gracefully
        mock_ax.set_title.assert_called_once()

    def test_generate_monitoring_report_success(self, dashboard, sample_episode_data):
        """Test successful monitoring report generation."""
        # Create test episode files
        episode_files = ["episode1.jsonl", "episode2.jsonl"]
        for episode_file in episode_files:
            episode_path = dashboard.log_dir / episode_file
            with open(episode_path, "w") as f:
                for step in sample_episode_data["steps"]:
                    f.write(json.dumps({"type": "step", **step}) + "\n")

        # Generate report
        dashboard.generate_monitoring_report(episode_files)

        # Check that report file was created
        report_path = dashboard.log_dir / "monitoring_report.txt"
        assert report_path.exists()

        # Check report content
        with open(report_path, "r") as f:
            content = f.read()
            assert "ENHANCED REGULATOR MONITORING REPORT" in content
            assert "episode1.jsonl" in content
            assert "episode2.jsonl" in content
            assert "Total Steps: 2" in content
            assert "Parallel Violations: 1" in content
            assert "Structural Break Violations: 0" in content

    def test_generate_monitoring_report_custom_filename(
        self, dashboard, sample_episode_data
    ):
        """Test monitoring report generation with custom filename."""
        # Create test episode file
        episode_file = "test_episode.jsonl"
        episode_path = dashboard.log_dir / episode_file
        with open(episode_path, "w") as f:
            for step in sample_episode_data["steps"]:
                f.write(json.dumps({"type": "step", **step}) + "\n")

        # Generate report with custom filename
        custom_filename = "custom_report.txt"
        dashboard.generate_monitoring_report(
            [episode_file], output_file=custom_filename
        )

        # Check that custom report file was created
        report_path = dashboard.log_dir / custom_filename
        assert report_path.exists()

    def test_generate_monitoring_report_file_not_found(self, dashboard):
        """Test monitoring report generation with non-existent files."""
        episode_files = ["nonexistent1.jsonl", "nonexistent2.jsonl"]

        dashboard.generate_monitoring_report(episode_files)

        # Check that report file was created
        report_path = dashboard.log_dir / "monitoring_report.txt"
        assert report_path.exists()

        # Check report content
        with open(report_path, "r") as f:
            content = f.read()
            assert "ERROR: Could not load" in content

    def test_generate_monitoring_report_empty_episodes(self, dashboard):
        """Test monitoring report generation with empty episodes."""
        # Create empty episode file
        episode_file = "empty_episode.jsonl"
        episode_path = dashboard.log_dir / episode_file
        with open(episode_path, "w"):
            pass  # Empty file

        dashboard.generate_monitoring_report([episode_file])

        # Check that report file was created
        report_path = dashboard.log_dir / "monitoring_report.txt"
        assert report_path.exists()

        # Check report content
        with open(report_path, "r") as f:
            content = f.read()
            assert "empty_episode.jsonl" in content

    def test_generate_monitoring_report_statistics_calculation(self, dashboard):
        """Test that monitoring report calculates statistics correctly."""
        # Create episode data with known violations
        episode_data = {
            "steps": [
                {
                    "step": 1,
                    "regulator_flags": {
                        "parallel_violation": True,
                        "structural_break_violation": False,
                        "fines_applied": [25.0, 25.0, 25.0],
                        "market_volatility": 0.1,
                    },
                },
                {
                    "step": 2,
                    "regulator_flags": {
                        "parallel_violation": False,
                        "structural_break_violation": True,
                        "fines_applied": [50.0, 50.0, 50.0],
                        "market_volatility": 0.2,
                    },
                },
                {
                    "step": 3,
                    "regulator_flags": {
                        "parallel_violation": False,
                        "structural_break_violation": False,
                        "fines_applied": [0.0, 0.0, 0.0],
                        "market_volatility": 0.15,
                    },
                },
            ]
        }

        # Create test episode file
        episode_file = "test_episode.jsonl"
        episode_path = dashboard.log_dir / episode_file
        with open(episode_path, "w") as f:
            for step in episode_data["steps"]:
                f.write(json.dumps({"type": "step", **step}) + "\n")

        dashboard.generate_monitoring_report([episode_file])

        # Check that report file was created
        report_path = dashboard.log_dir / "monitoring_report.txt"
        assert report_path.exists()

        # Check report content
        with open(report_path, "r") as f:
            content = f.read()
            assert "Total Steps: 3" in content
            assert "Parallel Violations: 1" in content
            assert "Structural Break Violations: 1" in content
            assert "Total Violations: 2" in content
            assert "Violation Rate: 66.7%" in content
            assert "Average Market Volatility: 0.150" in content
            assert "Total Fines Applied: 225.00" in content

    def test_main_function(self):
        """Test the main function."""
        with patch(
            "src.monitoring.enhanced_dashboard.EnhancedMonitoringDashboard"
        ) as mock_dashboard_class:
            mock_dashboard = Mock()
            mock_dashboard_class.return_value = mock_dashboard

            from src.monitoring.enhanced_dashboard import main

            main()

            # Verify dashboard was created and methods were called
            mock_dashboard_class.assert_called_once()
            mock_dashboard.create_comprehensive_dashboard.assert_called_once()
            mock_dashboard.generate_monitoring_report.assert_called_once()

    def test_edge_case_missing_regulator_flags(self, dashboard):
        """Test edge case with missing regulator flags."""
        episode_data = {
            "steps": [
                {
                    "step": 1,
                    "prices": [50.0, 52.0, 48.0],
                    "profits": [1000.0, 1100.0, 900.0],
                    # Missing regulator_flags
                }
            ]
        }

        # Create test episode file
        episode_file = "test_episode.jsonl"
        episode_path = dashboard.log_dir / episode_file
        with open(episode_path, "w") as f:
            for step in episode_data["steps"]:
                f.write(json.dumps({"type": "step", **step}) + "\n")

        # Should handle missing regulator flags gracefully
        loaded_data = dashboard.load_episode_data(episode_file)
        assert len(loaded_data["steps"]) == 1

        # Test plotting with missing flags
        mock_ax = Mock()
        episode_data_dict = {episode_file: loaded_data}

        dashboard._plot_fines_over_time(mock_ax, episode_data_dict)
        dashboard._plot_market_volatility(mock_ax, episode_data_dict)
        dashboard._plot_penalty_analysis(mock_ax, episode_data_dict)

    def test_edge_case_malformed_fines_data(self, dashboard):
        """Test edge case with malformed fines data."""
        episode_data = {
            "steps": [
                {
                    "step": 1,
                    "regulator_flags": {
                        "fines_applied": "not_a_list"  # Should be a list
                    },
                }
            ]
        }

        # Create test episode file
        episode_file = "test_episode.jsonl"
        episode_path = dashboard.log_dir / episode_file
        with open(episode_path, "w") as f:
            for step in episode_data["steps"]:
                f.write(json.dumps({"type": "step", **step}) + "\n")

        # Should handle malformed data gracefully
        loaded_data = dashboard.load_episode_data(episode_file)

        # Test plotting with malformed data
        mock_ax = Mock()
        episode_data_dict = {episode_file: loaded_data}

        dashboard._plot_penalty_analysis(mock_ax, episode_data_dict)
        # Should not raise an exception
