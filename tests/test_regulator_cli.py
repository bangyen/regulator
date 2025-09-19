"""
Tests for the regulator CLI module.

This module tests the CLI functionality including command execution,
argument parsing, and error handling.
"""

import subprocess
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

# Import from the package
from regulator_cli import main, experiment, train, episode, dashboard


class TestRegulatorCLI:
    """Test the regulator CLI commands."""

    def test_main_command_group(self) -> None:
        """Test that the main command group is properly configured."""
        # Test that main is a click group
        assert hasattr(main, "commands")
        assert hasattr(main, "invoke")

    def test_experiment_command_basic(self) -> None:
        """Test basic experiment command execution."""
        with patch("regulator_cli.run_experiment") as mock_run:
            mock_run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                experiment,
                [
                    "--steps",
                    "5",
                    "--firms",
                    "random,tit_for_tat",
                    "--regulator",
                    "rule_based",
                    "--seed",
                    "42",
                    "--log-dir",
                    "test_logs",
                ],
            )

            # Verify the function was called and command succeeded
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_experiment_command_with_exception(self) -> None:
        """Test experiment command with exception handling."""
        with patch("regulator_cli.run_experiment") as mock_run:
            mock_run.side_effect = Exception("Test error")

            runner = CliRunner()
            result = runner.invoke(
                experiment,
                [
                    "--steps",
                    "5",
                    "--firms",
                    "random",
                    "--regulator",
                    "rule_based",
                    "--seed",
                    "42",
                    "--log-dir",
                    "test_logs",
                ],
            )

            # Should exit with error code
            assert result.exit_code == 1

    def test_train_command_basic(self) -> None:
        """Test basic train command execution."""
        with patch("regulator_cli.train_ml_detector") as mock_train:
            mock_train.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                train,
                [
                    "--n-episodes",
                    "10",
                    "--model-type",
                    "logistic",
                    "--output-dir",
                    "test_output",
                ],
            )

            # Verify the function was called and command succeeded
            assert result.exit_code == 0
            mock_train.assert_called_once()

    def test_train_command_with_existing_logs(self) -> None:
        """Test train command with existing logs parameter."""
        with patch("regulator_cli.train_ml_detector") as mock_train:
            mock_train.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                train,
                [
                    "--n-episodes",
                    "10",
                    "--model-type",
                    "lightgbm",
                    "--existing-logs",
                    "/path/to/logs",
                    "--output-dir",
                    "test_output",
                ],
            )

            # Verify the function was called and command succeeded
            assert result.exit_code == 0
            mock_train.assert_called_once()

    def test_train_command_with_exception(self) -> None:
        """Test train command with exception handling."""
        with patch("regulator_cli.train_ml_detector") as mock_train:
            mock_train.side_effect = Exception("Training failed")

            runner = CliRunner()
            result = runner.invoke(
                train,
                [
                    "--n-episodes",
                    "10",
                    "--model-type",
                    "random_forest",
                    "--output-dir",
                    "test_output",
                ],
            )

            # Should exit with error code
            assert result.exit_code == 1

    def test_episode_command_basic(self) -> None:
        """Test basic episode command execution."""
        with patch("regulator_cli.run_episode") as mock_run:
            mock_run.return_value = {"episode_id": "test_123"}

            runner = CliRunner()
            result = runner.invoke(
                episode,
                [
                    "--firms",
                    "random,tit_for_tat",
                    "--steps",
                    "10",
                    "--seed",
                    "42",
                    "--log-dir",
                    "test_logs",
                ],
            )

            # Verify the function was called and command succeeded
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_episode_command_with_n_firms(self) -> None:
        """Test episode command with specified number of firms."""
        with patch("regulator_cli.run_episode") as mock_run:
            mock_run.return_value = {"episode_id": "test_456"}

            runner = CliRunner()
            result = runner.invoke(
                episode,
                [
                    "--firms",
                    "random,random,random",
                    "--steps",
                    "5",
                    "--n-firms",
                    "3",
                    "--seed",
                    "123",
                    "--log-dir",
                    "test_logs",
                ],
            )

            # Verify the function was called and command succeeded
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_episode_command_with_exception(self) -> None:
        """Test episode command with exception handling."""
        with patch("regulator_cli.run_episode") as mock_run:
            mock_run.side_effect = Exception("Episode failed")

            runner = CliRunner()
            result = runner.invoke(
                episode,
                [
                    "--firms",
                    "random",
                    "--steps",
                    "10",
                    "--seed",
                    "42",
                    "--log-dir",
                    "test_logs",
                ],
            )

            # Should exit with error code
            assert result.exit_code == 1

    def test_dashboard_command_basic(self) -> None:
        """Test basic dashboard command execution."""
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = Mock(returncode=0)

            runner = CliRunner()
            result = runner.invoke(dashboard, ["--port", "8501"])

            # Verify the command succeeded and subprocess was called
            assert result.exit_code == 0
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert "streamlit" in call_args
            assert "run" in call_args
            assert "dashboard/app.py" in call_args
            assert "--server.port" in call_args
            assert "8501" in call_args

    def test_dashboard_command_custom_port(self) -> None:
        """Test dashboard command with custom port."""
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = Mock(returncode=0)

            runner = CliRunner()
            result = runner.invoke(dashboard, ["--port", "8080"])

            # Verify the command succeeded and subprocess was called with correct port
            assert result.exit_code == 0
            call_args = mock_subprocess.call_args[0][0]
            assert "8080" in call_args

    def test_dashboard_command_subprocess_error(self) -> None:
        """Test dashboard command with subprocess error."""
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = subprocess.CalledProcessError(1, "streamlit")

            runner = CliRunner()
            result = runner.invoke(dashboard, ["--port", "8501"])

            # Should exit with error code
            assert result.exit_code == 1

    def test_dashboard_command_keyboard_interrupt(self) -> None:
        """Test dashboard command with keyboard interrupt."""
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = KeyboardInterrupt()

            runner = CliRunner()
            result = runner.invoke(dashboard, ["--port", "8501"])

            # Should exit successfully for keyboard interrupt (graceful shutdown)
            assert result.exit_code == 0
            assert "Dashboard stopped" in result.output


class TestRegulatorCLIIntegration:
    """Integration tests for the regulator CLI."""

    def test_cli_help_commands(self) -> None:
        """Test that CLI help commands work."""
        runner = CliRunner()

        # Test main help
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Regulator: Market Competition" in result.output

        # Test experiment help
        result = runner.invoke(experiment, ["--help"])
        assert result.exit_code == 0
        assert "--steps" in result.output

        # Test train help
        result = runner.invoke(train, ["--help"])
        assert result.exit_code == 0
        assert "--model-type" in result.output

        # Test episode help
        result = runner.invoke(episode, ["--help"])
        assert result.exit_code == 0
        assert "--firms" in result.output

        # Test dashboard help
        result = runner.invoke(dashboard, ["--help"])
        assert result.exit_code == 0
        assert "--port" in result.output

    def test_cli_version_option(self) -> None:
        """Test that CLI version option works."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


if __name__ == "__main__":
    pytest.main([__file__])
