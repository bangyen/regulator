"""
Tests for the regulator CLI module.

This module tests the CLI functionality including command execution,
argument parsing, and error handling.
"""

from unittest.mock import Mock, patch

from click.testing import CliRunner

# Import from the package
from regulator_cli import main, experiment, dashboard


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
                    "10",
                    "--firms",
                    "random,tit_for_tat",
                    "--seed",
                    "42",
                ],
            )

            # Verify the function was called and command succeeded
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_experiment_command_with_custom_options(self) -> None:
        """Test experiment command with custom options."""
        with patch("regulator_cli.run_experiment") as mock_run:
            mock_run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                experiment,
                [
                    "--steps",
                    "50",
                    "--firms",
                    "random,competitive",
                    "--regulator",
                    "rule_based",
                    "--seed",
                    "123",
                    "--log-dir",
                    "custom_logs",
                    "--episode-id",
                    "test_episode",
                ],
            )

            # Verify the function was called and command succeeded
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_experiment_command_with_exception(self) -> None:
        """Test experiment command with exception handling."""
        with patch("regulator_cli.run_experiment") as mock_run:
            mock_run.side_effect = Exception("Experiment failed")

            runner = CliRunner()
            result = runner.invoke(
                experiment,
                [
                    "--steps",
                    "10",
                    "--firms",
                    "random",
                    "--seed",
                    "42",
                ],
            )

            # Should exit with error code
            assert result.exit_code == 1

    def test_dashboard_command_basic(self) -> None:
        """Test basic dashboard command execution."""
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = Mock()

            runner = CliRunner()
            result = runner.invoke(dashboard, ["--port", "8501"])

            # Verify subprocess was called
            assert result.exit_code == 0
            mock_subprocess.assert_called_once()

    def test_dashboard_command_with_custom_port(self) -> None:
        """Test dashboard command with custom port."""
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = Mock()

            runner = CliRunner()
            result = runner.invoke(dashboard, ["--port", "8502"])

            # Verify subprocess was called with correct port
            assert result.exit_code == 0
            mock_subprocess.assert_called_once()

    def test_help_commands(self) -> None:
        """Test that help commands work properly."""
        runner = CliRunner()

        # Test main help
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Regulator: Market Competition & Collusion Detection" in result.output

        # Test experiment help
        result = runner.invoke(experiment, ["--help"])
        assert result.exit_code == 0
        assert "--steps" in result.output

        # Test dashboard help
        result = runner.invoke(dashboard, ["--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
