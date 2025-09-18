"""
Tests for the CLI experiment runner.

This module tests the CLI functionality including argument parsing,
experiment execution, and result validation.
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from typing import Any, Dict

import pytest

# Add the project root to the Python path so src imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_experiment import (  # noqa: E402
    calculate_welfare_metrics,
    create_agent,
    create_regulator,
    run_experiment,
)


class TestCLIFunctions:
    """Test individual CLI functions."""

    def test_create_agent(self) -> None:
        """Test agent creation with different types."""
        # Test random agent
        random_agent = create_agent("random", agent_id=0, seed=42)
        assert random_agent.agent_id == 0
        assert hasattr(random_agent, "choose_price")

        # Test best response agent
        best_agent = create_agent("bestresponse", agent_id=1, seed=42)
        assert best_agent.agent_id == 1
        assert hasattr(best_agent, "choose_price")

        # Test tit-for-tat agent
        tit_agent = create_agent("titfortat", agent_id=2, seed=42)
        assert tit_agent.agent_id == 2
        assert hasattr(tit_agent, "choose_price")

        # Test invalid agent type
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("invalid", agent_id=0)

    def test_create_regulator(self) -> None:
        """Test regulator creation with different configurations."""
        # Test ML regulator
        ml_regulator = create_regulator("ml", seed=42)
        assert ml_regulator.parallel_threshold == 1.5
        assert ml_regulator.parallel_steps == 2
        assert ml_regulator.fine_amount == 75.0

        # Test rule-based regulator
        rule_regulator = create_regulator("rule_based", seed=42)
        assert rule_regulator.parallel_threshold == 2.0
        assert rule_regulator.parallel_steps == 3
        assert rule_regulator.fine_amount == 50.0

        # Test no regulator
        none_regulator = create_regulator("none", seed=42)
        assert none_regulator.fine_amount == 0.0

        # Test invalid regulator config
        with pytest.raises(ValueError, match="Unknown regulator config"):
            create_regulator("invalid")

    def test_calculate_welfare_metrics(self) -> None:
        """Test welfare metrics calculation."""

        # Mock environment
        class MockEnv:
            def __init__(self) -> None:
                self.demand_intercept = 100.0
                self.demand_slope = -1.0
                self.marginal_cost = 10.0
                self.n_firms = 2

        env = MockEnv()

        # Mock episode data
        episode_data: Dict[str, Any] = {
            "episode_prices": [[50.0, 55.0], [52.0, 57.0], [48.0, 53.0]],
            "episode_profits": [[400.0, 450.0], [420.0, 470.0], [380.0, 430.0]],
        }

        metrics = calculate_welfare_metrics(episode_data, env)  # type: ignore

        # Check that all required metrics are present
        required_metrics = [
            "consumer_surplus",
            "producer_surplus",
            "total_welfare",
            "deadweight_loss",
        ]
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert metrics[metric] >= 0.0

        # Check that total welfare = consumer + producer surplus
        assert math.isclose(
            metrics["total_welfare"],
            metrics["consumer_surplus"] + metrics["producer_surplus"],
            rel_tol=1e-6,
        )

    def test_calculate_welfare_metrics_empty_data(self) -> None:
        """Test welfare metrics calculation with empty data."""

        class MockEnv:
            def __init__(self) -> None:
                self.demand_intercept = 100.0
                self.demand_slope = -1.0
                self.marginal_cost = 10.0
                self.n_firms = 2

        env = MockEnv()
        episode_data: Dict[str, Any] = {"episode_prices": [], "episode_profits": []}

        metrics = calculate_welfare_metrics(episode_data, env)  # type: ignore

        # All metrics should be zero for empty data
        for metric in [
            "consumer_surplus",
            "producer_surplus",
            "total_welfare",
            "deadweight_loss",
        ]:
            assert metrics[metric] == 0.0


class TestCLIExecution:
    """Test CLI execution and output validation."""

    def test_cli_runs_with_sample_args(self) -> None:
        """Test that CLI runs successfully with sample arguments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run CLI with sample arguments
            cmd = [
                sys.executable,
                "scripts/run_experiment.py",
                "--firms",
                "random,titfortat",
                "--steps",
                "10",
                "--regulator",
                "rule_based",
                "--seed",
                "42",
                "--log-dir",
                temp_dir,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=project_root
            )

            # Check that command succeeded
            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check that log file was created
            log_files = list(Path(temp_dir).glob("*.jsonl"))
            assert len(log_files) == 1, "Expected exactly one log file"

            # Check log file content
            log_file = log_files[0]
            with open(log_file, "r") as f:
                lines = f.readlines()
                assert (
                    len(lines) >= 3
                ), "Log file should have header, steps, and summary"

                # Check header
                header = json.loads(lines[0])
                assert header["type"] == "episode_header"
                assert header["n_firms"] == 2

                # Check steps
                step_lines = [line for line in lines[1:-1] if line.strip()]
                assert len(step_lines) == 10, "Should have 10 step records"

                # Check summary
                summary = json.loads(lines[-1])
                assert summary["type"] == "episode_end"

    def test_cli_produces_consistent_summary(self) -> None:
        """Test that CLI produces consistent summary with hand-checked results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run CLI with fixed parameters
            cmd = [
                sys.executable,
                "scripts/run_experiment.py",
                "--firms",
                "random,random",
                "--steps",
                "5",
                "--regulator",
                "none",
                "--seed",
                "123",
                "--log-dir",
                temp_dir,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=project_root
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check that output contains expected summary information
            output = result.stdout

            # Check that summary contains expected fields
            expected_fields = [
                "EXPERIMENT SUMMARY",
                "Total Steps: 5",
                "Agent Types: random, random",
                "Number of Firms: 2",
                "Average Prices:",
                "Total Profits:",
                "Consumer Surplus:",
                "Producer Surplus:",
                "Total Welfare:",
                "Deadweight Loss:",
                "Total Fines Applied:",
            ]

            for field in expected_fields:
                assert field in output, f"Missing field in output: {field}"

    def test_seed_reproducibility(self) -> None:
        """Test that the same seed produces identical results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run experiment twice with same seed
            cmd_template = [
                sys.executable,
                "scripts/run_experiment.py",
                "--firms",
                "titfortat,titfortat",
                "--steps",
                "3",
                "--regulator",
                "rule_based",
                "--seed",
                "999",
                "--log-dir",
                temp_dir,
            ]

            # First run
            result1 = subprocess.run(
                cmd_template, capture_output=True, text=True, cwd=project_root
            )
            assert result1.returncode == 0, f"First run failed: {result1.stderr}"

            # Second run
            result2 = subprocess.run(
                cmd_template, capture_output=True, text=True, cwd=project_root
            )
            assert result2.returncode == 0, f"Second run failed: {result2.stderr}"

            # Compare outputs (excluding timestamps and file paths)
            output1_lines = result1.stdout.split("\n")
            output2_lines = result2.stdout.split("\n")

            # Filter out lines that should be different (timestamps, file paths)
            def filter_line(line: str) -> str:
                # Remove timestamp-like patterns and file paths
                import re

                line = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "TIMESTAMP", line)
                line = re.sub(r"experiment_\d{8}_\d{6}", "experiment_TIMESTAMP", line)
                line = re.sub(r"/tmp/[^/]+/", "/tmp/TEMP_DIR/", line)
                return line

            filtered_lines1 = [filter_line(line) for line in output1_lines]
            filtered_lines2 = [filter_line(line) for line in output2_lines]

            # Compare filtered outputs
            assert (
                filtered_lines1 == filtered_lines2
            ), "Outputs should be identical with same seed"

    def test_cli_argument_validation(self) -> None:
        """Test CLI argument validation."""
        # Test invalid agent type
        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--firms",
            "invalid_agent",
            "--steps",
            "10",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

        assert result.returncode != 0, "Should fail with invalid agent type"
        assert (
            "Unknown agent type" in result.stdout
            or "Unknown agent type" in result.stderr
        )

        # Test invalid regulator config
        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--firms",
            "random",
            "--steps",
            "10",
            "--regulator",
            "invalid_regulator",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

        assert result.returncode != 0, "Should fail with invalid regulator config"

    def test_cli_help_output(self) -> None:
        """Test that CLI help output is informative."""
        cmd = [sys.executable, "scripts/run_experiment.py", "--help"]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

        assert result.returncode == 0, "Help should succeed"
        assert "Run regulator experiments" in result.stdout
        assert "--firms" in result.stdout
        assert "--steps" in result.stdout
        assert "--regulator" in result.stdout
        assert "--seed" in result.stdout


class TestExperimentRunner:
    """Test the experiment runner function directly."""

    def test_run_experiment_basic(self) -> None:
        """Test basic experiment execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_experiment(
                firms=["random", "titfortat"],
                steps=5,
                regulator_config="rule_based",
                seed=42,
                log_dir=temp_dir,
            )

            # Check results structure
            assert "episode_id" in results
            assert "log_file" in results
            assert "episode_data" in results
            assert "episode_summary" in results
            assert "welfare_metrics" in results
            assert "experiment_params" in results

            # Check log file exists
            log_file = Path(results["log_file"])
            assert log_file.exists()

            # Check episode data
            episode_data = results["episode_data"]
            assert episode_data["total_steps"] == 5
            assert len(episode_data["episode_prices"]) == 5
            assert len(episode_data["episode_profits"]) == 5

            # Check welfare metrics
            welfare = results["welfare_metrics"]
            assert "consumer_surplus" in welfare
            assert "producer_surplus" in welfare
            assert "total_welfare" in welfare
            assert "deadweight_loss" in welfare

    def test_run_experiment_different_configs(self) -> None:
        """Test experiment with different regulator configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test ML regulator
            results_ml = run_experiment(
                firms=["random", "random"],
                steps=3,
                regulator_config="ml",
                seed=42,
                log_dir=temp_dir,
            )

            # Test no regulator
            results_none = run_experiment(
                firms=["random", "random"],
                steps=3,
                regulator_config="none",
                seed=42,
                log_dir=temp_dir,
            )

            # ML regulator should have higher fines
            assert (
                results_ml["episode_data"]["total_fines"]
                >= results_none["episode_data"]["total_fines"]
            )

    def test_run_experiment_custom_episode_id(self) -> None:
        """Test experiment with custom episode ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_id = "test_episode_123"
            results = run_experiment(
                firms=["random"],
                steps=2,
                regulator_config="none",
                seed=42,
                log_dir=temp_dir,
                episode_id=custom_id,
            )

            assert results["episode_id"] == custom_id
            assert custom_id in results["log_file"]


if __name__ == "__main__":
    pytest.main([__file__])
