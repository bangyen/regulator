"""
Integration tests for complete experiment workflows.

This module tests end-to-end experiment execution including:
- Full experiment pipeline from CLI to results
- Multi-agent interactions with regulator monitoring
- Logging and data persistence
- Welfare calculations and metrics
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.run_experiment import run_experiment


class TestExperimentWorkflows:
    """Test complete experiment workflows."""

    def test_basic_experiment_workflow(self) -> None:
        """Test a basic experiment from start to finish."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run a simple experiment
            results = run_experiment(
                firms=["random", "random"],
                steps=10,
                regulator_config="rule_based",
                seed=42,
                log_dir=temp_dir,
                episode_id="test_basic_workflow",
            )

            # Verify results structure
            assert "episode_id" in results
            assert "episode_data" in results
            assert "welfare_metrics" in results
            assert "log_file" in results
            assert results["episode_id"] == "test_basic_workflow"

            # Verify episode data
            episode_data = results["episode_data"]
            assert "total_steps" in episode_data
            assert "total_rewards" in episode_data
            assert episode_data["total_steps"] == 10

            # Verify log file exists
            log_file = Path(results["log_file"])
            assert log_file.exists()
            assert log_file.suffix == ".jsonl"

    def test_multi_agent_experiment_workflow(self) -> None:
        """Test experiment with different agent types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_experiment(
                firms=["random", "bestresponse", "titfortat"],
                steps=20,
                regulator_config="rule_based",
                seed=123,
                log_dir=temp_dir,
                episode_id="test_multi_agent",
            )

            # Verify 3 firms
            episode_data = results["episode_data"]
            assert len(episode_data["total_rewards"]) == 3

            # Verify welfare metrics
            welfare = results["welfare_metrics"]
            assert "consumer_surplus" in welfare
            assert "producer_surplus" in welfare
            assert "total_welfare" in welfare

    def test_regulator_monitoring_workflow(self) -> None:
        """Test experiment with regulator monitoring and penalties."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_experiment(
                firms=["titfortat", "titfortat"],  # Likely to show parallel behavior
                steps=15,
                regulator_config="rule_based",
                seed=456,
                log_dir=temp_dir,
                episode_id="test_regulator_monitoring",
            )

            # Verify regulator data
            episode_data = results["episode_data"]
            assert "total_fines" in episode_data
            assert "violations" in episode_data

            # Check that violations are tracked
            violations = episode_data["violations"]
            assert "parallel" in violations
            assert "structural_break" in violations

    def test_environment_parameter_workflow(self) -> None:
        """Test experiment with custom environment parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_params = {
                "marginal_cost": 15.0,
                "demand_intercept": 120.0,
                "demand_slope": -1.5,
                "shock_std": 3.0,
                "price_min": 5.0,
                "price_max": 80.0,
            }

            results = run_experiment(
                firms=["random", "random"],
                steps=10,
                regulator_config="rule_based",
                seed=789,
                log_dir=temp_dir,
                episode_id="test_custom_env",
                env_params=env_params,
            )

            # Verify experiment parameters are saved
            exp_params = results["experiment_params"]
            assert exp_params["env_params"]["marginal_cost"] == 15.0
            assert exp_params["env_params"]["demand_intercept"] == 120.0

    def test_log_file_structure_workflow(self) -> None:
        """Test that log files have correct structure and content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_experiment(
                firms=["random", "bestresponse"],
                steps=5,
                regulator_config="rule_based",
                seed=999,
                log_dir=temp_dir,
                episode_id="test_log_structure",
            )

            # Read and verify log file
            log_file = Path(results["log_file"])
            lines = log_file.read_text().strip().split("\n")

            # Should have header + 5 steps + possibly summary
            assert len(lines) >= 6  # At least header + 5 steps

            # Parse first line (header)
            header = json.loads(lines[0])
            assert header["type"] == "episode_header"
            assert header["episode_id"] == "test_log_structure"
            assert header["n_firms"] == 2

            # Parse step lines (skip episode_end if present)
            step_count = 0
            for line in lines[1:]:
                data = json.loads(line)
                if data["type"] == "step":
                    step_count += 1
                    assert data["step"] == step_count
                    assert "prices" in data
                    assert "profits" in data
                    assert len(data["prices"]) == 2
                elif data["type"] == "episode_end":
                    # Episode end is optional
                    pass

            # Should have 5 steps
            assert step_count == 5

    def test_reproducibility_workflow(self) -> None:
        """Test that experiments are reproducible with same seed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run same experiment twice
            results1 = run_experiment(
                firms=["random", "random"],
                steps=10,
                regulator_config="rule_based",
                seed=42,
                log_dir=temp_dir,
                episode_id="test_repro_1",
            )

            results2 = run_experiment(
                firms=["random", "random"],
                steps=10,
                regulator_config="rule_based",
                seed=42,
                log_dir=temp_dir,
                episode_id="test_repro_2",
            )

            # Compare episode data (should be identical)
            ep1 = results1["episode_data"]
            ep2 = results2["episode_data"]

            np.testing.assert_array_equal(ep1["total_rewards"], ep2["total_rewards"])
            np.testing.assert_array_equal(ep1["episode_prices"], ep2["episode_prices"])

    def test_error_handling_workflow(self) -> None:
        """Test error handling in experiment workflows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test invalid agent type
            with pytest.raises(ValueError, match="Unknown agent type"):
                run_experiment(
                    firms=["invalid_agent"],
                    steps=10,
                    regulator_config="rule_based",
                    seed=42,
                    log_dir=temp_dir,
                )

            # Test invalid regulator config
            with pytest.raises(ValueError, match="Unknown regulator config"):
                run_experiment(
                    firms=["random"],
                    steps=10,
                    regulator_config="invalid_regulator",
                    seed=42,
                    log_dir=temp_dir,
                )

    def test_welfare_calculation_workflow(self) -> None:
        """Test welfare calculations in experiment results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_experiment(
                firms=["random", "random"],
                steps=10,
                regulator_config="rule_based",
                seed=42,
                log_dir=temp_dir,
                episode_id="test_welfare",
            )

            welfare = results["welfare_metrics"]

            # Verify welfare metrics exist and are reasonable
            assert isinstance(welfare["consumer_surplus"], (int, float))
            assert isinstance(welfare["producer_surplus"], (int, float))
            assert isinstance(welfare["total_welfare"], (int, float))

            # Total welfare should equal sum of components
            expected_total = welfare["consumer_surplus"] + welfare["producer_surplus"]
            assert abs(welfare["total_welfare"] - expected_total) < 1e-6
