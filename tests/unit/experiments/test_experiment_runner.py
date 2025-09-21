"""
Unit tests for experiment runner functionality.

This module tests the core experiment execution functions including
agent creation, regulator creation, welfare calculations, and experiment execution.
"""

import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.experiments.experiment_runner import (
    calculate_welfare_metrics,
    create_agent,
    create_regulator,
    print_experiment_summary,
    run_experiment,
)

# Import the actual classes that experiment_runner uses
from agents.firm_agents import BaseAgent
from agents.regulator import Regulator
from cartel.cartel_env import CartelEnv


class TestCreateAgent:
    """Test agent creation functionality."""

    def test_create_random_agent(self) -> None:
        """Test creating a random agent."""
        agent = create_agent("random", agent_id=0, seed=42)
        assert isinstance(agent, BaseAgent)
        assert agent.agent_id == 0

    def test_create_bestresponse_agent(self) -> None:
        """Test creating a best response agent."""
        agent = create_agent("bestresponse", agent_id=1, seed=42)
        assert isinstance(agent, BaseAgent)
        assert agent.agent_id == 1

    def test_create_titfortat_agent(self) -> None:
        """Test creating a tit-for-tat agent."""
        agent = create_agent("titfortat", agent_id=2, seed=42)
        assert isinstance(agent, BaseAgent)
        assert agent.agent_id == 2

    def test_create_agent_case_insensitive(self) -> None:
        """Test that agent type is case insensitive."""
        agent1 = create_agent("RANDOM", agent_id=0, seed=42)
        agent2 = create_agent("Random", agent_id=1, seed=42)
        agent3 = create_agent("random", agent_id=2, seed=42)

        assert isinstance(agent1, BaseAgent)
        assert isinstance(agent2, BaseAgent)
        assert isinstance(agent3, BaseAgent)

    def test_create_agent_with_seed(self) -> None:
        """Test creating agents with different seeds."""
        agent1 = create_agent("random", agent_id=0, seed=42)
        agent2 = create_agent("random", agent_id=1, seed=123)

        assert agent1.agent_id == 0
        assert agent2.agent_id == 1
        # Agents should be different instances
        assert agent1 is not agent2

    def test_create_agent_invalid_type(self) -> None:
        """Test creating agent with invalid type raises error."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("invalid_type", agent_id=0, seed=42)

    def test_create_agent_empty_type(self) -> None:
        """Test creating agent with empty type raises error."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("", agent_id=0, seed=42)


class TestCreateRegulator:
    """Test regulator creation functionality."""

    def test_create_regulator_rule_based(self) -> None:
        """Test creating a rule-based regulator."""
        regulator = create_regulator("rule_based", seed=42)
        assert isinstance(regulator, Regulator)

    def test_create_regulator_ml(self) -> None:
        """Test creating an ML regulator."""
        regulator = create_regulator("ml", seed=42)
        assert isinstance(regulator, Regulator)

    def test_create_regulator_none(self) -> None:
        """Test creating a 'none' regulator (dummy)."""
        regulator = create_regulator("none", seed=42)
        assert isinstance(regulator, Regulator)

    def test_create_regulator_with_seed(self) -> None:
        """Test creating regulators with different seeds."""
        regulator1 = create_regulator("rule_based", seed=42)
        regulator2 = create_regulator("rule_based", seed=123)

        assert isinstance(regulator1, Regulator)
        assert isinstance(regulator2, Regulator)
        # Should be different instances
        assert regulator1 is not regulator2

    def test_create_regulator_invalid_config(self) -> None:
        """Test creating regulator with invalid config."""
        # Note: The current implementation doesn't validate config,
        # so this test documents current behavior
        regulator = create_regulator("invalid_config", seed=42)
        assert isinstance(regulator, Regulator)


class TestCalculateWelfareMetrics:
    """Test welfare metrics calculation functionality."""

    def test_calculate_welfare_metrics_empty_data(self) -> None:
        """Test welfare calculation with empty episode data."""
        env = CartelEnv(n_firms=2, max_steps=10, seed=42)
        episode_data = []

        metrics = calculate_welfare_metrics(episode_data, env)

        assert metrics["consumer_surplus"] == 0.0
        assert metrics["producer_surplus"] == 0.0
        assert metrics["total_welfare"] == 0.0
        assert metrics["deadweight_loss"] == 0.0

    def test_calculate_welfare_metrics_single_step(self) -> None:
        """Test welfare calculation with single step data."""
        env = CartelEnv(n_firms=2, max_steps=10, seed=42)
        episode_data = [
            {
                "market_price": 20.0,
                "total_quantity": 80.0,
                "profits": [100.0, 120.0],
            }
        ]

        metrics = calculate_welfare_metrics(episode_data, env)

        # Consumer surplus = 0.5 * (demand_intercept - market_price) * total_quantity
        # = 0.5 * (100 - 20) * 80 = 0.5 * 80 * 80 = 3200
        expected_consumer_surplus = 0.5 * (env.demand_intercept - 20.0) * 80.0
        assert metrics["consumer_surplus"] == expected_consumer_surplus

        # Producer surplus = total profits = 100 + 120 = 220
        assert metrics["producer_surplus"] == 220.0

        # Total welfare = consumer + producer surplus
        expected_total = expected_consumer_surplus + 220.0
        assert metrics["total_welfare"] == expected_total

        # Deadweight loss should be calculated
        assert metrics["deadweight_loss"] >= 0.0

    def test_calculate_welfare_metrics_multiple_steps(self) -> None:
        """Test welfare calculation with multiple step data."""
        env = CartelEnv(n_firms=2, max_steps=10, seed=42)
        episode_data = [
            {
                "market_price": 20.0,
                "total_quantity": 80.0,
                "profits": [100.0, 120.0],
            },
            {
                "market_price": 25.0,
                "total_quantity": 75.0,
                "profits": [90.0, 110.0],
            },
        ]

        metrics = calculate_welfare_metrics(episode_data, env)

        # Should sum across all steps
        assert metrics["consumer_surplus"] > 0.0
        assert metrics["producer_surplus"] == 420.0  # 100+120+90+110
        assert metrics["total_welfare"] > 0.0
        assert metrics["deadweight_loss"] >= 0.0

    def test_calculate_welfare_metrics_missing_data(self) -> None:
        """Test welfare calculation with missing data fields."""
        env = CartelEnv(n_firms=2, max_steps=10, seed=42)
        episode_data = [
            {
                "market_price": 20.0,
                # Missing total_quantity and profits
            }
        ]

        metrics = calculate_welfare_metrics(episode_data, env)

        # Should handle missing data gracefully
        assert metrics["consumer_surplus"] >= 0.0
        assert metrics["producer_surplus"] == 0.0  # No profits data
        assert metrics["total_welfare"] >= 0.0
        assert metrics["deadweight_loss"] >= 0.0

    def test_calculate_welfare_metrics_negative_consumer_surplus(self) -> None:
        """Test welfare calculation when consumer surplus would be negative."""
        env = CartelEnv(n_firms=2, max_steps=10, seed=42)
        episode_data = [
            {
                "market_price": 150.0,  # Higher than demand_intercept (100)
                "total_quantity": 80.0,
                "profits": [100.0, 120.0],
            }
        ]

        metrics = calculate_welfare_metrics(episode_data, env)

        # Consumer surplus should be 0 (max with 0)
        assert metrics["consumer_surplus"] == 0.0
        assert metrics["producer_surplus"] == 220.0
        assert metrics["total_welfare"] == 220.0


class TestPrintExperimentSummary:
    """Test experiment summary printing functionality."""

    @patch("builtins.print")
    def test_print_experiment_summary_basic(self, mock_print: Mock) -> None:
        """Test printing basic experiment summary."""
        results = {
            "episode_id": "test_episode",
            "log_file": "/path/to/log.jsonl",
        }
        episode_data = [
            {
                "prices": [20.0, 25.0],
                "profits": [100.0, 120.0],
                "market_price": 22.5,
            }
        ]
        welfare_metrics = {
            "consumer_surplus": 1000.0,
            "producer_surplus": 220.0,
            "total_welfare": 1220.0,
            "deadweight_loss": 50.0,
        }

        print_experiment_summary(results, episode_data, welfare_metrics)

        # Verify that print was called multiple times
        assert mock_print.call_count > 0

        # Check that key information is printed
        printed_text = " ".join(str(call) for call in mock_print.call_args_list)
        assert "test_episode" in printed_text
        assert "Results saved to:" in printed_text

    @patch("builtins.print")
    def test_print_experiment_summary_empty_data(self, mock_print: Mock) -> None:
        """Test printing summary with empty episode data."""
        results = {"episode_id": "empty_episode"}
        episode_data = []
        welfare_metrics = {
            "consumer_surplus": 0.0,
            "producer_surplus": 0.0,
            "total_welfare": 0.0,
            "deadweight_loss": 0.0,
        }

        print_experiment_summary(results, episode_data, welfare_metrics)

        # Should still print without errors
        assert mock_print.call_count > 0


class TestRunExperiment:
    """Test main experiment execution functionality."""

    @patch("src.experiments.experiment_runner.run_episode_with_regulator_logging")
    def test_run_experiment_basic(self, mock_run_episode: Mock) -> None:
        """Test running a basic experiment."""
        # Mock the episode runner
        mock_logger = Mock()
        mock_logger.get_log_file_path.return_value = "/path/to/log.jsonl"
        mock_logger.load_episode_data.return_value = {
            "steps": [
                {
                    "prices": [20.0, 25.0],
                    "profits": [100.0, 120.0],
                    "market_price": 22.5,
                    "total_quantity": 77.5,
                }
            ]
        }
        mock_run_episode.return_value = {
            "episode_data": [
                {
                    "prices": [20.0, 25.0],
                    "profits": [100.0, 120.0],
                    "market_price": 22.5,
                    "total_quantity": 77.5,
                }
            ],
            "log_file": "/path/to/log.jsonl",
            "logger": mock_logger,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_experiment(
                firms=["random", "random"],
                steps=10,
                regulator_config="rule_based",
                seed=42,
                log_dir=temp_dir,
                episode_id="test_experiment",
            )

        # Verify results structure
        assert "episode_id" in results
        assert "episode_data" in results
        assert "experiment_params" in results
        assert "welfare_metrics" in results
        assert "log_file" in results

        # Verify experiment parameters
        assert results["episode_id"] == "test_experiment"
        exp_params = results["experiment_params"]
        assert exp_params["firms"] == ["random", "random"]
        assert exp_params["steps"] == 10
        assert exp_params["regulator_config"] == "rule_based"
        assert exp_params["seed"] == 42

    @patch("src.experiments.experiment_runner.run_episode_with_regulator_logging")
    def test_run_experiment_with_env_params(self, mock_run_episode: Mock) -> None:
        """Test running experiment with custom environment parameters."""
        mock_logger = Mock()
        mock_logger.get_log_file_path.return_value = "/path/to/log.jsonl"
        mock_logger.load_episode_data.return_value = {"steps": []}
        mock_run_episode.return_value = {
            "episode_data": [],
            "log_file": "/path/to/log.jsonl",
            "logger": mock_logger,
        }

        env_params = {
            "marginal_cost": 15.0,
            "demand_intercept": 120.0,
            "demand_slope": -1.5,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_experiment(
                firms=["random"],
                steps=5,
                regulator_config="rule_based",
                seed=42,
                log_dir=temp_dir,
                env_params=env_params,
            )

        # Verify environment parameters are saved
        exp_params = results["experiment_params"]
        assert exp_params["env_params"]["marginal_cost"] == 15.0
        assert exp_params["env_params"]["demand_intercept"] == 120.0
        assert exp_params["env_params"]["demand_slope"] == -1.5

    @patch("src.experiments.experiment_runner.run_episode_with_regulator_logging")
    def test_run_experiment_auto_episode_id(self, mock_run_episode: Mock) -> None:
        """Test running experiment with auto-generated episode ID."""
        mock_logger = Mock()
        mock_logger.get_log_file_path.return_value = "/path/to/log.jsonl"
        mock_logger.load_episode_data.return_value = {"steps": []}
        mock_run_episode.return_value = {
            "episode_data": [],
            "log_file": "/path/to/log.jsonl",
            "logger": mock_logger,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_experiment(
                firms=["random"],
                steps=5,
                regulator_config="rule_based",
                seed=42,
                log_dir=temp_dir,
                # No episode_id provided
            )

        # Should have auto-generated episode ID
        assert "episode_id" in results
        assert results["episode_id"].startswith("experiment_")
        assert len(results["episode_id"]) > 20  # Should include timestamp

    @patch("src.experiments.experiment_runner.run_episode_with_regulator_logging")
    def test_run_experiment_numpy_type_conversion(self, mock_run_episode: Mock) -> None:
        """Test that numpy types are converted to Python native types."""
        mock_logger = Mock()
        mock_logger.get_log_file_path.return_value = "/path/to/log.jsonl"
        mock_logger.load_episode_data.return_value = {
            "steps": [
                {
                    "prices": np.array([20.0, 25.0]),
                    "profits": np.array([100.0, 120.0]),
                    "market_price": np.float32(22.5),
                    "total_quantity": np.int32(77),
                }
            ]
        }
        mock_run_episode.return_value = {
            "episode_data": [
                {
                    "prices": np.array([20.0, 25.0]),
                    "profits": np.array([100.0, 120.0]),
                    "market_price": np.float32(22.5),
                    "total_quantity": np.int32(77),
                }
            ],
            "log_file": "/path/to/log.jsonl",
            "logger": mock_logger,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_experiment(
                firms=["random"],
                steps=5,
                regulator_config="rule_based",
                seed=42,
                log_dir=temp_dir,
            )

        # Verify numpy types are converted
        episode_data = results["episode_data"][0]
        assert isinstance(episode_data["prices"], list)
        assert isinstance(episode_data["profits"], list)
        assert isinstance(episode_data["market_price"], float)
        assert isinstance(episode_data["total_quantity"], int)

    @patch("src.experiments.experiment_runner.run_episode_with_regulator_logging")
    def test_run_experiment_different_agent_types(self, mock_run_episode: Mock) -> None:
        """Test running experiment with different agent types."""
        mock_logger = Mock()
        mock_logger.get_log_file_path.return_value = "/path/to/log.jsonl"
        mock_logger.load_episode_data.return_value = {"steps": []}
        mock_run_episode.return_value = {
            "episode_data": [],
            "log_file": "/path/to/log.jsonl",
            "logger": mock_logger,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_experiment(
                firms=["random", "bestresponse", "titfortat"],
                steps=5,
                regulator_config="rule_based",
                seed=42,
                log_dir=temp_dir,
            )

        # Verify all agent types are recorded
        exp_params = results["experiment_params"]
        assert exp_params["firms"] == ["random", "bestresponse", "titfortat"]
        assert exp_params["env_params"]["n_firms"] == 3

    @patch("src.experiments.experiment_runner.run_episode_with_regulator_logging")
    def test_run_experiment_different_regulator_configs(
        self, mock_run_episode: Mock
    ) -> None:
        """Test running experiment with different regulator configurations."""
        mock_logger = Mock()
        mock_logger.get_log_file_path.return_value = "/path/to/log.jsonl"
        mock_logger.load_episode_data.return_value = {"steps": []}
        mock_run_episode.return_value = {
            "episode_data": [],
            "log_file": "/path/to/log.jsonl",
            "logger": mock_logger,
        }

        configs = ["rule_based", "ml", "none"]

        for config in configs:
            with tempfile.TemporaryDirectory() as temp_dir:
                results = run_experiment(
                    firms=["random"],
                    steps=5,
                    regulator_config=config,
                    seed=42,
                    log_dir=temp_dir,
                )

            # Verify regulator config is recorded
            exp_params = results["experiment_params"]
            assert exp_params["regulator_config"] == config

    @patch("builtins.print")
    @patch("src.experiments.experiment_runner.run_episode_with_regulator_logging")
    def test_run_experiment_prints_progress(
        self, mock_run_episode: Mock, mock_print: Mock
    ) -> None:
        """Test that experiment prints progress information."""
        mock_logger = Mock()
        mock_logger.get_log_file_path.return_value = "/path/to/log.jsonl"
        mock_logger.load_episode_data.return_value = {"steps": []}
        mock_run_episode.return_value = {
            "episode_data": [],
            "log_file": "/path/to/log.jsonl",
            "logger": mock_logger,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            run_experiment(
                firms=["random"],
                steps=5,
                regulator_config="rule_based",
                seed=42,
                log_dir=temp_dir,
                episode_id="test_print",
            )

        # Verify progress information is printed
        assert mock_print.call_count > 0
        printed_text = " ".join(str(call) for call in mock_print.call_args_list)
        assert "test_print" in printed_text
        assert "random" in printed_text
        assert "rule_based" in printed_text
