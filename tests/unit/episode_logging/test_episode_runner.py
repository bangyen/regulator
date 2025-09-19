"""
Final tests for episode runner functionality.

This module tests the episode runner functions that integrate logging
with CartelEnv episode execution.
"""

import tempfile
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from src.episode_logging.episode_runner import (
    run_episode_with_logging,
    run_episode_with_regulator_logging,
)
from src.episode_logging.logger import Logger


class TestRunEpisodeWithLogging:
    """Test basic episode execution with logging."""

    def test_run_episode_basic(self) -> None:
        """Test basic episode execution with logging."""
        # Create mock environment
        env = Mock()
        env.n_firms = 2
        env.max_steps = 3
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.shock_std = 0.1
        env.price_min = 0.0
        env.price_max = 100.0

        # Mock environment methods
        env.reset.return_value = (np.array([0.0, 0.0]), {"step": 0})

        # Track step count
        step_count = [0]

        def mock_step(
            action: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
            step_count[0] += 1
            return (
                np.array([step_count[0], step_count[0]]),  # next_obs
                np.array([100.0, 110.0]),  # rewards
                step_count[0] >= 3,  # terminated
                False,  # truncated
                {  # step_info
                    "prices": action,
                    "demand_shock": 0.1,
                    "market_price": np.mean(action),
                    "total_demand": 100.0,
                    "individual_quantity": 50.0,
                    "total_profits": np.array([100.0, 110.0]),
                },
            )

        env.step.side_effect = mock_step

        # Create mock agents
        agents = [Mock(), Mock()]
        for i, agent in enumerate(agents):
            agent.choose_price.return_value = 50.0 + i * 5.0
            agent.reset.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            # Run episode
            results = run_episode_with_logging(
                env=env,
                agents=agents,
                log_dir=temp_dir,
                episode_id="test_episode",
                agent_types=["random", "tit_for_tat"],
            )

            # Check results structure
            assert "logger" in results
            assert "log_file" in results
            assert "episode_data" in results
            assert "episode_summary" in results

            # Check episode data
            episode_data = results["episode_data"]
            assert episode_data["total_steps"] == 3
            assert episode_data["terminated"] is True
            assert episode_data["truncated"] is False

            # Check episode summary
            episode_summary = results["episode_summary"]
            assert episode_summary["total_steps"] == 3
            assert "environment_params" in episode_summary
            assert episode_summary["environment_params"]["n_firms"] == 2

    def test_run_episode_with_custom_logger(self) -> None:
        """Test episode execution with custom logger."""
        # Create mock environment
        env = Mock()
        env.n_firms = 2
        env.max_steps = 2
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.shock_std = 0.1
        env.price_min = 0.0
        env.price_max = 100.0

        env.reset.return_value = (np.array([0.0, 0.0]), {"step": 0})

        step_count = [0]

        def mock_step(
            action: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
            step_count[0] += 1
            return (
                np.array([step_count[0], step_count[0]]),
                np.array([100.0, 110.0]),
                step_count[0] >= 2,
                False,
                {
                    "prices": action,
                    "demand_shock": 0.1,
                    "market_price": np.mean(action),
                    "total_demand": 100.0,
                    "individual_quantity": 50.0,
                    "total_profits": np.array([100.0, 110.0]),
                },
            )

        env.step.side_effect = mock_step

        # Create mock agents
        agents = [Mock(), Mock()]
        for i, agent in enumerate(agents):
            agent.choose_price.return_value = 50.0 + i * 5.0
            agent.reset.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            custom_logger = Logger(
                log_dir=temp_dir, episode_id="custom_episode", n_firms=2
            )

            results = run_episode_with_logging(
                env=env,
                agents=agents,
                logger=custom_logger,
                agent_types=["random", "random"],
            )

            assert results["logger"] is custom_logger
            assert results["episode_data"]["total_steps"] == 2

    def test_run_episode_with_additional_info(self) -> None:
        """Test episode execution with additional info."""
        # Create mock environment
        env = Mock()
        env.n_firms = 2
        env.max_steps = 1
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.shock_std = 0.1
        env.price_min = 0.0
        env.price_max = 100.0

        env.reset.return_value = (np.array([0.0, 0.0]), {"step": 0})

        def mock_step(
            action: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
            return (
                np.array([1.0, 1.0]),
                np.array([100.0, 110.0]),
                True,
                False,
                {
                    "prices": action,
                    "demand_shock": 0.1,
                    "market_price": np.mean(action),
                    "total_demand": 100.0,
                    "individual_quantity": 50.0,
                    "total_profits": np.array([100.0, 110.0]),
                },
            )

        env.step.side_effect = mock_step

        # Create mock agents
        agents = [Mock(), Mock()]
        for i, agent in enumerate(agents):
            agent.choose_price.return_value = 50.0 + i * 5.0
            agent.reset.return_value = None

        additional_info = {"custom_param": 42.0, "test_flag": True}

        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_episode_with_logging(
                env=env,
                agents=agents,
                log_dir=temp_dir,
                additional_info=additional_info,
            )

            assert results["episode_data"]["total_steps"] == 1

    def test_run_episode_with_regulator_flags(self) -> None:
        """Test episode execution with regulator flags."""
        # Create mock environment
        env = Mock()
        env.n_firms = 2
        env.max_steps = 1
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.shock_std = 0.1
        env.price_min = 0.0
        env.price_max = 100.0

        env.reset.return_value = (np.array([0.0, 0.0]), {"step": 0})

        def mock_step(
            action: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
            return (
                np.array([1.0, 1.0]),
                np.array([100.0, 110.0]),
                True,
                False,
                {
                    "prices": action,
                    "demand_shock": 0.1,
                    "market_price": np.mean(action),
                    "total_demand": 100.0,
                    "individual_quantity": 50.0,
                    "total_profits": np.array([100.0, 110.0]),
                },
            )

        env.step.side_effect = mock_step

        # Create mock agents
        agents = [Mock(), Mock()]
        for i, agent in enumerate(agents):
            agent.choose_price.return_value = 50.0 + i * 5.0
            agent.reset.return_value = None

        regulator_flags = {
            "parallel_violation": True,
            "structural_break_violation": False,
            "fines_applied": [50.0, 50.0],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_episode_with_logging(
                env=env,
                agents=agents,
                log_dir=temp_dir,
                regulator_flags=regulator_flags,
            )

            assert results["episode_data"]["total_steps"] == 1

    def test_run_episode_agent_without_reset(self) -> None:
        """Test episode execution with agents that don't have reset method."""
        # Create mock environment
        env = Mock()
        env.n_firms = 2
        env.max_steps = 1
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.shock_std = 0.1
        env.price_min = 0.0
        env.price_max = 100.0

        env.reset.return_value = (np.array([0.0, 0.0]), {"step": 0})

        def mock_step(
            action: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
            return (
                np.array([1.0, 1.0]),
                np.array([100.0, 110.0]),
                True,
                False,
                {
                    "prices": action,
                    "demand_shock": 0.1,
                    "market_price": np.mean(action),
                    "total_demand": 100.0,
                    "individual_quantity": 50.0,
                    "total_profits": np.array([100.0, 110.0]),
                },
            )

        env.step.side_effect = mock_step

        # Create mock agents without reset method
        agents = [Mock(), Mock()]
        for i, agent in enumerate(agents):
            agent.choose_price.return_value = 50.0 + i * 5.0
            # Don't add reset method

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise error even without reset method
            results = run_episode_with_logging(env=env, agents=agents, log_dir=temp_dir)

            assert results["episode_data"]["total_steps"] == 1

    def test_run_episode_agent_without_update_history(self) -> None:
        """Test episode execution with agents that don't have update_history method."""
        # Create mock environment
        env = Mock()
        env.n_firms = 2
        env.max_steps = 1
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.shock_std = 0.1
        env.price_min = 0.0
        env.price_max = 100.0

        env.reset.return_value = (np.array([0.0, 0.0]), {"step": 0})

        def mock_step(
            action: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
            return (
                np.array([1.0, 1.0]),
                np.array([100.0, 110.0]),
                True,
                False,
                {
                    "prices": action,
                    "demand_shock": 0.1,
                    "market_price": np.mean(action),
                    "total_demand": 100.0,
                    "individual_quantity": 50.0,
                    "total_profits": np.array([100.0, 110.0]),
                },
            )

        env.step.side_effect = mock_step

        # Create mock agents without update_history method
        agents = [Mock(), Mock()]
        for i, agent in enumerate(agents):
            agent.choose_price.return_value = 50.0 + i * 5.0
            agent.reset.return_value = None
            # Don't add update_history method

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise error even without update_history method
            results = run_episode_with_logging(env=env, agents=agents, log_dir=temp_dir)

            assert results["episode_data"]["total_steps"] == 1

    def test_run_episode_early_termination(self) -> None:
        """Test episode execution with early termination."""
        # Create mock environment
        env = Mock()
        env.n_firms = 2
        env.max_steps = 5
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.shock_std = 0.1
        env.price_min = 0.0
        env.price_max = 100.0

        env.reset.return_value = (np.array([0.0, 0.0]), {"step": 0})

        step_count = [0]

        def mock_step(
            action: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
            step_count[0] += 1
            return (
                np.array([step_count[0], step_count[0]]),
                np.array([100.0, 110.0]),
                step_count[0] >= 2,  # Terminate after 2 steps
                False,
                {
                    "prices": action,
                    "demand_shock": 0.1,
                    "market_price": np.mean(action),
                    "total_demand": 100.0,
                    "individual_quantity": 50.0,
                    "total_profits": np.array([100.0, 110.0]),
                },
            )

        env.step.side_effect = mock_step

        # Create mock agents
        agents = [Mock(), Mock()]
        for i, agent in enumerate(agents):
            agent.choose_price.return_value = 50.0 + i * 5.0
            agent.reset.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_episode_with_logging(env=env, agents=agents, log_dir=temp_dir)

            # Should terminate early after 2 steps
            assert results["episode_data"]["total_steps"] == 2
            assert results["episode_data"]["terminated"] is True


class TestRunEpisodeWithRegulatorLogging:
    """Test episode execution with regulator monitoring and logging."""

    def test_run_episode_with_regulator_basic(self) -> None:
        """Test basic episode execution with regulator monitoring."""
        # Create mock environment
        env = Mock()
        env.n_firms = 2
        env.max_steps = 2
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.shock_std = 0.1
        env.price_min = 0.0
        env.price_max = 100.0

        env.reset.return_value = (np.array([0.0, 0.0]), {"step": 0})

        step_count = [0]

        def mock_step(
            action: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
            step_count[0] += 1
            return (
                np.array([step_count[0], step_count[0]]),
                np.array([100.0, 110.0]),
                step_count[0] >= 2,
                False,
                {
                    "prices": action,
                    "demand_shock": 0.1,
                    "market_price": np.mean(action),
                    "total_demand": 100.0,
                    "individual_quantity": 50.0,
                    "total_profits": np.array([100.0, 110.0]),
                },
            )

        env.step.side_effect = mock_step

        # Create mock agents
        agents = [Mock(), Mock()]
        for i, agent in enumerate(agents):
            agent.choose_price.return_value = 50.0 + i * 5.0
            agent.reset.return_value = None

        # Create mock regulator
        regulator = Mock()
        regulator.monitor_step.return_value = {
            "parallel_violation": True,
            "structural_break_violation": False,
            "fines_applied": np.array([50.0, 50.0]),
            "violation_details": [{"type": "parallel", "step": 1}],
        }
        regulator.apply_penalties.return_value = np.array(
            [50.0, 60.0]
        )  # Modified rewards

        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_episode_with_regulator_logging(
                env=env,
                agents=agents,
                regulator=regulator,
                log_dir=temp_dir,
                episode_id="test_regulator_episode",
                agent_types=["random", "tit_for_tat"],
            )

            # Check results structure
            assert "logger" in results
            assert "log_file" in results
            assert "episode_data" in results
            assert "episode_summary" in results

            # Check episode data
            episode_data = results["episode_data"]
            assert episode_data["total_steps"] == 2
            assert episode_data["total_fines"] > 0
            assert "violations" in episode_data

    def test_run_episode_with_regulator_custom_logger(self) -> None:
        """Test episode execution with regulator and custom logger."""
        # Create mock environment
        env = Mock()
        env.n_firms = 2
        env.max_steps = 1
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.shock_std = 0.1
        env.price_min = 0.0
        env.price_max = 100.0

        env.reset.return_value = (np.array([0.0, 0.0]), {"step": 0})

        def mock_step(
            action: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
            return (
                np.array([1.0, 1.0]),
                np.array([100.0, 110.0]),
                True,
                False,
                {
                    "prices": action,
                    "demand_shock": 0.1,
                    "market_price": np.mean(action),
                    "total_demand": 100.0,
                    "individual_quantity": 50.0,
                    "total_profits": np.array([100.0, 110.0]),
                },
            )

        env.step.side_effect = mock_step

        # Create mock agents
        agents = [Mock(), Mock()]
        for i, agent in enumerate(agents):
            agent.choose_price.return_value = 50.0 + i * 5.0
            agent.reset.return_value = None

        # Create mock regulator
        regulator = Mock()
        regulator.monitor_step.return_value = {
            "parallel_violation": False,
            "structural_break_violation": False,
            "fines_applied": np.array([0.0, 0.0]),
            "violation_details": [],
        }
        regulator.apply_penalties.return_value = np.array(
            [100.0, 110.0]
        )  # No penalties

        with tempfile.TemporaryDirectory() as temp_dir:
            custom_logger = Logger(
                log_dir=temp_dir, episode_id="custom_regulator_episode", n_firms=2
            )

            results = run_episode_with_regulator_logging(
                env=env,
                agents=agents,
                regulator=regulator,
                logger=custom_logger,
                agent_types=["random", "random"],
            )

            assert results["logger"] is custom_logger
            assert results["episode_data"]["total_steps"] == 1

    def test_run_episode_with_regulator_no_violations(self) -> None:
        """Test episode execution with regulator but no violations."""
        # Create mock environment
        env = Mock()
        env.n_firms = 2
        env.max_steps = 1
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.shock_std = 0.1
        env.price_min = 0.0
        env.price_max = 100.0

        env.reset.return_value = (np.array([0.0, 0.0]), {"step": 0})

        def mock_step(
            action: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
            return (
                np.array([1.0, 1.0]),
                np.array([100.0, 110.0]),
                True,
                False,
                {
                    "prices": action,
                    "demand_shock": 0.1,
                    "market_price": np.mean(action),
                    "total_demand": 100.0,
                    "individual_quantity": 50.0,
                    "total_profits": np.array([100.0, 110.0]),
                },
            )

        env.step.side_effect = mock_step

        # Create mock agents
        agents = [Mock(), Mock()]
        for i, agent in enumerate(agents):
            agent.choose_price.return_value = 50.0 + i * 5.0
            agent.reset.return_value = None

        # Create mock regulator with no violations
        regulator = Mock()
        regulator.monitor_step.return_value = {
            "parallel_violation": False,
            "structural_break_violation": False,
            "fines_applied": np.array([0.0, 0.0]),
            "violation_details": [],
        }
        regulator.apply_penalties.return_value = np.array([100.0, 110.0])

        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_episode_with_regulator_logging(
                env=env, agents=agents, regulator=regulator, log_dir=temp_dir
            )

            assert results["episode_data"]["total_steps"] == 1
            assert results["episode_data"]["total_fines"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
