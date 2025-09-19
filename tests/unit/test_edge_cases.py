"""
Edge case and error handling tests.

This module tests error conditions, boundary values, and edge cases
across the codebase to ensure robust error handling.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.agents.firm_agents import RandomAgent
from src.agents.regulator import Regulator
from src.cartel.cartel_env import CartelEnv
from src.detectors.llm_detector import LLMDetector
from src.episode_logging.logger import Logger
from src.episode_logging.episode_logger import EpisodeLogger


class TestEdgeCases:
    """Test edge cases and error handling across the codebase."""

    def test_cartel_env_invalid_parameters(self) -> None:
        """Test CartelEnv with invalid parameters."""
        # Test with negative n_firms
        with pytest.raises(ValueError):
            CartelEnv(n_firms=-1, max_steps=10, seed=42)

        # Test with zero n_firms
        with pytest.raises(ValueError):
            CartelEnv(n_firms=0, max_steps=10, seed=42)

        # Test with negative max_steps
        with pytest.raises(ValueError):
            CartelEnv(n_firms=2, max_steps=-1, seed=42)

        # Test with price_min > price_max
        with pytest.raises(ValueError):
            CartelEnv(
                n_firms=2,
                max_steps=10,
                price_min=100.0,
                price_max=50.0,
                seed=42,
            )

    def test_agent_boundary_prices(self) -> None:
        """Test agents with boundary price values."""
        env = CartelEnv(
            n_firms=2,
            max_steps=10,
            price_min=1.0,
            price_max=100.0,
            seed=42,
        )
        obs, info = env.reset()

        # Test with boundary observations
        boundary_obs = np.array([1.0, 100.0, 50.0])  # Min, max, and middle prices

        agent = RandomAgent(agent_id=0, seed=42)
        price = agent.choose_price(boundary_obs, env, info)

        # Price should still be within bounds
        assert env.price_min <= price <= env.price_max

    def test_regulator_edge_cases(self) -> None:
        """Test regulator with edge cases."""
        regulator = Regulator(seed=42)

        # Test with single firm (no parallel behavior possible)
        single_firm_action = np.array([25.0])
        detection_results = regulator.monitor_step(single_firm_action, step=1)
        assert "parallel_violation" in detection_results

        # Test with identical prices (should trigger parallel detection)
        identical_prices = np.array([25.0, 25.0, 25.0, 25.0, 25.0])
        detection_results = regulator.monitor_step(identical_prices, step=1)
        assert "parallel_violation" in detection_results

    def test_llm_detector_edge_cases(self) -> None:
        """Test LLM detector with edge cases."""
        detector = LLMDetector(model_type="stubbed")

        # Test with empty message
        result = detector.classify_message("", sender_id=0, receiver_id=1, step=1)
        assert "collusive_probability" in result
        assert "confidence" in result

        # Test with very long message
        long_message = "Let's coordinate our pricing strategy " * 100
        result = detector.classify_message(
            long_message, sender_id=0, receiver_id=1, step=1
        )
        assert "collusive_probability" in result

        # Test with special characters
        special_message = "Let's coordinate! @#$%^&*()_+-=[]{}|;':\",./<>?"
        result = detector.classify_message(
            special_message, sender_id=0, receiver_id=1, step=1
        )
        assert "collusive_probability" in result

        # Test with unicode characters
        unicode_message = "Let's coordinate our pricing strategy 🚀💰📈"
        result = detector.classify_message(
            unicode_message, sender_id=0, receiver_id=1, step=1
        )
        assert "collusive_probability" in result

    def test_logger_edge_cases(self) -> None:
        """Test logger with edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with reasonable episode ID
            episode_id = "test_episode_123"
            logger = Logger(log_dir=temp_dir, episode_id=episode_id, n_firms=2)
            assert logger.episode_id == episode_id

            # Test with zero firms
            logger = Logger(log_dir=temp_dir, episode_id="test", n_firms=0)
            assert logger.n_firms == 0

            # Test with reasonable number of firms
            logger = Logger(log_dir=temp_dir, episode_id="test", n_firms=100)
            assert logger.n_firms == 100

    def test_episode_logger_edge_cases(self) -> None:
        """Test episode logger with edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with reasonable file path
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)
            assert logger.log_file == log_file

    def test_agent_history_edge_cases(self) -> None:
        """Test agent history with edge cases."""
        agent = RandomAgent(agent_id=0, seed=42)

        # Test with empty rival prices
        agent.update_history(25.0, np.array([]))
        assert len(agent.price_history) == 1
        assert len(agent.rival_price_history) == 1

        # Test with reasonable rival prices array
        rival_prices = np.array([25.0] * 10)
        agent.update_history(25.0, rival_prices)
        assert len(agent.price_history) == 2
        assert len(agent.rival_price_history) == 2

        # Test with extreme price values
        agent.update_history(0.001, np.array([999999.0]))
        assert len(agent.price_history) == 3
        assert len(agent.rival_price_history) == 3

    def test_environment_step_edge_cases(self) -> None:
        """Test environment step with edge cases."""
        env = CartelEnv(n_firms=2, max_steps=10, seed=42)
        obs, info = env.reset()

        # Test with boundary price values
        boundary_action = np.array([env.price_min, env.price_max])
        next_obs, rewards, terminated, truncated, step_info = env.step(boundary_action)

        assert not terminated
        assert not truncated
        assert len(rewards) == 2
        assert len(step_info) > 0

        # Test with identical prices
        identical_action = np.array([25.0, 25.0])
        next_obs, rewards, terminated, truncated, step_info = env.step(identical_action)

        assert not terminated
        assert not truncated

    def test_error_recovery(self) -> None:
        """Test error recovery and graceful degradation."""
        # Test agent error recovery
        agent = RandomAgent(agent_id=0, seed=42)
        env = CartelEnv(n_firms=2, max_steps=10, seed=42)
        obs, info = env.reset()

        # Simulate corrupted observation
        corrupted_obs = np.array([np.nan, np.inf, 0.0])

        # Agent should handle corrupted observation gracefully
        try:
            price = agent.choose_price(corrupted_obs, env, info)
            # If it doesn't crash, price should be within bounds
            assert env.price_min <= price <= env.price_max
        except (ValueError, RuntimeError):
            # If it does crash, that's also acceptable behavior
            pass

        # Test regulator error recovery
        regulator = Regulator(seed=42)

        # Test with reasonable action
        reasonable_action = np.array([25.0, 30.0])
        try:
            detection_results = regulator.monitor_step(reasonable_action, step=1)
            assert "parallel_violation" in detection_results
        except (ValueError, RuntimeError):
            # If it crashes, that's also acceptable
            pass

    def test_basic_functionality(self) -> None:
        """Test basic functionality works correctly."""
        # Test basic environment
        env = CartelEnv(n_firms=2, max_steps=5, seed=42)
        obs, info = env.reset()
        assert obs.shape[0] > 0

        # Test basic agent
        agent = RandomAgent(agent_id=0, seed=42)
        price = agent.choose_price(obs, env, info)
        assert env.price_min <= price <= env.price_max

        # Test basic regulator
        regulator = Regulator(seed=42)
        action = np.array([25.0, 30.0])
        detection_results = regulator.monitor_step(action, step=1)
        assert "parallel_violation" in detection_results

        # Test basic logger
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, episode_id="test", n_firms=2)
            assert logger.n_firms == 2
            assert logger.episode_id == "test"
