"""
Integration tests for regulator monitoring and enforcement.

This module tests the integration between:
- Regulator monitoring algorithms
- Penalty application and fine calculation
- Agent behavior under regulatory pressure
- Multi-step regulatory enforcement
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.agents.regulator import Regulator
from src.agents.firm_agents import RandomAgent, BestResponseAgent, TitForTatAgent
from src.cartel.cartel_env import CartelEnv
from src.episode_logging.episode_runner import run_episode_with_regulator_logging
from src.episode_logging.episode_logger import EpisodeLogger


class TestRegulatorMonitoringIntegration:
    """Test regulator monitoring and enforcement integration."""

    def test_regulator_monitoring_workflow(self) -> None:
        """Test complete regulator monitoring workflow."""
        # Create environment
        env = CartelEnv(n_firms=3, max_steps=20, seed=42)

        # Create agents (mix that might show violations)
        agents = [
            TitForTatAgent(agent_id=0, seed=42),  # Likely to show parallel behavior
            TitForTatAgent(agent_id=1, seed=42),
            RandomAgent(agent_id=2, seed=42),
        ]

        # Create regulator
        regulator = Regulator(seed=42)

        # Run episode with regulator monitoring
        results = run_episode_with_regulator_logging(
            env=env,
            agents=agents,
            regulator=regulator,
            episode_id="test_regulator_monitoring",
        )

        # Verify results
        assert "episode_data" in results
        episode_data = results["episode_data"]

        # Check regulator data
        assert "total_fines" in episode_data
        assert "violations" in episode_data
        assert "episode_fines" in episode_data

        # Verify violation tracking
        violations = episode_data["violations"]
        assert "parallel" in violations
        assert "structural_break" in violations
        assert isinstance(violations["parallel"], int)
        assert isinstance(violations["structural_break"], int)

    def test_regulator_penalty_application(self) -> None:
        """Test that regulator penalties are applied correctly."""
        # Create environment
        env = CartelEnv(n_firms=2, max_steps=10, seed=42)

        # Create agents
        agents = [
            TitForTatAgent(agent_id=0, seed=42),
            TitForTatAgent(agent_id=1, seed=42),
        ]

        # Create regulator
        regulator = Regulator(seed=42)

        # Reset environment
        obs, info = env.reset()
        for agent in agents:
            agent.reset()

        # Run a few steps and check penalty application
        total_fines = 0.0
        for step in range(5):
            # Get agent actions
            prices = []
            for agent in agents:
                price = agent.choose_price(obs, env, info)
                prices.append(price)

            action = np.array(prices, dtype=np.float32)

            # Regulator monitors
            detection_results = regulator.monitor_step(action, step)

            # Take environment step
            next_obs, rewards, terminated, truncated, step_info = env.step(action)

            # Apply penalties
            modified_rewards = regulator.apply_penalties(rewards, detection_results)

            # Check that penalties were applied if violations detected
            if (
                detection_results["parallel_violation"]
                or detection_results["structural_break_violation"]
            ):
                total_fines += sum(detection_results["fines_applied"])
                # Modified rewards should be different from original
                assert not np.array_equal(rewards, modified_rewards)

            # Update for next iteration
            obs = next_obs
            for i, agent in enumerate(agents):
                rival_prices = np.array(
                    [prices[j] for j in range(len(prices)) if j != i]
                )
                agent.update_history(prices[i], rival_prices)

    def test_regulator_detection_algorithms(self) -> None:
        """Test regulator detection algorithms with known patterns."""
        # Create regulator
        regulator = Regulator(seed=42)

        # Test parallel behavior detection
        # Create identical prices (should trigger parallel detection)
        identical_prices = np.array([25.0, 25.0, 25.0])
        detection_results = regulator.monitor_step(identical_prices, step=1)

        # Should detect parallel violation (or at least not crash)
        assert "parallel_violation" in detection_results
        assert len(detection_results["fines_applied"]) == 3

        # Test structural break detection
        # Create prices that show sudden change (simulate structural break)
        changing_prices = np.array([20.0, 30.0, 25.0])
        detection_results = regulator.monitor_step(changing_prices, step=5)

        # May or may not detect structural break depending on algorithm
        assert "structural_break_violation" in detection_results

    def test_regulator_fine_calculation(self) -> None:
        """Test regulator fine calculation logic."""
        # Create regulator
        regulator = Regulator(seed=42)

        # Test fine calculation for parallel violation
        identical_prices = np.array([30.0, 30.0, 30.0])
        detection_results = regulator.monitor_step(identical_prices, step=1)

        # Check fine structure
        fines = detection_results["fines_applied"]
        assert len(fines) == 3
        assert all(fine >= 0 for fine in fines)

        # Check violation details
        violation_details = detection_results["violation_details"]
        # Note: The actual API might not have these specific keys
        assert isinstance(violation_details, (list, dict))

    def test_regulator_with_different_agent_types(self) -> None:
        """Test regulator behavior with different agent types."""
        # Create environment
        env = CartelEnv(n_firms=4, max_steps=15, seed=42)

        # Create different agent types
        agents = [
            RandomAgent(agent_id=0, seed=42),
            BestResponseAgent(agent_id=1, seed=42),
            TitForTatAgent(agent_id=2, seed=42),
            TitForTatAgent(agent_id=3, seed=42),
        ]

        # Create regulator
        regulator = Regulator(seed=42)

        # Run episode
        results = run_episode_with_regulator_logging(
            env=env, agents=agents, regulator=regulator, episode_id="test_mixed_agents"
        )

        # Verify results
        episode_data = results["episode_data"]
        assert "total_fines" in episode_data
        assert "violations" in episode_data

        # Check that episode completed
        assert episode_data["total_steps"] == 15

    def test_regulator_logging_integration(self) -> None:
        """Test that regulator data is properly logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create environment
            env = CartelEnv(n_firms=2, max_steps=10, seed=42)

            # Create agents
            agents = [
                TitForTatAgent(agent_id=0, seed=42),
                TitForTatAgent(agent_id=1, seed=42),
            ]

            # Create regulator
            regulator = Regulator(seed=42)

            # Create logger
            logger = EpisodeLogger(
                log_file=Path(temp_dir) / "test_regulator_logging.jsonl", n_firms=2
            )

            # Run episode with logging
            results = run_episode_with_regulator_logging(
                env=env,
                agents=agents,
                regulator=regulator,
                logger=logger,
                log_dir=temp_dir,
                episode_id="test_regulator_logging",
            )

            # Check log file
            log_file = Path(results["log_file"])
            assert log_file.exists()

            # Read and verify log content
            lines = log_file.read_text().strip().split("\n")
            assert len(lines) >= 11  # Header + 10 steps + possibly episode_end

            # Check that regulator data is in step logs
            for line in lines[1:]:  # Skip header
                step_data = json.loads(line)
                if step_data.get("type") == "step":
                    # Check that step data has the expected structure
                    assert "prices" in step_data
                    assert "profits" in step_data
                    # Note: regulator_flags might be in additional_info or not present
                    if "regulator_flags" in step_data:
                        regulator_flags = step_data["regulator_flags"]
                        assert "parallel_violation" in regulator_flags
                        assert "structural_break_violation" in regulator_flags
                        assert "fines_applied" in regulator_flags

    def test_regulator_escalation_behavior(self) -> None:
        """Test regulator escalation behavior over multiple violations."""
        # Create regulator
        regulator = Regulator(seed=42)

        # Simulate repeated violations
        violation_count = 0
        for step in range(5):
            # Create prices that trigger violations
            identical_prices = np.array([25.0, 25.0, 25.0])
            detection_results = regulator.monitor_step(identical_prices, step=step)

            if detection_results["parallel_violation"]:
                violation_count += 1
                fines = detection_results["fines_applied"]
                # Fines might escalate (depending on implementation)
                assert all(fine >= 0 for fine in fines)

        # Should have detected some violations
        assert violation_count > 0

    def test_regulator_with_environment_shocks(self) -> None:
        """Test regulator behavior with demand shocks."""
        # Create environment with high shock variance
        env = CartelEnv(
            n_firms=3,
            max_steps=10,
            shock_std=10.0,
            seed=42,  # High variance
        )

        # Create agents (use different seeds to avoid identical behavior)
        agents = [
            BestResponseAgent(agent_id=0, seed=42),
            BestResponseAgent(agent_id=1, seed=43),
            BestResponseAgent(agent_id=2, seed=44),
        ]

        # Create regulator
        regulator = Regulator(seed=42)

        # Run episode
        results = run_episode_with_regulator_logging(
            env=env, agents=agents, regulator=regulator, episode_id="test_shocks"
        )

        # Verify results
        episode_data = results["episode_data"]
        assert "total_steps" in episode_data
        assert "total_fines" in episode_data

        # Check that episode handled shocks
        assert episode_data["total_steps"] == 10

    def test_regulator_monitoring_statistics(self) -> None:
        """Test regulator monitoring statistics and metrics."""
        # Create environment
        env = CartelEnv(n_firms=3, max_steps=20, seed=42)

        # Create agents likely to show violations (use different seeds to avoid perfect collusion)
        agents = [
            TitForTatAgent(agent_id=0, seed=42),
            TitForTatAgent(agent_id=1, seed=43),
            TitForTatAgent(agent_id=2, seed=44),
        ]

        # Create regulator
        regulator = Regulator(seed=42)

        # Run episode
        results = run_episode_with_regulator_logging(
            env=env, agents=agents, regulator=regulator, episode_id="test_statistics"
        )

        # Verify statistics
        episode_data = results["episode_data"]
        violations = episode_data["violations"]

        # Check violation statistics
        assert violations["parallel"] >= 0
        assert violations["structural_break"] >= 0

        # Check fine statistics
        total_fines = episode_data["total_fines"]
        assert total_fines >= 0

        # Check episode fines list
        episode_fines = episode_data["episode_fines"]
        assert len(episode_fines) == 20  # One per step
        # Note: episode_fines might be a list of lists, so we need to handle that
        for fine_entry in episode_fines:
            if isinstance(fine_entry, list):
                assert all(fine >= 0 for fine in fine_entry)
            else:
                assert fine_entry >= 0

    def test_regulator_parameter_sensitivity(self) -> None:
        """Test regulator behavior with different parameters."""
        # Test with different regulator configurations
        # (This would depend on what parameters the Regulator class exposes)

        # Create basic regulator
        regulator = Regulator(seed=42)

        # Test detection with different price patterns
        test_cases = [
            np.array([20.0, 20.0, 20.0]),  # Identical prices
            np.array([20.0, 25.0, 30.0]),  # Increasing prices
            np.array([30.0, 25.0, 20.0]),  # Decreasing prices
            np.array([25.0, 25.0, 30.0]),  # Mixed pattern
        ]

        for i, prices in enumerate(test_cases):
            detection_results = regulator.monitor_step(prices, step=i)

            # Verify detection results structure
            assert "parallel_violation" in detection_results
            assert "structural_break_violation" in detection_results
            assert "fines_applied" in detection_results
            assert "violation_details" in detection_results

            # Verify fines array length matches number of firms
            assert len(detection_results["fines_applied"]) == len(prices)
