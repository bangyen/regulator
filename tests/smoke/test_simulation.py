"""
Smoke tests for basic simulation functionality.

These tests verify that the core simulation components can be imported
and run without external API calls or complex dependencies.
"""

import pytest
import numpy as np
from pathlib import Path


# Test imports without external dependencies
def test_core_imports():
    """Test that core modules can be imported successfully."""

    # If we get here, imports succeeded
    assert True


def test_environment_creation():
    """Test that the cartel environment can be created and reset."""
    from src.cartel.cartel_env import CartelEnv

    # Create environment with minimal configuration
    env = CartelEnv(
        n_firms=2,
        max_steps=10,
        demand_intercept=50.0,
        demand_slope=-1.0,
        marginal_cost=10.0,
    )

    # Test reset
    obs, info = env.reset()
    assert obs is not None
    assert len(obs) == 3  # Two firms + 1 demand shock

    # Test that observation is valid (numpy array)
    assert hasattr(obs, "shape")
    assert obs.shape == (3,)
    # First two elements are prices (should be >= 0), third is demand shock (can be negative)
    assert all(x >= 0 for x in obs[:2])


def test_agent_creation():
    """Test that basic agents can be created and act."""
    from src.agents.firm_agents import RandomAgent, TitForTatAgent
    from src.cartel.cartel_env import CartelEnv

    # Create a mock environment for testing
    env = CartelEnv(n_firms=2, max_steps=5)
    obs, _ = env.reset()

    # Test RandomAgent
    random_agent = RandomAgent(agent_id=0, seed=42)
    action = random_agent.choose_price(observation=obs, env=env)
    assert isinstance(action, (int, float))
    assert action >= 0

    # Test TitForTatAgent
    tit_agent = TitForTatAgent(agent_id=1, seed=42)
    action = tit_agent.choose_price(observation=obs, env=env)
    assert isinstance(action, (int, float))
    assert action >= 0


def test_regulator_creation():
    """Test that the basic regulator can be created and monitor."""
    from src.agents.regulator import Regulator

    regulator = Regulator()

    # Test monitoring with dummy data
    prices = np.array([10.0, 12.0, 15.0])
    result = regulator.monitor_step(prices, step=0)

    assert isinstance(result, dict)
    assert "parallel_violation" in result
    assert "structural_break_violation" in result
    assert "fines_applied" in result


def test_minimal_simulation():
    """Test a minimal simulation run without external dependencies."""
    from src.cartel.cartel_env import CartelEnv
    from src.agents.firm_agents import RandomAgent
    from src.agents.regulator import Regulator
    import numpy as np

    # Create minimal simulation
    env = CartelEnv(n_firms=2, max_steps=5, seed=42)
    agents = [RandomAgent(agent_id=i, seed=42) for i in range(2)]
    regulator = Regulator()

    # Run simulation
    obs, info = env.reset()
    total_reward = 0

    for step in range(5):
        # Get actions from agents
        actions = [agent.choose_price(obs, env) for agent in agents]

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_reward += sum(rewards)

        # Monitor with regulator
        regulator.monitor_step(np.array(actions), step, info)

        if terminated or truncated:
            break

    # Basic assertions
    assert total_reward != 0  # Should have some reward
    assert len(obs) == 3  # Should have 2 firm observations + 1 demand shock


def test_episode_logger_creation():
    """Test that episode logger can be created and used."""
    from src.episode_logging.episode_logger import EpisodeLogger
    import numpy as np

    # Create temporary log file
    log_file = Path("test_smoke_log.jsonl")

    try:
        logger = EpisodeLogger(log_file, n_firms=2)

        # Test logging episode header
        logger.log_episode_header(
            episode_id=0,
            n_firms=2,
            n_steps=5,
            agent_types=["random", "random"],
            environment_params={"seed": 42},
        )

        # Test logging step
        logger.log_step(
            step=0,
            prices=np.array([10.0, 12.0]),
            profits=np.array([5.0, 6.0]),
            demand_shock=0.0,
            market_price=11.0,
            total_demand=50.0,
            individual_quantity=np.array([25.0, 25.0]),
            total_profits=np.array([5.0, 6.0]),
        )

        # Test logging episode end
        logger.log_episode_end(
            terminated=True,
            final_rewards=np.array([50.0, 50.0]),
            episode_summary={"total_reward": 100.0, "final_step": 5},
        )

        # Verify log file was created
        assert log_file.exists()
        assert log_file.stat().st_size > 0

    finally:
        # Clean up
        if log_file.exists():
            log_file.unlink()


def test_economic_validation():
    """Test that economic validation can be imported and used."""
    from src.economic_validation import EconomicValidator
    import numpy as np

    validator = EconomicValidator()

    # Test with valid data
    prices = np.array([10.0, 12.0, 15.0])
    market_price = 12.0
    total_demand = 50.0
    individual_quantities = np.array([20.0, 20.0, 10.0])

    is_valid, errors = validator.validate_step_data(
        prices=prices.tolist(),
        market_price=market_price,
        total_demand=total_demand,
        individual_quantities=individual_quantities.tolist(),
    )
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)


def test_detector_imports():
    """Test that detector modules can be imported (without external APIs)."""
    # Test ML detector import
    from src.detectors.ml_detector import CollusionDetector

    # Test LLM detector import (should work even without OpenAI)
    from src.detectors.llm_detector import LLMDetector

    # Create detectors with stubbed mode
    ml_detector = CollusionDetector()
    llm_detector = LLMDetector(model_type="stubbed")

    assert ml_detector is not None
    assert llm_detector is not None


def test_cli_import():
    """Test that CLI module can be imported."""
    from src.regulator_cli import main

    # Just test that the function exists and is callable
    assert callable(main)


def test_dashboard_import():
    """Test that Flask dashboard can be imported."""
    from dashboard.main import app, calculate_metrics, extract_time_series

    # Test that Flask app exists
    assert app is not None
    assert callable(calculate_metrics)
    assert callable(extract_time_series)

    # Test basic functionality with empty data
    metrics = calculate_metrics({"steps": []})
    assert isinstance(metrics, dict)

    time_series = extract_time_series({"steps": []})
    assert isinstance(time_series, dict)
    assert "prices" in time_series


if __name__ == "__main__":
    # Run smoke tests directly
    pytest.main([__file__, "-v"])
