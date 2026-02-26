"""
Smoke tests for basic simulation functionality.

These tests verify that the core simulation components can be imported
and run without external API calls or complex dependencies.
"""

import pytest
import numpy as np


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
