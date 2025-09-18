"""
Unit tests for firm agents.

This module contains comprehensive tests for the firm agent classes,
including validation of price selection, history tracking, and strategic behavior.
"""

import math

import numpy as np

from src.agents.firm_agents import BestResponseAgent, RandomAgent, TitForTatAgent
from src.cartel.cartel_env import CartelEnv


class TestRandomAgent:
    """Test suite for RandomAgent class."""

    def test_initialization(self) -> None:
        """Test RandomAgent initialization."""
        agent = RandomAgent(agent_id=0, seed=42)

        assert agent.agent_id == 0
        assert len(agent.price_history) == 0
        assert len(agent.rival_price_history) == 0

    def test_choose_price_within_bounds(self) -> None:
        """Test that RandomAgent produces prices within bounds."""
        env = CartelEnv(n_firms=2, price_min=10.0, price_max=50.0, seed=42)
        agent = RandomAgent(agent_id=0, seed=42)

        # Test multiple price selections
        for _ in range(100):
            observation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            price = agent.choose_price(observation, env)

            assert env.price_min <= price <= env.price_max
            assert isinstance(price, float)

    def test_choose_price_reproducibility(self) -> None:
        """Test that RandomAgent with same seed produces same prices."""
        env = CartelEnv(n_firms=2, seed=42)
        agent1 = RandomAgent(agent_id=0, seed=123)
        agent2 = RandomAgent(agent_id=0, seed=123)

        observation = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Should produce same prices with same seed
        price1 = agent1.choose_price(observation, env)
        price2 = agent2.choose_price(observation, env)

        assert math.isclose(price1, price2)

    def test_update_history(self) -> None:
        """Test that RandomAgent correctly updates price history."""
        agent = RandomAgent(agent_id=0, seed=42)

        # Update history
        agent.update_history(20.0, np.array([25.0, 30.0]))

        assert len(agent.price_history) == 1
        assert agent.price_history[0] == 20.0
        assert len(agent.rival_price_history) == 1
        assert math.isclose(agent.rival_price_history[0], 27.5)  # (25 + 30) / 2

    def test_reset(self) -> None:
        """Test that RandomAgent reset clears history."""
        agent = RandomAgent(agent_id=0, seed=42)

        # Add some history
        agent.update_history(20.0, np.array([25.0, 30.0]))
        agent.update_history(22.0, np.array([26.0, 28.0]))

        # Reset
        agent.reset()

        assert len(agent.price_history) == 0
        assert len(agent.rival_price_history) == 0


class TestBestResponseAgent:
    """Test suite for BestResponseAgent class."""

    def test_initialization(self) -> None:
        """Test BestResponseAgent initialization."""
        agent = BestResponseAgent(agent_id=0, seed=42)

        assert agent.agent_id == 0
        assert len(agent.price_history) == 0
        assert len(agent.rival_price_history) == 0

    def test_choose_price_no_history_uses_nash(self) -> None:
        """Test that BestResponseAgent uses Nash equilibrium when no history."""
        env = CartelEnv(
            n_firms=2,
            marginal_cost=10.0,
            demand_intercept=100.0,
            demand_slope=-1.0,
            price_min=1.0,
            price_max=100.0,
            seed=42,
        )
        agent = BestResponseAgent(agent_id=0, seed=42)

        observation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        price = agent.choose_price(observation, env)

        # For 2-firm symmetric game with D = 100 - p_market, c = 10
        # Nash equilibrium: p* = (100 + 10) / (2*(-1) - (-1)/2) = 110 / (-2 + 0.5) = 110 / (-1.5) = -73.33
        # But this is negative, so it should be clipped to price_min
        expected_nash = max(
            env.price_min, min(env.price_max, (100 + 10) / (2 * (-1) - (-1) / 2))
        )

        assert math.isclose(price, expected_nash)

    def test_choose_price_with_history(self) -> None:
        """Test that BestResponseAgent responds optimally to rival prices."""
        env = CartelEnv(
            n_firms=2,
            marginal_cost=10.0,
            demand_intercept=100.0,
            demand_slope=-1.0,
            price_min=1.0,
            price_max=100.0,
            seed=42,
        )
        agent = BestResponseAgent(agent_id=0, seed=42)

        # Add some rival price history
        agent.update_history(20.0, np.array([30.0]))  # Rival price = 30

        observation = np.array([20.0, 30.0, 0.0], dtype=np.float32)
        price = agent.choose_price(observation, env)

        # Best response to rival price 30:
        # p* = (a + c + b*p_rival) / (2*b) = (100 + 10 + (-1)*30) / (2*(-1)) = 80 / (-2) = -40
        # But this is negative, so it should be clipped to price_min
        expected_response = max(
            env.price_min, min(env.price_max, (100 + 10 + (-1) * 30) / (2 * (-1)))
        )

        assert math.isclose(price, expected_response)

    def test_choose_price_clipping(self) -> None:
        """Test that BestResponseAgent clips prices to valid range."""
        env = CartelEnv(
            n_firms=2,
            marginal_cost=10.0,
            demand_intercept=100.0,
            demand_slope=-1.0,
            price_min=50.0,  # High minimum
            price_max=60.0,  # Low maximum
            seed=42,
        )
        agent = BestResponseAgent(agent_id=0, seed=42)

        observation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        price = agent.choose_price(observation, env)

        assert env.price_min <= price <= env.price_max

    def test_nash_equilibrium_calculation(self) -> None:
        """Test Nash equilibrium calculation for 2-firm static game."""
        env = CartelEnv(
            n_firms=2,
            marginal_cost=10.0,
            demand_intercept=100.0,
            demand_slope=-1.0,
            price_min=1.0,
            price_max=100.0,
            seed=42,
        )
        agent = BestResponseAgent(agent_id=0, seed=42)

        nash_price = agent._calculate_nash_equilibrium_price(env)

        # For 2-firm symmetric game: p* = (a + c) / (2*b - b/n)
        # = (100 + 10) / (2*(-1) - (-1)/2) = 110 / (-2 + 0.5) = 110 / (-1.5) = -73.33
        # But this is negative, so it should be clipped to price_min
        expected_nash = max(
            env.price_min, min(env.price_max, (100 + 10) / (2 * (-1) - (-1) / 2))
        )

        assert math.isclose(nash_price, expected_nash)

    def test_analytical_optimal_response_2_firm(self) -> None:
        """Test that BestResponseAgent matches analytically optimal response in 2-firm static game."""
        # Set up a scenario where we can analytically verify the best response
        env = CartelEnv(
            n_firms=2,
            marginal_cost=5.0,
            demand_intercept=50.0,
            demand_slope=-0.5,  # Less steep demand curve
            price_min=1.0,
            price_max=100.0,
            seed=42,
        )
        agent = BestResponseAgent(agent_id=0, seed=42)

        # Set rival price to 20
        agent.update_history(15.0, np.array([20.0]))

        observation = np.array([15.0, 20.0, 0.0], dtype=np.float32)
        price = agent.choose_price(observation, env)

        # Best response to rival price 20:
        # p* = (a + c + b*p_rival) / (2*b) = (50 + 5 + (-0.5)*20) / (2*(-0.5)) = 45 / (-1) = -45
        # But this is negative, so it should be clipped to price_min
        expected_response = max(
            env.price_min, min(env.price_max, (50 + 5 + (-0.5) * 20) / (2 * (-0.5)))
        )

        assert math.isclose(price, expected_response)


class TestTitForTatAgent:
    """Test suite for TitForTatAgent class."""

    def test_initialization(self) -> None:
        """Test TitForTatAgent initialization."""
        agent = TitForTatAgent(agent_id=0, seed=42)

        assert agent.agent_id == 0
        assert len(agent.price_history) == 0
        assert len(agent.rival_price_history) == 0

    def test_choose_price_no_history_uses_default(self) -> None:
        """Test that TitForTatAgent uses default price when no history."""
        env = CartelEnv(
            n_firms=2, marginal_cost=10.0, price_min=1.0, price_max=100.0, seed=42
        )
        agent = TitForTatAgent(agent_id=0, seed=42)

        observation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        price = agent.choose_price(observation, env)

        # Should use marginal_cost + 5.0 = 15.0, clipped to bounds
        expected_default = max(env.price_min, min(env.price_max, 15.0))

        assert math.isclose(price, expected_default)

    def test_choose_price_copies_prior_prices(self) -> None:
        """Test that TitForTatAgent copies prior rival average prices correctly."""
        env = CartelEnv(n_firms=3, price_min=1.0, price_max=100.0, seed=42)
        agent = TitForTatAgent(agent_id=0, seed=42)

        # Add rival price history
        agent.update_history(20.0, np.array([25.0, 30.0]))  # Rival avg = 27.5
        agent.update_history(22.0, np.array([28.0, 32.0]))  # Rival avg = 30.0

        observation = np.array([22.0, 28.0, 32.0, 0.0], dtype=np.float32)
        price = agent.choose_price(observation, env)

        # Should copy the most recent rival average (30.0)
        assert math.isclose(price, 30.0)

    def test_choose_price_clipping(self) -> None:
        """Test that TitForTatAgent clips prices to valid range."""
        env = CartelEnv(
            n_firms=2,
            price_min=50.0,  # High minimum
            price_max=60.0,  # Low maximum
            seed=42,
        )
        agent = TitForTatAgent(agent_id=0, seed=42)

        # Add rival price history with price outside bounds
        agent.update_history(20.0, np.array([40.0]))  # Rival avg = 40.0 (below min)

        observation = np.array([20.0, 40.0, 0.0], dtype=np.float32)
        price = agent.choose_price(observation, env)

        # Should be clipped to price_min
        assert price == env.price_min

    def test_multiple_price_copies(self) -> None:
        """Test that TitForTatAgent correctly copies multiple rival prices."""
        env = CartelEnv(n_firms=2, seed=42)
        agent = TitForTatAgent(agent_id=0, seed=42)

        # Test sequence of rival prices
        rival_prices = [20.0, 25.0, 30.0, 35.0]

        for i, rival_price in enumerate(rival_prices):
            # Update history with this rival price
            agent.update_history(15.0 + i, np.array([rival_price]))

            observation = np.array([15.0 + i, rival_price, 0.0], dtype=np.float32)
            price = agent.choose_price(observation, env)

            # Should copy the rival price
            assert math.isclose(price, rival_price)

    def test_update_history(self) -> None:
        """Test that TitForTatAgent correctly updates price history."""
        agent = TitForTatAgent(agent_id=0, seed=42)

        # Update history
        agent.update_history(20.0, np.array([25.0, 30.0]))

        assert len(agent.price_history) == 1
        assert agent.price_history[0] == 20.0
        assert len(agent.rival_price_history) == 1
        assert math.isclose(agent.rival_price_history[0], 27.5)  # (25 + 30) / 2

    def test_reset(self) -> None:
        """Test that TitForTatAgent reset clears history."""
        agent = TitForTatAgent(agent_id=0, seed=42)

        # Add some history
        agent.update_history(20.0, np.array([25.0, 30.0]))
        agent.update_history(22.0, np.array([26.0, 28.0]))

        # Reset
        agent.reset()

        assert len(agent.price_history) == 0
        assert len(agent.rival_price_history) == 0


class TestAgentIntegration:
    """Integration tests for agents with CartelEnv."""

    def test_agents_with_environment_step(self) -> None:
        """Test that agents work correctly with environment steps."""
        env = CartelEnv(n_firms=2, seed=42)
        agent1 = RandomAgent(agent_id=0, seed=42)
        agent2 = BestResponseAgent(agent_id=1, seed=43)

        obs, info = env.reset(seed=42)

        # Get prices from both agents
        price1 = agent1.choose_price(obs, env, info)
        price2 = agent2.choose_price(obs, env, info)

        # Take step with these prices
        action = np.array([price1, price2], dtype=np.float32)
        next_obs, rewards, terminated, truncated, next_info = env.step(action)

        # Update agent histories
        agent1.update_history(price1, np.array([price2]))
        agent2.update_history(price2, np.array([price1]))

        # Verify results
        assert len(rewards) == 2
        assert not terminated
        assert not truncated
        assert len(next_obs) == 3  # 2 prices + 1 demand shock

    def test_agent_price_bounds_with_environment(self) -> None:
        """Test that agent prices are within environment bounds."""
        env = CartelEnv(n_firms=3, price_min=10.0, price_max=50.0, seed=42)
        agents = [
            RandomAgent(agent_id=0, seed=42),
            BestResponseAgent(agent_id=1, seed=43),
            TitForTatAgent(agent_id=2, seed=44),
        ]

        obs, info = env.reset(seed=42)

        prices = []
        for agent in agents:
            price = agent.choose_price(obs, env, info)
            prices.append(price)

            # Verify price is within bounds
            assert env.price_min <= price <= env.price_max

        # Take step with these prices
        action = np.array(prices, dtype=np.float32)
        next_obs, rewards, terminated, truncated, next_info = env.step(action)

        # Verify environment accepted the prices
        assert len(rewards) == 3
        assert np.allclose(next_obs[:3], prices)
