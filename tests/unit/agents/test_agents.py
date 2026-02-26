"""
Unit tests for firm agents in the CartelEnv environment.

This module contains comprehensive tests for all baseline firm agents:
- RandomAgent: Tests price bounds and randomness
- BestResponseAgent: Tests analytical optimal response in 2-firm static game
- TitForTatAgent: Tests price copying behavior
"""

import math

import numpy as np

from typing import Any, Dict, Optional

from src.agents.firm_agents import (
    BaseAgent,
    BestResponseAgent,
    RandomAgent,
    TitForTatAgent,
)
from src.cartel.cartel_env import CartelEnv


class TestBaseAgent:
    """Test suite for BaseAgent abstract class functionality."""

    def test_base_agent_initialization(self) -> None:
        """Test that BaseAgent initializes correctly."""

        # Create a concrete implementation for testing
        class TestAgent(BaseAgent):
            def choose_price(
                self,
                observation: np.ndarray,
                env: CartelEnv,
                info: Optional[Dict[str, Any]] = None,
            ) -> float:
                return 10.0

        agent = TestAgent(agent_id=0, seed=42)

        assert agent.agent_id == 0
        assert list(agent.price_history) == []
        assert list(agent.rival_price_history) == []

    def test_base_agent_update_history(self) -> None:
        """Test that BaseAgent updates price history correctly."""

        class TestAgent(BaseAgent):
            def choose_price(
                self,
                observation: np.ndarray,
                env: CartelEnv,
                info: Optional[Dict[str, Any]] = None,
            ) -> float:
                return 10.0

        agent = TestAgent(agent_id=0)

        # Update history
        agent.update_history(my_price=20.0, rival_prices=np.array([15.0, 25.0]))

        assert list(agent.price_history) == [20.0]
        assert list(agent.rival_price_history) == [20.0]  # Average of 15 and 25

    def test_base_agent_reset(self) -> None:
        """Test that BaseAgent reset clears history."""

        class TestAgent(BaseAgent):
            def choose_price(
                self,
                observation: np.ndarray,
                env: CartelEnv,
                info: Optional[Dict[str, Any]] = None,
            ) -> float:
                return 10.0

        agent = TestAgent(agent_id=0)

        # Add some history
        agent.update_history(my_price=20.0, rival_prices=np.array([15.0, 25.0]))
        agent.update_history(my_price=30.0, rival_prices=np.array([20.0, 30.0]))

        # Reset
        agent.reset()

        assert list(agent.price_history) == []
        assert list(agent.rival_price_history) == []


class TestRandomAgent:
    """Test suite for RandomAgent class."""

    def test_random_agent_initialization(self) -> None:
        """Test RandomAgent initialization."""
        agent = RandomAgent(agent_id=0, seed=42)

        assert agent.agent_id == 0
        assert isinstance(agent.np_random, np.random.Generator)

    def test_random_agent_chooses_prices_within_bounds(self) -> None:
        """Test that RandomAgent produces prices within environment bounds."""
        env = CartelEnv(price_min=10.0, price_max=50.0, seed=42)
        agent = RandomAgent(agent_id=0, seed=123)

        # Test multiple price choices
        for _ in range(100):
            price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)

            assert env.price_min <= price <= env.price_max
            assert isinstance(price, float)

    def test_random_agent_reproducibility_with_seed(self) -> None:
        """Test that RandomAgent with same seed produces same sequence."""
        env = CartelEnv(seed=42)
        agent1 = RandomAgent(agent_id=0, seed=123)
        agent2 = RandomAgent(agent_id=0, seed=123)

        # Generate sequences
        prices1 = [
            agent1.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
            for _ in range(10)
        ]
        prices2 = [
            agent2.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
            for _ in range(10)
        ]

        assert prices1 == prices2

    def test_random_agent_different_seeds_produce_different_sequences(self) -> None:
        """Test that RandomAgent with different seeds produces different sequences."""
        env = CartelEnv(seed=42)
        agent1 = RandomAgent(agent_id=0, seed=123)
        agent2 = RandomAgent(agent_id=0, seed=456)

        # Generate sequences
        prices1 = [
            agent1.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
            for _ in range(10)
        ]
        prices2 = [
            agent2.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
            for _ in range(10)
        ]

        assert prices1 != prices2

    def test_random_agent_price_distribution(self) -> None:
        """Test that RandomAgent produces approximately uniform distribution."""
        env = CartelEnv(price_min=0.0, price_max=100.0, seed=42)
        agent = RandomAgent(agent_id=0, seed=123)

        # Generate many prices
        n_samples = 10000
        prices = [
            agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
            for _ in range(n_samples)
        ]

        # Check that mean is approximately in the middle
        mean_price = np.mean(prices)
        expected_mean = (env.price_min + env.price_max) / 2
        assert math.isclose(mean_price, expected_mean, rel_tol=0.1)

        # Check that all prices are within bounds
        assert all(env.price_min <= p <= env.price_max for p in prices)


class TestBestResponseAgent:
    """Test suite for BestResponseAgent class."""

    def test_best_response_agent_initialization(self) -> None:
        """Test BestResponseAgent initialization."""
        agent = BestResponseAgent(agent_id=0, seed=42)

        assert agent.agent_id == 0
        assert isinstance(agent.np_random, np.random.Generator)

    def test_best_response_agent_no_history_uses_nash_equilibrium(self) -> None:
        """Test that BestResponseAgent uses Nash equilibrium when no rival history."""
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

        price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)

        # For 2-firm symmetric game with D = 100 - p_market, c = 10
        # Enhanced Nash equilibrium: p* = (|b|*c + a) / (2*|b|) = (1*10 + 100) / (2*1) = 110 / 2 = 55.0
        expected_nash = (1 * 10 + 100) / (2 * 1)

        assert math.isclose(price, expected_nash)

    def test_best_response_agent_analytical_2_firm_static_game(self) -> None:
        """Test BestResponseAgent matches analytically optimal response in 2-firm static game."""
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

        # Simulate rival setting price of 30
        rival_price = 30.0
        agent.update_history(my_price=25.0, rival_prices=np.array([rival_price]))

        price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)

        # Corrected best response: p* = (a + 2*c) / (2*|b|) - p_rival/2
        # p* = (100 + 2*10) / (2*1) - 30/2 = 120/2 - 15 = 60 - 15 = 45.0
        expected_best_response = (100 + 2 * 10) / (2 * 1) - rival_price / 2

        assert math.isclose(price, expected_best_response)

    def test_best_response_agent_with_positive_profitable_response(self) -> None:
        """Test BestResponseAgent with parameters that yield positive best response."""
        env = CartelEnv(
            n_firms=2,
            marginal_cost=5.0,
            demand_intercept=200.0,
            demand_slope=-2.0,
            price_min=1.0,
            price_max=100.0,
            seed=42,
        )
        agent = BestResponseAgent(agent_id=0, seed=42)

        # Simulate rival setting price of 20
        rival_price = 20.0
        agent.update_history(my_price=25.0, rival_prices=np.array([rival_price]))

        price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)

        # Corrected best response: p* = (a + 2*c) / (2*|b|) - p_rival/2
        # p* = (200 + 2*5) / (2*2) - 20/2 = 210/4 - 10 = 52.5 - 10 = 42.5
        expected_best_response = (200 + 2 * 5) / (2 * 2) - rival_price / 2

        assert math.isclose(price, expected_best_response)

    def test_best_response_agent_clips_to_price_bounds(self) -> None:
        """Test that BestResponseAgent clips prices to environment bounds."""
        env = CartelEnv(
            n_firms=2,
            marginal_cost=10.0,
            demand_intercept=100.0,
            demand_slope=-1.0,
            price_min=20.0,
            price_max=80.0,
            seed=42,
        )
        agent = BestResponseAgent(agent_id=0, seed=42)

        # Test with no history (should use Nash equilibrium)
        price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
        assert env.price_min <= price <= env.price_max

        # Test with rival history
        agent.update_history(my_price=25.0, rival_prices=np.array([30.0]))
        price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
        assert env.price_min <= price <= env.price_max

    def test_best_response_agent_nash_equilibrium_calculation(self) -> None:
        """Test the internal Nash equilibrium calculation."""
        env = CartelEnv(
            n_firms=3,
            marginal_cost=15.0,
            demand_intercept=150.0,
            demand_slope=-1.5,
            price_min=1.0,
            price_max=100.0,
            seed=42,
        )
        agent = BestResponseAgent(agent_id=0, seed=42)

        nash_price = agent._calculate_nash_equilibrium_price(env)

        # For n=3, c=15, a=150, b=-1.5:
        # Enhanced Nash equilibrium: p* = (|b|*c + a) / (2*|b|) = (1.5*15 + 150) / (2*1.5) = 172.5 / 3 = 57.5
        expected_nash = (1.5 * 15 + 150) / (2 * 1.5)

        assert math.isclose(nash_price, expected_nash)


class TestTitForTatAgent:
    """Test suite for TitForTatAgent class."""

    def test_tit_for_tat_agent_initialization(self) -> None:
        """Test TitForTatAgent initialization."""
        agent = TitForTatAgent(agent_id=0, seed=42)

        assert agent.agent_id == 0
        assert isinstance(agent.np_random, np.random.Generator)

    def test_tit_for_tat_agent_no_history_uses_default(self) -> None:
        """Test that TitForTatAgent uses default price when no rival history."""
        env = CartelEnv(marginal_cost=10.0, price_min=1.0, price_max=100.0, seed=42)
        agent = TitForTatAgent(agent_id=0, seed=42)

        price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)

        # Default should be marginal_cost + 5.0 = 15.0
        expected_default = env.marginal_cost + 5.0
        assert math.isclose(price, expected_default)

    def test_tit_for_tat_agent_copies_prior_prices_correctly(self) -> None:
        """Test that TitForTatAgent copies prior rival average prices correctly."""
        env = CartelEnv(price_min=1.0, price_max=100.0, seed=42)
        agent = TitForTatAgent(agent_id=0, seed=42)

        # Add some rival price history
        rival_prices_sequence = [20.0, 25.0, 30.0, 35.0]

        for i, rival_avg in enumerate(rival_prices_sequence):
            # Simulate rival prices that average to the target
            rival_prices = np.array([rival_avg - 2.0, rival_avg + 2.0])
            agent.update_history(my_price=15.0, rival_prices=rival_prices)

            # Choose price should copy the most recent rival average
            price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
            assert math.isclose(price, rival_avg)

    def test_tit_for_tat_agent_clips_to_price_bounds(self) -> None:
        """Test that TitForTatAgent clips prices to environment bounds."""
        env = CartelEnv(price_min=20.0, price_max=80.0, seed=42)
        agent = TitForTatAgent(agent_id=0, seed=42)

        # Test with no history (default price)
        price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
        assert env.price_min <= price <= env.price_max

        # Test with rival history that would be out of bounds
        agent.update_history(
            my_price=25.0, rival_prices=np.array([10.0, 15.0])
        )  # Avg = 12.5, below min
        price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
        assert price == env.price_min

        agent.update_history(
            my_price=25.0, rival_prices=np.array([90.0, 95.0])
        )  # Avg = 92.5, above max
        price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
        assert price == env.price_max

    def test_tit_for_tat_agent_multiple_rivals_averaging(self) -> None:
        """Test that TitForTatAgent correctly averages multiple rival prices."""
        env = CartelEnv(price_min=1.0, price_max=100.0, seed=42)
        agent = TitForTatAgent(agent_id=0, seed=42)

        # Test with 3 rivals
        rival_prices = np.array([20.0, 30.0, 40.0])  # Average = 30.0
        agent.update_history(my_price=25.0, rival_prices=rival_prices)

        price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
        assert math.isclose(price, 30.0)

        # Test with 4 rivals
        rival_prices = np.array([10.0, 20.0, 30.0, 40.0])  # Average = 25.0
        agent.update_history(my_price=25.0, rival_prices=rival_prices)

        price = agent.choose_price(observation=np.array([0.0, 0.0, 0.0]), env=env)
        assert math.isclose(price, 25.0)


class TestAgentIntegration:
    """Integration tests for agents working together in CartelEnv."""

    def test_agents_can_play_episode_together(self) -> None:
        """Test that multiple agents can play an episode together."""
        env = CartelEnv(n_firms=3, max_steps=5, seed=42)
        agents = [
            RandomAgent(agent_id=0, seed=123),
            BestResponseAgent(agent_id=1, seed=456),
            TitForTatAgent(agent_id=2, seed=789),
        ]

        obs, _ = env.reset(seed=42)

        for step in range(5):
            # Each agent chooses a price
            prices = []
            for i, agent in enumerate(agents):
                price = agent.choose_price(obs, env)
                prices.append(price)

            # Take environment step
            action = np.array(prices, dtype=np.float32)
            obs, rewards, terminated, truncated, info = env.step(action)

            # Update agent histories
            for i, agent in enumerate(agents):
                rival_prices = np.array(
                    [prices[j] for j in range(len(prices)) if j != i]
                )
                agent.update_history(prices[i], rival_prices)

            # Check that all agents received rewards
            assert len(rewards) == 3
            assert all(isinstance(r, (int, float, np.floating)) for r in rewards)

            if truncated:
                break

    def test_agent_price_histories_accumulate_correctly(self) -> None:
        """Test that agent price histories accumulate correctly during episode."""
        env = CartelEnv(n_firms=2, max_steps=3, seed=42)
        agents = [
            RandomAgent(agent_id=0, seed=123),
            TitForTatAgent(agent_id=1, seed=456),
        ]

        obs, _ = env.reset(seed=42)

        for step in range(3):
            # Choose prices
            prices = [agent.choose_price(obs, env) for agent in agents]

            # Take step
            action = np.array(prices, dtype=np.float32)
            obs, rewards, terminated, truncated, info = env.step(action)

            # Update histories
            for i, agent in enumerate(agents):
                rival_prices = np.array(
                    [prices[j] for j in range(len(prices)) if j != i]
                )
                agent.update_history(prices[i], rival_prices)

            # Check history lengths
            for agent in agents:
                assert len(agent.price_history) == step + 1
                assert len(agent.rival_price_history) == step + 1

        # Check that histories contain the expected values
        assert len(agents[0].price_history) == 3
        assert len(agents[1].price_history) == 3

    def test_agents_reset_clears_histories(self) -> None:
        """Test that calling reset on agents clears their histories."""
        env = CartelEnv(n_firms=2, max_steps=2, seed=42)
        agents = [
            RandomAgent(agent_id=0, seed=123),
            TitForTatAgent(agent_id=1, seed=456),
        ]

        # Play one episode
        obs, _ = env.reset(seed=42)
        for step in range(2):
            prices = [agent.choose_price(obs, env) for agent in agents]
            action = np.array(prices, dtype=np.float32)
            obs, rewards, terminated, truncated, info = env.step(action)

            for i, agent in enumerate(agents):
                rival_prices = np.array(
                    [prices[j] for j in range(len(prices)) if j != i]
                )
                agent.update_history(prices[i], rival_prices)

        # Check histories are populated
        for agent in agents:
            assert len(agent.price_history) == 2
            assert len(agent.rival_price_history) == 2

        # Reset agents
        for agent in agents:
            agent.reset()

        # Check histories are cleared
        for agent in agents:
            assert len(agent.price_history) == 0
            assert len(agent.rival_price_history) == 0
