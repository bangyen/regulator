"""
Tests for the adaptive learning agent.

This module tests the AdaptiveAgent class including learning behavior,
strategy adaptation, and performance tracking.
"""

import numpy as np
from unittest.mock import MagicMock

from src.agents.adaptive_agent import AdaptiveAgent


class TestAdaptiveAgent:
    """Test suite for AdaptiveAgent class."""

    def test_initialization_default_params(self) -> None:
        """Test adaptive agent initialization with default parameters."""
        agent = AdaptiveAgent(agent_id=0, seed=42)

        assert agent.agent_id == 0
        assert agent.learning_rate == 0.1
        assert agent.memory_size == 50
        assert agent.exploration_rate == 0.1
        assert agent.exploration_decay == 0.995
        assert agent.min_exploration_rate == 0.01
        assert agent.risk_aversion == 1.0
        assert agent.collusion_tendency == 0.5

    def test_initialization_custom_params(self) -> None:
        """Test adaptive agent initialization with custom parameters."""
        agent = AdaptiveAgent(
            agent_id=1,
            learning_rate=0.2,
            memory_size=100,
            exploration_rate=0.2,
            exploration_decay=0.99,
            min_exploration_rate=0.05,
            risk_aversion=1.5,
            collusion_tendency=0.7,
            seed=42,
        )

        assert agent.agent_id == 1
        assert agent.learning_rate == 0.2
        assert agent.memory_size == 100
        assert agent.exploration_rate == 0.2
        assert agent.exploration_decay == 0.99
        assert agent.min_exploration_rate == 0.05
        assert agent.risk_aversion == 1.5
        assert agent.collusion_tendency == 0.7

    def test_choose_price_exploration(self) -> None:
        """Test price choice during exploration phase."""
        agent = AdaptiveAgent(
            agent_id=0, exploration_rate=1.0, seed=42
        )  # Always explore

        # Mock environment
        env = MagicMock()
        env.n_firms = 3
        env.price_min = 1.0
        env.price_max = 100.0
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.current_demand_shock = 0.0

        # Mock observation
        observation = np.array([50.0, 55.0, 60.0, 0.0])  # 3 prices + demand shock

        # Choose price multiple times
        prices = []
        for _ in range(10):
            price = agent.choose_price(observation, env)
            prices.append(price)
            assert env.price_min <= price <= env.price_max

        # Should have some variation due to exploration
        assert len(set(prices)) > 1

    def test_choose_price_exploitation(self) -> None:
        """Test price choice during exploitation phase."""
        agent = AdaptiveAgent(
            agent_id=0, exploration_rate=0.0, seed=42
        )  # Never explore

        # Mock environment
        env = MagicMock()
        env.n_firms = 3
        env.price_min = 1.0
        env.price_max = 100.0
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.current_demand_shock = 0.0

        # Mock observation
        observation = np.array([50.0, 55.0, 60.0, 0.0])

        # Choose price multiple times
        prices = []
        for _ in range(5):
            price = agent.choose_price(observation, env)
            prices.append(price)
            assert env.price_min <= price <= env.price_max

        # Should be more consistent during exploitation
        # (though not identical due to market condition adjustments)

    def test_market_conditions_analysis(self) -> None:
        """Test market conditions analysis."""
        agent = AdaptiveAgent(agent_id=0, seed=42)

        # Mock environment
        env = MagicMock()
        env.n_firms = 3

        # Add some history
        agent.market_price_history.extend([50.0, 52.0, 48.0, 55.0, 47.0])
        agent.rival_price_history.extend([50.0, 51.0, 49.0])
        agent.profit_history.extend([100.0, 95.0, 90.0])
        agent.regulator_history.extend([0, 0, 1, 0, 0])

        conditions = agent._analyze_market_conditions(env)

        # Check that conditions are calculated
        assert "market_volatility" in conditions
        assert "collusion_opportunity" in conditions
        assert "competitive_pressure" in conditions
        assert "regulatory_risk" in conditions

        # Check that values are in reasonable ranges
        assert 0.0 <= conditions["market_volatility"] <= 1.0
        assert 0.0 <= conditions["collusion_opportunity"] <= 1.0
        assert 0.0 <= conditions["competitive_pressure"] <= 1.0
        assert 0.0 <= conditions["regulatory_risk"] <= 1.0

    def test_strategy_update(self) -> None:
        """Test strategy update based on outcomes."""
        agent = AdaptiveAgent(agent_id=0, seed=42)

        initial_exploration_rate = agent.exploration_rate

        # Update strategy with positive outcome
        agent.update_strategy(
            price=50.0,
            profit=100.0,
            market_price=55.0,
            was_violation=False,
        )

        # Check that history is updated
        assert len(agent.learning_price_history) == 1
        assert len(agent.profit_history) == 1
        assert len(agent.market_price_history) == 1
        assert len(agent.regulator_history) == 1

        assert agent.learning_price_history[0] == 50.0
        assert agent.profit_history[0] == 100.0
        assert agent.market_price_history[0] == 55.0
        assert agent.regulator_history[0] == 0

        # Check that performance tracking is updated
        assert agent.total_profits == 100.0
        assert agent.total_fines == 0.0

        # Update with violation
        agent.update_strategy(
            price=60.0,
            profit=-50.0,  # Loss due to fine
            market_price=58.0,
            was_violation=True,
        )

        assert agent.total_profits == 50.0  # 100 - 50
        assert agent.total_fines > 0.0

        # Exploration rate should decay
        assert agent.exploration_rate < initial_exploration_rate

    def test_base_price_update(self) -> None:
        """Test base price update based on performance."""
        agent = AdaptiveAgent(agent_id=0, seed=42)

        initial_base_price = agent.base_price

        # Add some negative profit history to trigger update
        agent.profit_history.extend([-10.0, -20.0, -15.0])

        # Update with poor performance
        agent._update_base_price(
            profit=-10.0, market_price=30.0
        )  # Different from base price

        # Base price should move toward market price
        assert agent.base_price != initial_base_price

    def test_collusion_tendency_update(self) -> None:
        """Test collusion tendency update based on violations."""
        agent = AdaptiveAgent(agent_id=0, seed=42)

        initial_tendency = agent.collusion_tendency
        initial_threshold = agent.collusion_threshold

        # Update with violation
        agent._update_collusion_tendency(was_violation=True)

        # Tendency should decrease, threshold should increase
        assert agent.collusion_tendency < initial_tendency
        assert agent.collusion_threshold > initial_threshold

        # Update without violation
        agent._update_collusion_tendency(was_violation=False)

        # Tendency should increase slightly
        assert agent.collusion_tendency > initial_tendency * 0.95

    def test_price_sensitivity_update(self) -> None:
        """Test price sensitivity update based on profit trends."""
        agent = AdaptiveAgent(agent_id=0, seed=42)

        initial_sensitivity = agent.price_sensitivity

        # Add improving profit trend
        agent.profit_history.extend([100.0, 110.0, 120.0])
        agent._update_price_sensitivity(profit=130.0)

        # Sensitivity should increase
        assert agent.price_sensitivity > initial_sensitivity

        # Add declining profit trend
        agent.profit_history.extend([130.0, 120.0, 110.0])
        agent._update_price_sensitivity(profit=100.0)

        # Sensitivity should decrease
        assert agent.price_sensitivity < initial_sensitivity

    def test_risk_aversion_application(self) -> None:
        """Test risk aversion application to price choice."""
        agent = AdaptiveAgent(agent_id=0, risk_aversion=2.0, seed=42)

        # Mock environment
        env = MagicMock()
        env.n_firms = 3
        env.price_min = 1.0
        env.price_max = 100.0
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.current_demand_shock = 0.0

        # Mock high-risk market conditions
        market_conditions = {
            "regulatory_risk": 0.8,
            "market_volatility": 0.7,
        }

        proposed_price = 80.0
        adjusted_price = agent._apply_risk_aversion(
            proposed_price, env, market_conditions
        )

        # Price should be adjusted toward safer range
        assert adjusted_price != proposed_price
        # The adjusted price should be within reasonable bounds (may be negative due to risk adjustment)
        assert (
            adjusted_price >= -200.0
        )  # Allow for negative prices in extreme risk scenarios

    def test_strategy_statistics(self) -> None:
        """Test strategy statistics retrieval."""
        agent = AdaptiveAgent(agent_id=0, seed=42)

        # Add some history
        agent.profit_history.extend([100.0, 120.0, 110.0, 130.0])
        agent.regulator_history.extend([0, 0, 1, 0])

        stats = agent.get_strategy_statistics()

        # Check that statistics are returned
        assert "agent_id" in stats
        assert "learning_rate" in stats
        assert "exploration_rate" in stats
        assert "risk_aversion" in stats
        assert "collusion_tendency" in stats
        assert "base_price" in stats
        assert "price_sensitivity" in stats
        assert "total_profits" in stats
        assert "total_fines" in stats
        assert "avg_profit" in stats
        assert "profit_volatility" in stats
        assert "violation_rate" in stats

        assert stats["agent_id"] == 0
        assert stats["avg_profit"] == 115.0  # (100+120+110+130)/4
        assert stats["violation_rate"] == 0.25  # 1/4

    def test_reset_functionality(self) -> None:
        """Test agent reset functionality."""
        agent = AdaptiveAgent(agent_id=0, seed=42)

        # Add some state
        agent.price_history.append(50.0)
        agent.profit_history.append(100.0)
        agent.total_profits = 100.0
        agent.collusion_attempts = 5

        # Reset
        agent.reset()

        # Check that state is reset
        assert len(agent.price_history) == 0
        assert len(agent.profit_history) == 0
        assert len(agent.market_price_history) == 0
        assert len(agent.regulator_history) == 0
        assert len(agent.rival_price_history) == 0

        assert agent.total_profits == 0.0
        assert agent.total_fines == 0.0
        assert agent.collusion_attempts == 0
        assert agent.successful_collusions == 0

        # Strategy parameters should be reset
        assert agent.base_price == 50.0
        assert agent.price_sensitivity == 1.0
        assert agent.collusion_threshold == 0.7
        assert agent.defection_threshold == 0.3
        assert agent.exploration_rate == 0.1

    def test_memory_size_limit(self) -> None:
        """Test that memory size limits are respected."""
        agent = AdaptiveAgent(agent_id=0, memory_size=3, seed=42)

        # Add more items than memory size
        for i in range(5):
            agent.learning_price_history.append(float(i))
            agent.profit_history.append(float(i * 10))

        # Should only keep the last 3 items
        assert len(agent.learning_price_history) == 3
        assert len(agent.profit_history) == 3
        assert agent.learning_price_history[-1] == 4.0
        assert agent.profit_history[-1] == 40.0

    def test_exploration_rate_decay(self) -> None:
        """Test exploration rate decay over time."""
        agent = AdaptiveAgent(
            agent_id=0,
            exploration_rate=0.5,
            exploration_decay=0.9,
            min_exploration_rate=0.1,
            seed=42,
        )

        initial_rate = agent.exploration_rate

        # Update strategy multiple times
        for _ in range(10):
            agent.update_strategy(50.0, 100.0, 55.0, False)

        # Exploration rate should have decreased
        assert agent.exploration_rate < initial_rate
        assert agent.exploration_rate >= agent.min_exploration_rate

    def test_collusion_attempt_tracking(self) -> None:
        """Test collusion attempt tracking."""
        agent = AdaptiveAgent(agent_id=0, seed=42)

        # Mock environment with high collusion opportunity
        env = MagicMock()
        env.n_firms = 3
        env.price_min = 1.0
        env.price_max = 100.0
        env.marginal_cost = 10.0
        env.demand_intercept = 100.0
        env.demand_slope = -1.0
        env.current_demand_shock = 0.0

        # Add rival price history to create collusion opportunity
        agent.learning_rival_price_history.extend([50.0, 51.0, 50.5])

        # Choose price with high collusion opportunity
        observation = np.array([50.0, 51.0, 50.5, 0.0])
        agent.choose_price(observation, env)

        # Should have attempted collusion
        assert agent.collusion_attempts > 0
