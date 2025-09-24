"""
Tests for the SimplifiedCartelEnv class.

This module tests the simplified cartel environment functionality including
environment initialization, step execution, economic calculations, and edge cases.
"""

import pytest
import numpy as np
import gymnasium as gym

from src.cartel.simplified_cartel_env import SimplifiedCartelEnv


class TestSimplifiedCartelEnv:
    """Test the SimplifiedCartelEnv class."""

    @pytest.fixture
    def env(self):
        """Create a basic simplified cartel environment for testing."""
        return SimplifiedCartelEnv(
            n_firms=3,
            max_steps=100,
            marginal_cost=10.0,
            demand_intercept=100.0,
            demand_slope=-1.0,
            shock_std=5.0,
            price_min=1.0,
            price_max=100.0,
            max_price_change=20.0,
            competition_intensity=2.0,
            use_fixed_costs=True,
            fixed_cost=50.0,
            use_capacity_constraints=False,
            seed=42,
        )

    def test_initialization_default_params(self):
        """Test environment initialization with default parameters."""
        env = SimplifiedCartelEnv()

        assert env.n_firms == 3
        assert env.max_steps == 100
        assert env.marginal_cost == 10.0
        assert env.demand_intercept == 100.0
        assert env.demand_slope == -1.0
        assert env.shock_std == 5.0
        assert env.price_min == 1.0
        assert env.price_max == 100.0
        assert env.max_price_change == 20.0
        assert env.competition_intensity == 2.0
        assert env.use_fixed_costs is True
        assert env.fixed_cost == 50.0
        assert env.use_capacity_constraints is False

    def test_initialization_custom_params(self, env):
        """Test environment initialization with custom parameters."""
        assert env.n_firms == 3
        assert env.max_steps == 100
        assert env.marginal_cost == 10.0
        assert env.demand_intercept == 100.0
        assert env.demand_slope == -1.0
        assert env.shock_std == 5.0
        assert env.price_min == 1.0
        assert env.price_max == 100.0
        assert env.max_price_change == 20.0
        assert env.competition_intensity == 2.0
        assert env.use_fixed_costs is True
        assert env.fixed_cost == 50.0
        assert env.use_capacity_constraints is False

    def test_initialization_validation_errors(self):
        """Test initialization parameter validation."""
        # Test invalid n_firms
        with pytest.raises(ValueError, match="n_firms must be at least 1"):
            SimplifiedCartelEnv(n_firms=0)

        # Test invalid max_steps
        with pytest.raises(ValueError, match="max_steps must be at least 1"):
            SimplifiedCartelEnv(max_steps=0)

        # Test invalid marginal_cost
        with pytest.raises(ValueError, match="marginal_cost must be positive"):
            SimplifiedCartelEnv(marginal_cost=0)

        # Test invalid demand_slope
        with pytest.raises(ValueError, match="demand_slope must be negative"):
            SimplifiedCartelEnv(demand_slope=1.0)

        # Test invalid shock_std
        with pytest.raises(ValueError, match="shock_std must be non-negative"):
            SimplifiedCartelEnv(shock_std=-1.0)

        # Test invalid price range
        with pytest.raises(ValueError, match="price_min must be less than price_max"):
            SimplifiedCartelEnv(price_min=100.0, price_max=50.0)

        # Test invalid max_price_change
        with pytest.raises(ValueError, match="max_price_change must be positive"):
            SimplifiedCartelEnv(max_price_change=0)

        # Test invalid competition_intensity
        with pytest.raises(ValueError, match="competition_intensity must be positive"):
            SimplifiedCartelEnv(competition_intensity=0)

        # Test invalid fixed_cost
        with pytest.raises(ValueError, match="fixed_cost must be non-negative"):
            SimplifiedCartelEnv(fixed_cost=-1.0)

    def test_initialization_marginal_costs_validation(self):
        """Test marginal costs parameter validation."""
        # Test wrong length
        with pytest.raises(
            ValueError, match="marginal_costs must have length equal to n_firms"
        ):
            SimplifiedCartelEnv(n_firms=3, marginal_costs=[10.0, 15.0])

        # Test negative marginal costs
        with pytest.raises(ValueError, match="All marginal costs must be positive"):
            SimplifiedCartelEnv(n_firms=3, marginal_costs=[10.0, -5.0, 15.0])

    def test_initialization_capacity_validation(self):
        """Test capacity parameter validation."""
        # Test wrong length
        with pytest.raises(
            ValueError, match="capacity must have length equal to n_firms"
        ):
            SimplifiedCartelEnv(
                n_firms=3, use_capacity_constraints=True, capacity=[100.0, 150.0]
            )

        # Test negative capacity
        with pytest.raises(ValueError, match="All capacity values must be positive"):
            SimplifiedCartelEnv(
                n_firms=3, use_capacity_constraints=True, capacity=[100.0, -50.0, 150.0]
            )

    def test_marginal_costs_setup(self):
        """Test marginal costs setup."""
        # Test with custom marginal costs
        env = SimplifiedCartelEnv(n_firms=3, marginal_costs=[10.0, 15.0, 20.0])
        expected_costs = np.array([10.0, 15.0, 20.0], dtype=np.float32)
        np.testing.assert_array_equal(env.marginal_costs, expected_costs)

        # Test with default marginal cost
        env = SimplifiedCartelEnv(n_firms=3, marginal_cost=12.0)
        expected_costs = np.array([12.0, 12.0, 12.0], dtype=np.float32)
        np.testing.assert_array_equal(env.marginal_costs, expected_costs)

    def test_capacity_setup(self):
        """Test capacity constraints setup."""
        # Test with capacity constraints
        env = SimplifiedCartelEnv(
            n_firms=3, use_capacity_constraints=True, capacity=[100.0, 150.0, 200.0]
        )
        expected_capacity = np.array([100.0, 150.0, 200.0], dtype=np.float32)
        np.testing.assert_array_equal(env.capacity_array, expected_capacity)

        # Test without capacity constraints
        env = SimplifiedCartelEnv(n_firms=3, use_capacity_constraints=False)
        assert env.capacity_array is None

    def test_action_space(self, env):
        """Test action space definition."""
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (3,)  # n_firms
        assert np.all(env.action_space.low == 1.0)
        assert np.all(env.action_space.high == 100.0)
        assert env.action_space.dtype == np.float32

    def test_observation_space(self, env):
        """Test observation space definition."""
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (8,)  # n_firms * 2 + 2
        assert env.observation_space.dtype == np.float32

    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset(seed=123)

        # Check observation shape and type
        assert obs.shape == (8,)
        assert obs.dtype == np.float32
        assert np.all(obs == 0)  # Initial observation should be zeros

        # Check info dictionary
        assert info["step"] == 0
        assert "demand_shock" in info
        assert info["total_demand"] == 0.0
        assert info["market_price"] == 0.0
        assert len(info["prices"]) == 3
        assert len(info["profits"]) == 3

        # Check state reset
        assert env.current_step == 0
        assert env.previous_prices is None

    def test_reset_with_seed(self, env):
        """Test environment reset with specific seed."""
        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)

        # Should be identical with same seed
        np.testing.assert_array_equal(obs1, obs2)
        assert info1["demand_shock"] == info2["demand_shock"]

    def test_step_basic(self, env):
        """Test basic step execution."""
        obs, info = env.reset(seed=42)

        # Simple action: all firms set price to 50
        action = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Check return values
        assert obs.shape == (8,)
        assert len(rewards) == 3
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Check info dictionary
        assert info["step"] == 1
        assert len(info["prices"]) == 3
        assert len(info["profits"]) == 3
        assert "market_price" in info
        assert "total_demand" in info
        assert "individual_quantity" in info
        assert "market_shares" in info
        assert "demand_shock" in info
        assert "costs" in info

    def test_step_price_clipping(self, env):
        """Test price clipping to valid range."""
        obs, info = env.reset(seed=42)

        # Action with prices outside valid range - but within action space bounds
        action = np.array([150.0, -10.0, 200.0], dtype=np.float32)

        # This should raise an error because action is outside action space
        with pytest.raises(ValueError, match="Action .* is not in action space"):
            env.step(action)

    def test_step_price_change_limits(self, env):
        """Test price change limits for stability."""
        obs, info = env.reset(seed=42)

        # First step
        action1 = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action1)

        # Second step with large price changes
        action2 = np.array([100.0, 100.0, 100.0], dtype=np.float32)  # +50 change
        obs, rewards, terminated, truncated, info = env.step(action2)

        # Price changes should be limited to max_price_change
        price_changes = np.abs(np.array(info["prices"]) - np.array(action1))
        assert all(change <= env.max_price_change for change in price_changes)

    def test_step_termination(self, env):
        """Test episode termination after max_steps."""
        obs, info = env.reset(seed=42)

        action = np.array([50.0, 50.0, 50.0], dtype=np.float32)

        # Run until termination
        for step in range(env.max_steps + 1):
            obs, rewards, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert terminated is True
        assert truncated is False
        assert info["step"] == env.max_steps

    def test_calculate_demand(self, env):
        """Test demand calculation."""
        # Test with different market prices
        prices1 = np.array([50.0, 50.0, 50.0])
        demand1 = env._calculate_demand(prices1)

        prices2 = np.array([60.0, 60.0, 60.0])
        demand2 = env._calculate_demand(prices2)

        # Higher prices should result in lower demand (negative slope)
        assert demand2 < demand1

        # Test with demand shock
        env.current_demand_shock = 10.0
        demand_with_shock = env._calculate_demand(prices1)
        assert demand_with_shock > demand1

    def test_calculate_demand_non_negative(self, env):
        """Test that demand is always non-negative."""
        # Test with very high prices
        high_prices = np.array([200.0, 200.0, 200.0])
        demand = env._calculate_demand(high_prices)
        assert demand >= 0.0

        # Test with negative shock
        env.current_demand_shock = -200.0
        demand = env._calculate_demand(high_prices)
        assert demand >= 0.0

    def test_calculate_market_shares(self, env):
        """Test market share calculation."""
        # Test with different prices
        prices = np.array([40.0, 50.0, 60.0])
        shares = env._calculate_market_shares(prices)

        # Lower prices should get higher market shares
        assert shares[0] > shares[1] > shares[2]

        # Market shares should sum to 1
        assert abs(sum(shares) - 1.0) < 1e-6

        # All shares should be non-negative
        assert all(share >= 0 for share in shares)

    def test_calculate_market_shares_equal_prices(self, env):
        """Test market share calculation with equal prices."""
        prices = np.array([50.0, 50.0, 50.0])
        shares = env._calculate_market_shares(prices)

        # Should be approximately equal
        assert all(abs(share - 1 / 3) < 1e-6 for share in shares)

    def test_calculate_costs_without_fixed_costs(self):
        """Test cost calculation without fixed costs."""
        env = SimplifiedCartelEnv(
            n_firms=3, use_fixed_costs=False, marginal_costs=[10.0, 15.0, 20.0]
        )

        quantities = np.array([5.0, 3.0, 2.0])
        costs = env._calculate_costs(quantities)

        expected_costs = np.array([50.0, 45.0, 40.0], dtype=np.float32)
        np.testing.assert_array_equal(costs, expected_costs)

    def test_calculate_costs_with_fixed_costs(self, env):
        """Test cost calculation with fixed costs."""
        quantities = np.array([5.0, 3.0, 2.0])
        costs = env._calculate_costs(quantities)

        # Should include both fixed and variable costs
        expected_variable_costs = np.array([50.0, 30.0, 20.0], dtype=np.float32)
        expected_fixed_costs = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        expected_total_costs = expected_variable_costs + expected_fixed_costs

        np.testing.assert_array_equal(costs, expected_total_costs)

    def test_capacity_constraints(self):
        """Test capacity constraints functionality."""
        env = SimplifiedCartelEnv(
            n_firms=3,
            use_capacity_constraints=True,
            capacity=[10.0, 20.0, 30.0],
            demand_intercept=200.0,  # High demand to test constraints
            demand_slope=-0.5,
        )

        obs, info = env.reset(seed=42)

        # Set low prices to generate high demand
        action = np.array([20.0, 20.0, 20.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Check that quantities don't exceed capacity
        quantities = info["individual_quantity"]
        assert quantities[0] <= 10.0
        assert quantities[1] <= 20.0
        assert quantities[2] <= 30.0

    def test_capacity_constraints_market_share_recalculation(self):
        """Test market share recalculation after capacity constraints."""
        env = SimplifiedCartelEnv(
            n_firms=3,
            use_capacity_constraints=True,
            capacity=[5.0, 5.0, 5.0],  # Low capacity
            demand_intercept=200.0,
            demand_slope=-0.5,
        )

        obs, info = env.reset(seed=42)

        # Set low prices to generate high demand
        action = np.array([20.0, 20.0, 20.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Market shares should still sum to 1 after capacity constraints
        shares = info["market_shares"]
        assert abs(sum(shares) - 1.0) < 1e-6

    def test_render(self, env):
        """Test render functionality."""
        obs, info = env.reset(seed=42)
        action = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Should not raise an exception
        env.render(mode="human")

    def test_close(self, env):
        """Test close functionality."""
        # Should not raise an exception
        env.close()

    def test_step_action_validation(self, env):
        """Test action validation."""
        obs, info = env.reset(seed=42)

        # Test invalid action shape
        invalid_action = np.array([50.0, 50.0])  # Wrong shape
        with pytest.raises(ValueError, match="Action .* is not in action space"):
            env.step(invalid_action)

    def test_step_info_consistency(self, env):
        """Test that step info is consistent."""
        obs, info = env.reset(seed=42)

        action = np.array([50.0, 60.0, 70.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Check that prices in info match action (after clipping)
        assert len(info["prices"]) == 3
        assert len(info["profits"]) == 3
        assert len(info["individual_quantity"]) == 3
        assert len(info["market_shares"]) == 3
        assert len(info["costs"]) == 3

        # Check that market price is average of individual prices
        expected_market_price = np.mean(info["prices"])
        assert abs(info["market_price"] - expected_market_price) < 1e-6

    def test_step_profit_calculation(self, env):
        """Test profit calculation."""
        obs, info = env.reset(seed=42)

        action = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Check that profits are calculated correctly
        for i, (price, quantity, profit, cost) in enumerate(
            zip(
                info["prices"],
                info["individual_quantity"],
                info["profits"],
                info["costs"],
            )
        ):
            expected_profit = price * quantity - cost
            assert abs(profit - expected_profit) < 1e-6

    def test_step_demand_shock_update(self, env):
        """Test that demand shock is updated each step."""
        obs, info = env.reset(seed=42)
        initial_shock = info["demand_shock"]

        action = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Demand shock should be different (random)
        assert info["demand_shock"] != initial_shock

    def test_step_previous_prices_update(self, env):
        """Test that previous prices are updated each step."""
        obs, info = env.reset(seed=42)
        assert env.previous_prices is None

        action = np.array([50.0, 60.0, 70.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Previous prices should be set
        assert env.previous_prices is not None
        np.testing.assert_array_equal(env.previous_prices, info["prices"])

    def test_step_current_step_update(self, env):
        """Test that current step is updated each step."""
        obs, info = env.reset(seed=42)
        assert env.current_step == 0

        action = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        assert env.current_step == 1
        assert info["step"] == 1

    def test_edge_case_zero_demand(self):
        """Test edge case with zero demand."""
        env = SimplifiedCartelEnv(
            n_firms=3,
            demand_intercept=0.0,
            demand_slope=-1.0,
            shock_std=0.0,  # No shocks
        )

        obs, info = env.reset(seed=42)

        # Set high prices to get zero demand
        action = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        assert info["total_demand"] == 0.0
        assert all(q == 0.0 for q in info["individual_quantity"])
        assert all(
            profit <= 0 for profit in info["profits"]
        )  # Should be losses due to fixed costs

    def test_edge_case_single_firm(self):
        """Test edge case with single firm."""
        env = SimplifiedCartelEnv(n_firms=1)

        obs, info = env.reset(seed=42)
        assert obs.shape == (
            4,
        )  # 1 * 2 + 2 (prices + profits + market_price + total_demand)

        action = np.array([50.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        assert len(info["prices"]) == 1
        assert len(info["profits"]) == 1
        assert info["market_price"] == 50.0
        assert info["market_shares"] == [1.0]

    def test_edge_case_very_high_prices(self, env):
        """Test edge case with very high prices."""
        obs, info = env.reset(seed=42)

        # Set prices at maximum
        action = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Should handle gracefully
        assert info["total_demand"] >= 0.0
        assert all(price == 100.0 for price in info["prices"])

    def test_edge_case_very_low_prices(self, env):
        """Test edge case with very low prices."""
        obs, info = env.reset(seed=42)

        # Set prices at minimum
        action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Should handle gracefully
        assert info["total_demand"] >= 0.0
        assert all(price == 1.0 for price in info["prices"])

    def test_competition_intensity_effect(self):
        """Test that competition intensity affects market shares."""
        # Low competition intensity
        env_low = SimplifiedCartelEnv(
            n_firms=3,
            competition_intensity=0.5,
        )

        # High competition intensity
        env_high = SimplifiedCartelEnv(
            n_firms=3,
            competition_intensity=5.0,
        )

        prices = np.array([40.0, 50.0, 60.0])

        shares_low = env_low._calculate_market_shares(prices)
        shares_high = env_high._calculate_market_shares(prices)

        # Higher competition intensity should make price differences more important
        # Low price firm should get even higher share with high competition intensity
        assert shares_high[0] > shares_low[0]
        assert shares_high[2] < shares_low[2]  # High price firm should get lower share
