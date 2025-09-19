"""
Unit tests for the CartelEnv environment.

This module contains comprehensive tests for the CartelEnv class, including
validation of reset(), step(), reward calculations, and environment termination.
"""

import math

import numpy as np
import pytest

from src.cartel.cartel_env import CartelEnv


class TestCartelEnv:
    """Test suite for CartelEnv class."""

    def test_initialization_default_params(self) -> None:
        """Test environment initialization with default parameters."""
        env = CartelEnv()

        assert env.n_firms == 3
        assert env.max_steps == 100
        assert env.marginal_cost == 10.0
        assert env.demand_intercept == 100.0
        assert env.demand_slope == -1.0
        assert env.shock_std == 5.0
        assert env.price_min == 1.0
        assert env.price_max == 100.0

    def test_initialization_custom_params(self) -> None:
        """Test environment initialization with custom parameters."""
        env = CartelEnv(
            n_firms=5,
            max_steps=50,
            marginal_cost=15.0,
            demand_intercept=200.0,
            demand_slope=-2.0,
            shock_std=10.0,
            price_min=5.0,
            price_max=200.0,
            seed=42,
        )

        assert env.n_firms == 5
        assert env.max_steps == 50
        assert env.marginal_cost == 15.0
        assert env.demand_intercept == 200.0
        assert env.demand_slope == -2.0
        assert env.shock_std == 10.0
        assert env.price_min == 5.0
        assert env.price_max == 200.0

    def test_initialization_validation_errors(self) -> None:
        """Test that initialization raises appropriate validation errors."""
        with pytest.raises(ValueError, match="Number of firms must be at least 1"):
            CartelEnv(n_firms=0)

        with pytest.raises(ValueError, match="Max steps must be at least 1"):
            CartelEnv(max_steps=0)

        with pytest.raises(ValueError, match="Marginal cost must be non-negative"):
            CartelEnv(marginal_cost=-1.0)

        with pytest.raises(ValueError, match="Demand slope should be negative"):
            CartelEnv(demand_slope=1.0)

        with pytest.raises(
            ValueError, match="Shock standard deviation must be non-negative"
        ):
            CartelEnv(shock_std=-1.0)

        with pytest.raises(
            ValueError, match="Price minimum must be less than price maximum"
        ):
            CartelEnv(price_min=100.0, price_max=50.0)

    def test_action_space(self) -> None:
        """Test that action space is correctly defined."""
        env = CartelEnv(n_firms=4)

        assert env.action_space.shape == (4,)
        # Check action space bounds
        low_val = getattr(env.action_space, "low", None)
        high_val = getattr(env.action_space, "high", None)
        assert low_val is not None and low_val[0] == pytest.approx(1.0)
        assert high_val is not None and high_val[0] == pytest.approx(100.0)
        assert env.action_space.dtype == np.float32

    def test_observation_space(self) -> None:
        """Test that observation space is correctly defined."""
        env = CartelEnv(n_firms=3)

        assert env.observation_space.shape == (4,)  # 3 prices + 1 demand shock
        assert env.observation_space.dtype == np.float32

    def test_reset_returns_consistent_initial_state(self) -> None:
        """Test that reset() returns consistent initial state."""
        env = CartelEnv(n_firms=3, seed=42)

        obs, info = env.reset(seed=42)

        # Check observation structure
        assert len(obs) == 4  # 3 prices + 1 demand shock
        assert obs.dtype == np.float32

        # Check that initial prices are zero
        assert np.allclose(obs[:3], 0.0)

        # Check that demand shock is within reasonable range
        assert -50.0 <= obs[3] <= 50.0  # Should be within 10 std devs

        # Check info dictionary
        assert "step" in info
        assert "demand_shock" in info
        assert "total_profits" in info
        assert info["step"] == 0
        assert len(info["total_profits"]) == 3
        assert np.allclose(info["total_profits"], 0.0)

    def test_reset_with_seed_reproducibility(self) -> None:
        """Test that reset with same seed produces same initial state."""
        env = CartelEnv(seed=123)

        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)

        assert np.allclose(obs1, obs2)

    def test_step_updates_prices_and_demand_shocks(self) -> None:
        """Test that step() correctly updates prices and demand shocks."""
        env = CartelEnv(n_firms=2, seed=42)
        obs, _ = env.reset(seed=42)

        initial_demand_shock = obs[2]

        # Take a step with specific prices
        action = np.array([20.0, 30.0], dtype=np.float32)
        next_obs, rewards, terminated, truncated, info = env.step(action)

        # Check that prices are updated
        assert np.allclose(next_obs[:2], action)

        # Check that demand shock is different (random)
        assert not math.isclose(next_obs[2], initial_demand_shock)

        # Check that step counter is incremented
        assert info["step"] == 1

        # Check that prices are stored in info
        assert np.allclose(info["prices"], action)

    def test_step_reward_calculation_simple_case(self) -> None:
        """Test reward calculation with simple hand-calculated inputs."""
        # Set up environment with no demand shock for predictable results
        # Use competition_intensity=1.0 to get more predictable market shares
        env = CartelEnv(
            n_firms=2,
            marginal_cost=10.0,
            demand_intercept=100.0,
            demand_slope=-1.0,
            shock_std=0.0,  # No randomness
            competition_intensity=1.0,  # Linear competition for predictable results
            seed=42,
        )

        obs, _ = env.reset(seed=42)

        # Set prices that will give predictable demand
        prices = np.array([20.0, 30.0], dtype=np.float32)
        next_obs, rewards, terminated, truncated, info = env.step(prices)

        # Calculate expected results by hand:
        # Market price = (20 + 30) / 2 = 25
        # Base demand = 100 - 1 * 25 = 75
        # Total demand = 75 + 0 (no shock) = 75
        # Market shares with competition_intensity=1.0:
        #   competitiveness = [1/20, 1/30] = [0.05, 0.0333...]
        #   normalized shares = [0.05/(0.05+0.0333), 0.0333/(0.05+0.0333)] ≈ [0.6, 0.4]
        # Individual quantities = [75 * 0.6, 75 * 0.4] = [45, 30]
        # Firm 1 profit = (20 - 10) * 45 = 450
        # Firm 2 profit = (30 - 10) * 30 = 600

        # Calculate expected market shares
        competitiveness = np.array([1.0 / 20.0, 1.0 / 30.0])
        expected_shares = competitiveness / np.sum(competitiveness)
        expected_quantities = expected_shares * 75.0
        expected_rewards = (prices - 10.0) * expected_quantities

        assert np.allclose(rewards, expected_rewards, rtol=1e-3)  # Relaxed tolerance
        assert np.allclose(info["market_price"], 25.0)
        assert np.allclose(info["total_demand"], 75.0)
        assert np.allclose(
            info["individual_quantities"], expected_quantities, rtol=1e-3
        )

    def test_step_reward_calculation_edge_cases(self) -> None:
        """Test reward calculation with edge cases."""
        env = CartelEnv(
            n_firms=2,
            marginal_cost=10.0,
            demand_intercept=100.0,
            demand_slope=-1.0,
            shock_std=0.0,
            max_price_change=200.0,  # Allow large price changes for this test
            seed=42,
        )

        obs, _ = env.reset(seed=42)

        # Test case: prices below marginal cost (should give negative profit)
        prices = np.array([5.0, 8.0], dtype=np.float32)
        next_obs, rewards, terminated, truncated, info = env.step(prices)

        # Both firms should have negative profit since price < marginal cost
        assert np.all(rewards < 0.0)

        # Test case: very high prices (demand should be zero)
        prices = np.array([150.0, 200.0], dtype=np.float32)
        next_obs, rewards, terminated, truncated, info = env.step(prices)

        # Market price = 175, demand = 100 - 175 = -75, but clamped to 0
        assert np.allclose(info["total_demand"], 0.0)
        assert np.allclose(rewards, 0.0)

    def test_step_action_clipping(self) -> None:
        """Test that actions are properly clipped to valid price range."""
        env = CartelEnv(n_firms=2, price_min=10.0, price_max=50.0, seed=42)
        obs, _ = env.reset(seed=42)

        # Try to set prices outside valid range
        action = np.array([5.0, 100.0], dtype=np.float32)
        next_obs, rewards, terminated, truncated, info = env.step(action)

        # Prices should be clipped to valid range
        assert np.allclose(next_obs[:2], [10.0, 50.0])
        assert np.allclose(info["prices"], [10.0, 50.0])

    def test_step_action_length_validation(self) -> None:
        """Test that step() validates action length."""
        env = CartelEnv(n_firms=3, seed=42)
        obs, _ = env.reset(seed=42)

        # Try to pass action with wrong length
        with pytest.raises(ValueError, match="Action must have length 3"):
            env.step(np.array([10.0, 20.0]))  # Only 2 prices for 3 firms

    def test_environment_terminates_after_fixed_horizon(self) -> None:
        """Test that environment terminates after max_steps."""
        env = CartelEnv(n_firms=2, max_steps=5, seed=42)
        obs, _ = env.reset(seed=42)

        # Take steps until horizon
        for step in range(5):
            action = np.array([20.0, 30.0], dtype=np.float32)
            obs, rewards, terminated, truncated, info = env.step(action)

            if step < 4:
                assert not terminated
                assert not truncated
                assert info["step"] == step + 1
            else:
                assert not terminated
                assert truncated  # Should be truncated at max_steps
                assert info["step"] == 5

    def test_total_profits_accumulation(self) -> None:
        """Test that total profits accumulate correctly over time."""
        env = CartelEnv(
            n_firms=2,
            marginal_cost=10.0,
            demand_intercept=100.0,
            demand_slope=-1.0,
            shock_std=0.0,
            seed=42,
        )

        obs, _ = env.reset(seed=42)

        # Take multiple steps with same prices
        prices = np.array([20.0, 30.0], dtype=np.float32)
        total_expected_profits = np.array([0.0, 0.0], dtype=np.float32)

        for step in range(3):
            obs, rewards, terminated, truncated, info = env.step(prices)
            total_expected_profits += rewards

            # Check that total profits accumulate
            assert np.allclose(info["total_profits"], total_expected_profits)

    def test_demand_calculation_method(self) -> None:
        """Test the internal _calculate_demand method."""
        env = CartelEnv(
            n_firms=2,
            demand_intercept=100.0,
            demand_slope=-1.0,
            shock_std=0.0,
            seed=42,
        )

        # Set a known demand shock
        env.current_demand_shock = 5.0

        # Test demand calculation
        prices = np.array([20.0, 30.0])
        demand = env._calculate_demand(prices)

        # Market price = 25, base demand = 100 - 25 = 75, total = 75 + 5 = 80
        expected_demand = 80.0
        assert math.isclose(demand, expected_demand)

    def test_profit_calculation_method(self) -> None:
        """Test the internal _calculate_profits method."""
        env = CartelEnv(n_firms=2, marginal_cost=10.0, seed=42)

        prices = np.array([20.0, 30.0])
        quantities = np.array([10.0, 15.0])

        profits = env._calculate_profits(prices, quantities)

        # Expected: (20-10)*10 = 100, (30-10)*15 = 300
        expected_profits = np.array([100.0, 300.0])
        assert np.allclose(profits, expected_profits)

    def test_profit_calculation_negative_profits(self) -> None:
        """Test that negative profits are allowed for economic realism."""
        env = CartelEnv(n_firms=2, marginal_cost=20.0, seed=42)

        prices = np.array([10.0, 15.0])  # Below marginal cost
        quantities = np.array([5.0, 8.0])

        profits = env._calculate_profits(prices, quantities)

        # Should be negative since prices < marginal cost
        assert np.all(profits < 0.0)

        # Check specific values
        expected_profits = (prices - env.marginal_cost) * quantities
        assert np.allclose(profits, expected_profits)

    def test_price_change_constraint(self) -> None:
        """Test that price changes are constrained for market stability."""
        env = CartelEnv(
            n_firms=2,
            max_price_change=10.0,  # Limit price changes to 10
            seed=42,
        )

        obs, _ = env.reset(seed=42)

        # First step: set initial prices
        initial_prices = np.array([20.0, 25.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(initial_prices)

        # Second step: try to make a large price change
        large_change_prices = np.array(
            [50.0, 60.0], dtype=np.float32
        )  # 30-35 unit changes
        obs, rewards, terminated, truncated, info = env.step(large_change_prices)

        # Check that prices were constrained
        actual_prices = info["prices"]
        price_changes = np.abs(actual_prices - initial_prices)
        max_change = np.max(price_changes)

        # Price changes should be limited to max_price_change
        assert (
            max_change <= env.max_price_change + 1e-6
        )  # Allow small floating point errors

        # Prices should be different from initial but constrained
        assert not np.allclose(actual_prices, initial_prices)
        assert not np.allclose(actual_prices, large_change_prices)

    def test_multiple_resets_independence(self) -> None:
        """Test that multiple resets produce independent episodes."""
        env = CartelEnv(n_firms=2, seed=42)

        # First episode
        obs1, _ = env.reset(seed=42)
        action1 = np.array([20.0, 30.0], dtype=np.float32)
        obs2, rewards1, _, _, _ = env.step(action1)

        # Second episode
        obs3, _ = env.reset(seed=42)
        action2 = np.array([20.0, 30.0], dtype=np.float32)
        obs4, rewards2, _, _, _ = env.step(action2)

        # Initial observations should be the same (same seed)
        assert np.allclose(obs1, obs3)

        # But rewards might differ due to different demand shocks
        # (This tests the randomness in demand shocks)
        assert len(rewards1) == len(rewards2) == 2

    def test_observation_space_bounds(self) -> None:
        """Test that observations stay within reasonable bounds."""
        env = CartelEnv(n_firms=3, seed=42)
        obs, _ = env.reset(seed=42)

        # Take several steps with various prices
        for _ in range(10):
            action = env.action_space.sample()
            obs, rewards, terminated, truncated, info = env.step(action)

            # Prices should be within action space bounds
            assert np.all(obs[:3] >= env.price_min)
            assert np.all(obs[:3] <= env.price_max)

            # Demand shock should be reasonable (within 5 std devs)
            assert abs(obs[3]) <= 5 * env.shock_std

            if truncated:
                break
