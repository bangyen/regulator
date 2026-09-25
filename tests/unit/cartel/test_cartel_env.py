"""
Unit tests for the minimalist CartelEnv environment.
"""

import math

import numpy as np

from regulator.cartel.cartel_env import CartelEnv


class TestCartelEnv:
    """Minimalist test suite for CartelEnv."""

    def test_initialization_default_params(self) -> None:
        """Test initialization with default parameters."""
        env = CartelEnv()
        assert env.n_firms == 3
        assert env.max_steps == 100
        assert env.marginal_cost == 10.0
        assert len(env.marginal_costs) == 3
        assert np.allclose(env.marginal_costs, 10.0)

    def test_initialization_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        env = CartelEnv(n_firms=5, max_steps=50, marginal_cost=15.0)
        assert env.n_firms == 5
        assert env.max_steps == 50
        assert np.allclose(env.marginal_costs, 15.0)

    def test_reset_returns_consistent_obs(self) -> None:
        """Test that reset() returns observations of correct shape."""
        env = CartelEnv(n_firms=3)
        obs, info = env.reset(seed=42)
        assert obs.shape == (4,)  # 3 prices + 1 demand shock
        assert np.all(obs[:3] == 0.0)
        assert info["step"] == 0

    def test_step_logic(self) -> None:
        """Test a single step of the environment."""
        env = CartelEnv(n_firms=2, seed=42)
        env.reset(seed=42)
        action = np.array([20.0, 30.0], dtype=np.float32)
        next_obs, rewards, terminated, truncated, info = env.step(action)

        assert np.allclose(next_obs[:2], action)
        assert len(rewards) == 2
        assert info["step"] == 1
        assert "market_price" in info
        assert "total_demand" in info

    def test_action_clipping(self) -> None:
        """Test that prices are clipped to valid ranges."""
        env = CartelEnv(n_firms=2, price_min=10.0, price_max=50.0)
        env.reset()
        action = np.array([5.0, 100.0], dtype=np.float32)
        next_obs, rewards, terminated, truncated, info = env.step(action)
        assert np.allclose(next_obs[:2], [10.0, 50.0])

    def test_termination(self) -> None:
        """Test environment truncation at max steps."""
        env = CartelEnv(n_firms=2, max_steps=2)
        env.reset()
        action = np.array([20.0, 20.0], dtype=np.float32)
        _, _, _, truncated, _ = env.step(action)
        assert not truncated
        _, _, _, truncated, _ = env.step(action)
        assert truncated

    def test_market_shares(self) -> None:
        """Test that market shares sum to 1 and favor lower prices."""
        env = CartelEnv(n_firms=2)
        # Low price, High price
        prices = np.array([10.0, 20.0])
        shares = env._calculate_market_shares(prices)
        assert shares[0] > shares[1]
        assert math.isclose(np.sum(shares), 1.0)


class TestInformationFrictions:
    """Tests for observation noise and limited price visibility."""

    def test_disabled_returns_exact_copy(self) -> None:
        env = CartelEnv(n_firms=3, seed=0)
        prices = np.array([10.0, 20.0, 30.0])

        observed = env.get_observed_prices(prices)

        np.testing.assert_array_equal(observed, prices)
        assert observed is not prices

    def test_full_visibility_hides_nothing(self) -> None:
        env = CartelEnv(
            n_firms=3,
            seed=0,
            use_information_asymmetry=True,
            price_visibility_prob=1.0,
            observation_noise_std=0.0,
        )
        prices = np.array([10.0, 20.0, 30.0])

        np.testing.assert_allclose(env.get_observed_prices(prices), prices)

    def test_zero_visibility_hides_everything(self) -> None:
        env = CartelEnv(
            n_firms=3, seed=0, use_information_asymmetry=True, price_visibility_prob=0.0
        )

        assert np.isnan(env.get_observed_prices(np.array([10.0, 20.0, 30.0]))).all()

    def test_hidden_rate_independent_of_firm_count(self) -> None:
        """Each price is hidden with probability 1 - p, regardless of n_firms."""
        env = CartelEnv(
            n_firms=5, seed=0, use_information_asymmetry=True, price_visibility_prob=0.8
        )
        prices = np.full(5, 20.0)

        hidden = np.mean(
            [np.isnan(env.get_observed_prices(prices)).mean() for _ in range(2000)]
        )

        # Old behaviour hid 1 - 0.8**4 ≈ 59% of prices with 5 firms
        assert abs(hidden - 0.2) < 0.02
