"""
Cartel Environment for Gymnasium.

This module implements a "Regulator vs Cartel" environment where N firms compete
by setting prices, with demand shocks affecting market outcomes. The environment
models oligopolistic competition with constant marginal costs and demand curves.
"""

from typing import Any, Dict, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CartelEnv(gym.Env):
    """
    Cartel Environment for oligopolistic price competition.

    This environment simulates N firms competing in a market where each firm
    chooses a price, and market demand is affected by price levels and random
    demand shocks. Firms have constant marginal costs and compete for profits.
    """

    def __init__(
        self,
        n_firms: int = 3,
        max_steps: int = 100,
        marginal_cost: float = 10.0,
        demand_intercept: float = 100.0,
        demand_slope: float = -1.0,
        shock_std: float = 5.0,
        price_min: float = 1.0,
        price_max: float = 100.0,
        seed: Union[int, None] = None,
    ):
        """
        Initialize the Cartel Environment.

        Args:
            n_firms: Number of firms in the market
            max_steps: Maximum number of steps before episode termination
            marginal_cost: Constant marginal cost for all firms
            demand_intercept: Intercept of the demand curve D(p) = a + b*p
            demand_slope: Slope of the demand curve (should be negative)
            shock_std: Standard deviation of demand shocks
            price_min: Minimum allowed price
            price_max: Maximum allowed price
            seed: Random seed for reproducibility
        """
        super().__init__()

        if n_firms < 1:
            raise ValueError("Number of firms must be at least 1")
        if max_steps < 1:
            raise ValueError("Max steps must be at least 1")
        if marginal_cost < 0:
            raise ValueError("Marginal cost must be non-negative")
        if demand_slope >= 0:
            raise ValueError("Demand slope should be negative for realistic demand")
        if shock_std < 0:
            raise ValueError("Shock standard deviation must be non-negative")
        if price_min >= price_max:
            raise ValueError("Price minimum must be less than price maximum")

        self.n_firms = n_firms
        self.max_steps = max_steps
        self.marginal_cost = marginal_cost
        self.demand_intercept = demand_intercept
        self.demand_slope = demand_slope
        self.shock_std = shock_std
        self.price_min = price_min
        self.price_max = price_max

        # Initialize random number generator
        self.np_random = np.random.default_rng(seed)

        # Action space: each firm chooses a price
        self.action_space = spaces.Box(
            low=price_min,
            high=price_max,
            shape=(n_firms,),
            dtype=np.float32,
        )

        # Observation space: previous prices + current demand shock
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_firms + 1,),  # n_firms prices + 1 demand shock
            dtype=np.float32,
        )

        # State variables
        self.current_step = 0
        self.previous_prices = np.zeros(n_firms, dtype=np.float32)
        self.current_demand_shock = 0.0
        self.total_profits = np.zeros(n_firms, dtype=np.float32)

    def reset(
        self, seed: Union[int, None] = None, options: Union[Dict[str, Any], None] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Initial observation and info dictionary
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Reset state variables
        self.current_step = 0
        self.previous_prices = np.zeros(self.n_firms, dtype=np.float32)
        self.current_demand_shock = self.np_random.normal(0, self.shock_std)
        self.total_profits = np.zeros(self.n_firms, dtype=np.float32)

        # Create initial observation
        observation = np.concatenate(
            [self.previous_prices, [self.current_demand_shock]]
        ).astype(np.float32)

        info = {
            "step": self.current_step,
            "demand_shock": self.current_demand_shock,
            "total_profits": self.total_profits.copy(),
        }

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Array of prices chosen by each firm

        Returns:
            observation: Next state observation
            rewards: Profit for each firm
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Validate action
        if len(action) != self.n_firms:
            raise ValueError(f"Action must have length {self.n_firms}")

        # Clip actions to valid price range
        prices = np.clip(action, self.price_min, self.price_max).astype(np.float32)

        # Calculate market demand using the consistent method
        market_price = np.mean(prices)  # Average market price
        total_demand = self._calculate_demand(prices)

        # Allocate demand among firms (equal split for simplicity)
        individual_quantity = total_demand / float(self.n_firms)

        # Calculate profits for each firm (allow negative profits for economic realism)
        profits = (prices - self.marginal_cost) * individual_quantity
        # Remove profit flooring to show true economic outcomes
        # profits = np.maximum(profits, 0)  # Ensure non-negative profits

        # Update state
        self.previous_prices = prices.copy()
        self.current_demand_shock = self.np_random.normal(0, self.shock_std)
        self.total_profits += profits
        self.current_step += 1

        # Create next observation
        observation = np.concatenate(
            [self.previous_prices, [self.current_demand_shock]]
        ).astype(np.float32)

        # Check termination conditions
        terminated = False  # No natural termination condition
        truncated = self.current_step >= self.max_steps

        info = {
            "step": self.current_step,
            "prices": prices.copy(),
            "market_price": market_price,
            "total_demand": total_demand,
            "individual_quantity": individual_quantity,
            "demand_shock": self.current_demand_shock,
            "total_profits": self.total_profits.copy(),
        }

        return observation, profits, terminated, truncated, info

    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass

    def close(self) -> None:
        """Close the environment (not implemented)."""
        pass

    def _calculate_demand(self, prices: np.ndarray) -> float:
        """
        Calculate total market demand given prices.

        Args:
            prices: Array of prices set by each firm

        Returns:
            Total market demand
        """
        market_price = np.mean(prices)
        base_demand = self.demand_intercept + self.demand_slope * market_price
        total_demand = base_demand + self.current_demand_shock
        return float(max(0, total_demand))

    def _calculate_profits(
        self, prices: np.ndarray, quantities: np.ndarray
    ) -> np.ndarray:
        """
        Calculate profits for each firm.

        Args:
            prices: Array of prices set by each firm
            quantities: Array of quantities sold by each firm

        Returns:
            Array of profits for each firm
        """
        profits = (prices - self.marginal_cost) * quantities
        # Allow negative profits for economic realism
        # result: np.ndarray = np.maximum(profits, 0)  # Ensure non-negative profits
        result: np.ndarray = profits.astype(np.float32)
        return result
