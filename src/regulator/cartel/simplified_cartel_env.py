"""
Simplified Cartel Environment for Gymnasium.

This module implements a streamlined "Regulator vs Cartel" environment focused on
core regulatory analysis. It removes unnecessarily complex features while maintaining
essential economic relationships for studying collusion and regulatory effectiveness.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SimplifiedCartelEnv(gym.Env):
    """
    Simplified Cartel Environment for oligopolistic price competition.

    This environment simulates N firms competing in a market where each firm
    chooses a price, and market demand is affected by price levels and random
    demand shocks. Firms have constant marginal costs and compete for profits.

    Key simplifications:
    - Simple price-based market share allocation
    - Fixed + variable cost structure
    - Linear demand with shocks
    - Optional capacity constraints
    - No information asymmetry, learning curves, or dynamic elasticity
    """

    def __init__(
        self,
        n_firms: int = 3,
        max_steps: int = 100,
        marginal_cost: float = 10.0,
        marginal_costs: Union[List[float], None] = None,
        demand_intercept: float = 100.0,
        demand_slope: float = -1.0,
        shock_std: float = 5.0,
        price_min: float = 1.0,
        price_max: float = 100.0,
        max_price_change: float = 20.0,
        competition_intensity: float = 2.0,
        # Simplified economic model parameters
        use_fixed_costs: bool = True,
        fixed_cost: float = 50.0,
        use_capacity_constraints: bool = False,
        capacity: Union[List[float], None] = None,
        seed: Union[int, None] = None,
    ):
        """
        Initialize the Simplified Cartel Environment.

        Args:
            n_firms: Number of firms in the market
            max_steps: Maximum number of steps before episode termination
            marginal_cost: Constant marginal cost for all firms (used if marginal_costs is None)
            marginal_costs: Firm-specific marginal costs (list of length n_firms) or None
            demand_intercept: Intercept of the demand curve D(p) = a + b*p
            demand_slope: Slope of the demand curve (should be negative)
            shock_std: Standard deviation of demand shocks
            price_min: Minimum allowed price
            price_max: Maximum allowed price
            max_price_change: Maximum price change per step for market stability
            competition_intensity: How sensitive market share is to price differences
            use_fixed_costs: Enable fixed costs per period
            fixed_cost: Per-period fixed cost per firm
            use_capacity_constraints: Enable production capacity limits
            capacity: Production capacity per firm (list of length n_firms) or None
            seed: Random seed for reproducibility
        """
        super().__init__()

        if n_firms < 1:
            raise ValueError("n_firms must be at least 1")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if marginal_cost <= 0:
            raise ValueError("marginal_cost must be positive")
        if demand_slope >= 0:
            raise ValueError(
                "demand_slope must be negative for downward-sloping demand"
            )
        if shock_std < 0:
            raise ValueError("shock_std must be non-negative")
        if price_min >= price_max:
            raise ValueError("price_min must be less than price_max")
        if max_price_change <= 0:
            raise ValueError("max_price_change must be positive")
        if competition_intensity <= 0:
            raise ValueError("competition_intensity must be positive")
        if fixed_cost < 0:
            raise ValueError("fixed_cost must be non-negative")
        if capacity is not None:
            if len(capacity) != n_firms:
                raise ValueError("capacity must have length equal to n_firms")
            if any(cap <= 0 for cap in capacity):
                raise ValueError("All capacity values must be positive")

        self.n_firms = n_firms
        self.max_steps = max_steps
        self.marginal_cost = marginal_cost
        self.demand_intercept = demand_intercept
        self.demand_slope = demand_slope
        self.shock_std = shock_std
        self.price_min = price_min
        self.price_max = price_max
        self.max_price_change = max_price_change
        self.competition_intensity = competition_intensity

        # Simplified economic model parameters
        self.use_fixed_costs = use_fixed_costs
        self.fixed_cost = fixed_cost
        self.use_capacity_constraints = use_capacity_constraints
        self.capacity = capacity

        # Set up marginal costs
        if marginal_costs is not None:
            if len(marginal_costs) != n_firms:
                raise ValueError("marginal_costs must have length equal to n_firms")
            if any(mc <= 0 for mc in marginal_costs):
                raise ValueError("All marginal costs must be positive")
            self.marginal_costs = np.array(marginal_costs, dtype=np.float32)
        else:
            self.marginal_costs = np.full(n_firms, marginal_cost, dtype=np.float32)

        # Set up capacity constraints
        if self.use_capacity_constraints and capacity is not None:
            self.capacity_array: Optional[np.ndarray] = np.array(
                capacity, dtype=np.float32
            )
        else:
            self.capacity_array = None

        # Initialize random number generator
        self.np_random = np.random.default_rng(seed)

        # Action space: each firm chooses a price
        self.action_space = spaces.Box(
            low=price_min,
            high=price_max,
            shape=(n_firms,),
            dtype=np.float32,
        )

        # Observation space: prices, profits, market price, total demand
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_firms * 2 + 2,),  # prices + profits + market_price + total_demand
            dtype=np.float32,
        )

        # Episode state
        self.current_step = 0
        self.current_demand_shock = 0.0
        self.previous_prices: Optional[np.ndarray] = None

    def reset(
        self, seed: Union[int, None] = None, options: Union[Dict[str, Any], None] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.current_step = 0
        self.current_demand_shock = float(self.np_random.normal(0, self.shock_std))
        self.previous_prices = None

        # Initial observation (all zeros)
        obs_shape = self.observation_space.shape
        if obs_shape is not None:
            obs = np.zeros(obs_shape, dtype=np.float32)
        else:
            obs = np.zeros((self.n_firms * 2 + 2,), dtype=np.float32)

        info = {
            "step": self.current_step,
            "demand_shock": self.current_demand_shock,
            "total_demand": 0.0,
            "market_price": 0.0,
            "prices": np.zeros(self.n_firms, dtype=np.float32),
            "profits": np.zeros(self.n_firms, dtype=np.float32),
        }

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not in action space")

        # Clip prices to valid range
        prices = np.clip(action, self.price_min, self.price_max)

        # Apply price change limits for stability
        if self.previous_prices is not None:
            price_changes = np.abs(prices - self.previous_prices)
            max_change_mask = price_changes > self.max_price_change
            if np.any(max_change_mask):
                # Limit price changes to max_price_change
                direction = np.sign(prices - self.previous_prices)
                prices[max_change_mask] = (
                    self.previous_prices[max_change_mask]
                    + direction[max_change_mask] * self.max_price_change
                )

        # Calculate market outcomes
        market_price = np.mean(prices)
        total_demand = self._calculate_demand(prices)
        market_shares = self._calculate_market_shares(prices)
        quantities = total_demand * market_shares

        # Apply capacity constraints if enabled
        if self.use_capacity_constraints and self.capacity_array is not None:
            quantities = np.minimum(quantities, self.capacity_array)
            # Recalculate market shares after capacity constraints
            total_quantity: float = float(np.sum(quantities))
            if total_quantity > 0:
                market_shares = quantities / total_quantity
            else:
                market_shares = np.ones(self.n_firms) / self.n_firms

        # Calculate costs and profits
        costs = self._calculate_costs(quantities)
        profits = prices * quantities - costs

        # Create observation
        obs = np.concatenate(
            [prices, profits, [market_price, total_demand]], dtype=np.float32
        )

        # Update state
        self.current_step += 1
        self.previous_prices = prices.copy()
        self.current_demand_shock = float(self.np_random.normal(0, self.shock_std))

        # Create info dictionary
        info = {
            "step": self.current_step,
            "prices": prices.copy(),
            "profits": profits.copy(),
            "market_price": market_price,
            "total_demand": total_demand,
            "individual_quantity": quantities.copy(),
            "market_shares": market_shares.copy(),
            "demand_shock": self.current_demand_shock,
            "costs": costs.copy(),
        }

        # Check termination conditions
        terminated = self.current_step >= self.max_steps
        truncated = False

        return obs, profits, terminated, truncated, info

    def _calculate_demand(self, prices: np.ndarray) -> float:
        """
        Calculate total market demand given prices.

        Args:
            prices: Array of prices set by each firm

        Returns:
            Total market demand
        """
        market_price = np.mean(prices)

        # Linear demand curve: D(p) = a + b*p
        base_demand = self.demand_intercept + self.demand_slope * market_price

        # Add demand shock
        total_demand = base_demand + self.current_demand_shock

        # Ensure non-negative demand
        return float(max(0, total_demand))

    def _calculate_market_shares(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate market shares based on price competitiveness.

        Args:
            prices: Array of prices set by each firm

        Returns:
            Array of market shares for each firm
        """
        # Simple price-based competitiveness: lower price = higher competitiveness
        competitiveness = 1.0 / (
            prices + 1e-6
        )  # Add small epsilon to avoid division by zero

        # Apply competition intensity
        competitiveness = competitiveness**self.competition_intensity

        # Normalize to get market shares
        total_competitiveness: float = float(np.sum(competitiveness))
        if total_competitiveness > 0:
            market_shares = competitiveness / total_competitiveness
        else:
            # Fallback to equal shares if all prices are very high
            market_shares = np.ones(self.n_firms, dtype=np.float32) / self.n_firms

        result: np.ndarray = market_shares.astype(np.float32)
        return result

    def _calculate_costs(self, quantities: np.ndarray) -> np.ndarray:
        """
        Calculate total costs for each firm.

        Args:
            quantities: Array of quantities sold by each firm

        Returns:
            Array of total costs for each firm
        """
        # Variable costs based on marginal costs
        variable_costs = self.marginal_costs * quantities

        # Add fixed costs if enabled
        if self.use_fixed_costs:
            fixed_costs = np.full(self.n_firms, self.fixed_cost, dtype=np.float32)
            total_costs = fixed_costs + variable_costs
        else:
            total_costs = variable_costs

        result: np.ndarray = total_costs.astype(np.float32)
        return result

    def render(self, mode: str = "human") -> None:
        """Render the environment (placeholder)."""
        if mode == "human":
            if self.previous_prices is not None:
                print(
                    f"Step {self.current_step}: Market price = {np.mean(self.previous_prices):.2f}"
                )

    def close(self) -> None:
        """Close the environment (placeholder)."""
        pass
