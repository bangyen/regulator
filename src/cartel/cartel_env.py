"""
Cartel Environment for Gymnasium.

This module implements a "Regulator vs Cartel" environment where N firms compete
by setting prices, with demand shocks affecting market outcomes. The environment
models oligopolistic competition with constant marginal costs and demand curves.
"""

from typing import Any, Dict, List, Tuple, Union

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
        marginal_costs: Union[List[float], None] = None,  # Allow firm-specific costs
        demand_intercept: float = 100.0,
        demand_slope: float = -1.0,
        shock_std: float = 5.0,
        price_min: float = 1.0,
        price_max: float = 100.0,
        max_price_change: float = 20.0,  # Maximum price change per step for stability
        competition_intensity: float = 2.0,  # How sensitive market share is to price differences
        price_elasticity: float = -1.5,  # Price elasticity of demand
        # New economic model parameters
        product_differentiation: float = 0.1,  # 0=identical products, 1=completely different
        firm_capacities: Union[List[float], None] = None,  # Production capacity limits
        market_share_model: str = "logit",  # "logit" or "inverse_price"
        demand_model: str = "linear",  # "linear", "ces", or "log_linear"
        learning_rate: float = 0.01,  # Rate of parameter adaptation
        seed: Union[int, None] = None,
    ):
        """
        Initialize the Cartel Environment.

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
            price_elasticity: Price elasticity of demand (negative value)
            product_differentiation: Degree of product differentiation (0=identical, 1=completely different)
            firm_capacities: Production capacity limits for each firm (None for unlimited)
            market_share_model: Market share calculation method ("logit" or "inverse_price")
            demand_model: Demand function type ("linear", "ces", or "log_linear")
            learning_rate: Rate of parameter adaptation over time
            seed: Random seed for reproducibility
        """
        super().__init__()

        if n_firms < 1:
            raise ValueError("Number of firms must be at least 1")
        if max_steps < 1:
            raise ValueError("Max steps must be at least 1")
        if marginal_cost < 0:
            raise ValueError("Marginal cost must be non-negative")
        if marginal_costs is not None:
            if len(marginal_costs) != n_firms:
                raise ValueError("marginal_costs must have length equal to n_firms")
            if any(cost < 0 for cost in marginal_costs):
                raise ValueError("All marginal costs must be non-negative")
        if demand_slope >= 0:
            raise ValueError("Demand slope should be negative for realistic demand")
        if shock_std < 0:
            raise ValueError("Shock standard deviation must be non-negative")
        if price_min >= price_max:
            raise ValueError("Price minimum must be less than price maximum")
        if competition_intensity <= 0:
            raise ValueError("Competition intensity must be positive")
        if price_elasticity >= 0:
            raise ValueError("Price elasticity should be negative for realistic demand")
        if not 0.0 <= product_differentiation <= 1.0:
            raise ValueError("Product differentiation must be between 0.0 and 1.0")
        if market_share_model not in ["logit", "inverse_price"]:
            raise ValueError("Market share model must be 'logit' or 'inverse_price'")
        if demand_model not in ["linear", "ces", "log_linear"]:
            raise ValueError("Demand model must be 'linear', 'ces', or 'log_linear'")
        if learning_rate < 0:
            raise ValueError("Learning rate must be non-negative")
        if firm_capacities is not None:
            if len(firm_capacities) != n_firms:
                raise ValueError("firm_capacities must have length equal to n_firms")
            if any(capacity <= 0 for capacity in firm_capacities):
                raise ValueError("All firm capacities must be positive")

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
        self.price_elasticity = price_elasticity

        # New economic model parameters
        self.product_differentiation = product_differentiation
        self.market_share_model = market_share_model
        self.demand_model = demand_model
        self.learning_rate = learning_rate

        # Set up firm-specific marginal costs
        if marginal_costs is not None:
            self.marginal_costs = np.array(marginal_costs, dtype=np.float32)
        else:
            self.marginal_costs = np.full(n_firms, marginal_cost, dtype=np.float32)

        # Set up firm capacities
        if firm_capacities is not None:
            self.firm_capacities = np.array(firm_capacities, dtype=np.float32)
        else:
            self.firm_capacities = np.full(n_firms, float("inf"), dtype=np.float32)

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

        # Apply price change constraints for market stability first
        if self.current_step > 0:  # Skip constraint on first step
            price_changes = np.abs(action - self.previous_prices)
            max_change = np.max(price_changes)

            if max_change > self.max_price_change:
                # Scale down price changes to maintain stability
                scale_factor = self.max_price_change / max_change
                price_deltas = action - self.previous_prices
                constrained_deltas = price_deltas * scale_factor
                action = self.previous_prices + constrained_deltas

        # Clip actions to valid price range
        prices = np.clip(action, self.price_min, self.price_max).astype(np.float32)

        # Update state first (including demand shock for this step)
        self.previous_prices = prices.copy()
        self.current_demand_shock = self.np_random.normal(0, self.shock_std)

        # Calculate market demand using the enhanced method
        market_price = np.mean(prices)  # Average market price
        total_demand = self._calculate_demand(prices)

        # Calculate market shares based on price competition
        market_shares = self._calculate_market_shares(prices)
        individual_quantities = market_shares * total_demand

        # Apply capacity constraints
        if np.any(self.firm_capacities < float("inf")):
            # Check for capacity constraints
            for i in range(self.n_firms):
                if individual_quantities[i] > self.firm_capacities[i]:
                    # Firm is capacity constrained
                    excess_demand = individual_quantities[i] - self.firm_capacities[i]
                    individual_quantities[i] = self.firm_capacities[i]

                    # Redistribute excess demand to other firms proportionally
                    if np.sum(individual_quantities) < total_demand:
                        remaining_capacity = (
                            self.firm_capacities - individual_quantities
                        )
                        if np.sum(remaining_capacity) > 0:
                            # Redistribute proportionally to remaining capacity
                            redistribution_weights = remaining_capacity / np.sum(
                                remaining_capacity
                            )
                            individual_quantities += (
                                redistribution_weights * excess_demand
                            )

        # Calculate profits for each firm using firm-specific costs
        # Formula: (price - marginal_cost_i) * quantity_i
        # Negative profits can occur when:
        # 1. Price < marginal_cost (firm selling at a loss)
        # 2. Very low demand leading to insufficient revenue
        # 3. Fines applied by regulator (handled elsewhere)
        profits = (prices - self.marginal_costs) * individual_quantities
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
            "individual_quantities": individual_quantities,
            "market_shares": market_shares,
            "demand_shock": self.current_demand_shock,
            "total_profits": self.total_profits.copy(),
            # New economic model information
            "market_share_model": self.market_share_model,
            "demand_model": self.demand_model,
            "product_differentiation": self.product_differentiation,
            "firm_capacities": self.firm_capacities.copy(),
            "capacity_utilization": individual_quantities / self.firm_capacities,
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
        Calculate total market demand given prices using improved demand models.

        Args:
            prices: Array of prices set by each firm

        Returns:
            Total market demand
        """
        market_price = np.mean(prices)

        if self.demand_model == "linear":
            # Original linear demand with elasticity
            base_demand = self.demand_intercept + self.demand_slope * market_price

            # Apply price elasticity effect
            baseline_price = self.demand_intercept / abs(self.demand_slope)
            if market_price > baseline_price:
                price_ratio = market_price / baseline_price
                elasticity_factor = price_ratio ** (self.price_elasticity + 1)
                base_demand *= elasticity_factor

        elif self.demand_model == "ces":
            # Constant Elasticity of Substitution demand
            # D = A * (Σ(price_i^(-σ)))^(1/σ) where σ is elasticity of substitution
            sigma = 2.0  # Elasticity of substitution
            A = self.demand_intercept  # Scale parameter

            # Calculate CES demand
            price_sum = np.sum(prices ** (-sigma))
            if price_sum > 0:
                base_demand = A * (price_sum ** (1 / sigma))
            else:
                base_demand = 0.0

        elif self.demand_model == "log_linear":
            # Log-linear demand: log(D) = α + β*log(price) + ε
            alpha = np.log(self.demand_intercept)  # Intercept in log space
            beta = self.demand_slope  # Price elasticity

            if market_price > 0:
                log_demand = alpha + beta * np.log(market_price)
                base_demand = np.exp(log_demand)
            else:
                base_demand = 0.0
        else:
            raise ValueError(f"Unknown demand model: {self.demand_model}")

        total_demand = base_demand + self.current_demand_shock
        return float(max(0, total_demand))

    def _calculate_market_shares(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate market shares based on price competition using improved models.

        Args:
            prices: Array of prices set by each firm

        Returns:
            Array of market shares for each firm
        """
        if self.market_share_model == "logit":
            # Logit model: more realistic market share distribution
            # share_i = exp(-λ * price_i) / Σ exp(-λ * price_j)
            lambda_param = 0.1  # Price sensitivity parameter
            utilities = -lambda_param * prices

            # Numerical stability: subtract max utility
            max_utility = np.max(utilities)
            exp_utilities = np.exp(utilities - max_utility)

            market_shares = exp_utilities / np.sum(exp_utilities)

        elif self.market_share_model == "inverse_price":
            # Original inverse price model (for backward compatibility)
            # Convert prices to competitiveness scores (lower price = higher competitiveness)
            competitiveness: np.ndarray = 1.0 / (prices + 1e-6)

            # Apply competition intensity
            competitiveness = competitiveness**self.competition_intensity

            # Normalize to get market shares
            total_competitiveness = np.sum(competitiveness)
            if total_competitiveness > 0:
                market_shares = competitiveness / total_competitiveness
            else:
                # Fallback to equal shares if all prices are very high
                market_shares = np.ones(self.n_firms, dtype=np.float32) / self.n_firms
        else:
            raise ValueError(f"Unknown market share model: {self.market_share_model}")

        # Apply product differentiation
        if self.product_differentiation > 0:
            # Mix competitive shares with equal shares based on differentiation level
            equal_shares = np.ones(self.n_firms) / self.n_firms
            market_shares = (
                1 - self.product_differentiation
            ) * market_shares + self.product_differentiation * equal_shares

        return market_shares.astype(np.float32)  # type: ignore

    def _calculate_profits(
        self, prices: np.ndarray, quantities: np.ndarray
    ) -> np.ndarray:
        """
        Calculate profits for each firm using firm-specific marginal costs.

        Args:
            prices: Array of prices set by each firm
            quantities: Array of quantities sold by each firm

        Returns:
            Array of profits for each firm

        Note:
            Negative profits are allowed for economic realism. They can occur when:
            - Price < marginal_cost (firm selling at a loss)
            - Very low demand leading to insufficient revenue
            - Fines applied by regulator (handled in penalty application)
        """
        profits = (prices - self.marginal_costs) * quantities
        result: np.ndarray = profits.astype(np.float32)
        return result
