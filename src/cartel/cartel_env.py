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
        # Enhanced economic model parameters
        use_enhanced_demand: bool = False,  # Enable constant elasticity demand
        use_logit_market_shares: bool = False,  # Enable logit-based market shares
        use_fixed_costs: bool = False,  # Enable fixed costs
        fixed_cost: float = 50.0,  # Per-period fixed cost per firm
        use_economies_of_scale: bool = False,  # Enable economies of scale
        scale_elasticity: float = 0.8,  # Cost reduction with scale (β < 1)
        use_information_asymmetry: bool = False,  # Enable information frictions
        observation_noise_std: float = 0.1,  # Standard deviation of price observation noise
        price_visibility_prob: float = 0.8,  # Probability of observing competitor prices
        # New economic model improvements
        use_dynamic_elasticity: bool = False,  # Enable dynamic price elasticity
        elasticity_sensitivity: float = 0.5,  # How much elasticity varies with market conditions
        use_capacity_constraints: bool = False,  # Enable production capacity limits
        capacity: Union[List[float], None] = None,  # Production capacity per firm
        use_learning_agents: bool = False,  # Enable adaptive learning in agents
        learning_rate: float = 0.1,  # Learning rate for adaptive agents
        use_market_entry_exit: bool = False,  # Enable dynamic firm entry/exit
        exit_threshold: float = -100.0,  # Profit threshold for firm exit
        max_consecutive_losses: int = 5,  # Max losses before exit
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
            use_enhanced_demand: Enable constant elasticity demand function
            use_logit_market_shares: Enable logit-based market share allocation
            use_fixed_costs: Enable fixed costs per period
            fixed_cost: Per-period fixed cost per firm
            use_economies_of_scale: Enable economies of scale in cost function
            scale_elasticity: Cost reduction with scale (β < 1 for economies of scale)
            use_information_asymmetry: Enable information frictions
            observation_noise_std: Standard deviation of price observation noise
            price_visibility_prob: Probability of observing competitor prices
            use_dynamic_elasticity: Enable dynamic price elasticity based on market conditions
            elasticity_sensitivity: How much elasticity varies with market conditions (0.0-1.0)
            use_capacity_constraints: Enable production capacity limits per firm
            capacity: Production capacity per firm (list of length n_firms) or None
            use_learning_agents: Enable adaptive learning in firm agents
            learning_rate: Learning rate for adaptive agents (0.0-1.0)
            use_market_entry_exit: Enable dynamic firm entry and exit
            exit_threshold: Profit threshold below which firms consider exiting
            max_consecutive_losses: Maximum consecutive loss periods before exit
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
        if fixed_cost < 0:
            raise ValueError("Fixed cost must be non-negative")
        if scale_elasticity <= 0 or scale_elasticity > 1:
            raise ValueError(
                "Scale elasticity must be between 0 and 1 for economies of scale"
            )
        if observation_noise_std < 0:
            raise ValueError(
                "Observation noise standard deviation must be non-negative"
            )
        if price_visibility_prob < 0 or price_visibility_prob > 1:
            raise ValueError("Price visibility probability must be between 0 and 1")

        # Validate new economic model parameters
        if not 0.0 <= elasticity_sensitivity <= 1.0:
            raise ValueError("Elasticity sensitivity must be between 0.0 and 1.0")
        if capacity is not None:
            if len(capacity) != n_firms:
                raise ValueError("capacity must have length equal to n_firms")
            if any(cap <= 0 for cap in capacity):
                raise ValueError("All capacity values must be positive")
        if not 0.0 <= learning_rate <= 1.0:
            raise ValueError("Learning rate must be between 0.0 and 1.0")
        if max_consecutive_losses < 1:
            raise ValueError("Max consecutive losses must be at least 1")

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

        # Enhanced economic model parameters
        self.use_enhanced_demand = use_enhanced_demand
        self.use_logit_market_shares = use_logit_market_shares
        self.use_fixed_costs = use_fixed_costs
        self.fixed_cost = fixed_cost
        self.use_economies_of_scale = use_economies_of_scale
        self.scale_elasticity = scale_elasticity
        self.use_information_asymmetry = use_information_asymmetry
        self.observation_noise_std = observation_noise_std
        self.price_visibility_prob = price_visibility_prob

        # New economic model improvements
        self.use_dynamic_elasticity = use_dynamic_elasticity
        self.elasticity_sensitivity = elasticity_sensitivity
        self.use_capacity_constraints = use_capacity_constraints
        self.use_learning_agents = use_learning_agents
        self.learning_rate = learning_rate
        self.use_market_entry_exit = use_market_entry_exit
        self.exit_threshold = exit_threshold
        self.max_consecutive_losses = max_consecutive_losses

        # Set up firm-specific marginal costs
        if marginal_costs is not None:
            self.marginal_costs = np.array(marginal_costs, dtype=np.float32)
        else:
            self.marginal_costs = np.full(n_firms, marginal_cost, dtype=np.float32)

        # Set up capacity constraints
        if capacity is not None:
            self.capacity = np.array(capacity, dtype=np.float32)
        else:
            self.capacity = np.full(n_firms, np.inf, dtype=np.float32)

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

        # New economic model state variables
        self.active_firms = np.ones(n_firms, dtype=bool)  # Track which firms are active
        self.consecutive_losses = np.zeros(
            n_firms, dtype=int
        )  # Track consecutive losses
        self.price_history: List[float] = (
            []
        )  # Store price history for dynamic elasticity
        self.demand_history: List[float] = (
            []
        )  # Store demand history for dynamic elasticity
        self.current_elasticity = price_elasticity  # Current dynamic elasticity

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

        # Reset new economic model state variables
        self.active_firms = np.ones(self.n_firms, dtype=bool)
        self.consecutive_losses = np.zeros(self.n_firms, dtype=int)
        self.price_history = []
        self.demand_history = []
        self.current_elasticity = self.price_elasticity

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
        individual_quantities = self._apply_capacity_constraints(individual_quantities)

        # Calculate profits for each firm using enhanced cost structure
        # Formula: revenue - total_costs (including fixed costs and economies of scale)
        # Negative profits can occur when:
        # 1. Price < marginal_cost (firm selling at a loss)
        # 2. Fixed costs exceed revenue
        # 3. Very low demand leading to insufficient revenue
        # 4. Fines applied by regulator (handled elsewhere)
        profits = self._calculate_profits(prices, individual_quantities)

        # Check firm viability (exit due to sustained losses)
        self._check_firm_viability(profits)

        self.total_profits += profits
        self.current_step += 1

        # Create next observation
        observation = np.concatenate(
            [self.previous_prices, [self.current_demand_shock]]
        ).astype(np.float32)

        # Check termination conditions
        terminated = False  # No natural termination condition
        truncated = self.current_step >= self.max_steps

        # Calculate observed prices with information frictions
        observed_prices = self.get_observed_prices(prices)

        info = {
            "step": self.current_step,
            "prices": prices.copy(),
            "observed_prices": observed_prices,
            "market_price": market_price,
            "total_demand": total_demand,
            "individual_quantities": individual_quantities,
            "market_shares": market_shares,
            "demand_shock": self.current_demand_shock,
            "total_profits": self.total_profits.copy(),
            "use_enhanced_demand": self.use_enhanced_demand,
            "use_logit_market_shares": self.use_logit_market_shares,
            "use_fixed_costs": self.use_fixed_costs,
            "use_economies_of_scale": self.use_economies_of_scale,
            "use_information_asymmetry": self.use_information_asymmetry,
            # New economic model features
            "use_dynamic_elasticity": self.use_dynamic_elasticity,
            "current_elasticity": self.current_elasticity,
            "use_capacity_constraints": self.use_capacity_constraints,
            "capacity": self.capacity.copy(),
            "use_market_entry_exit": self.use_market_entry_exit,
            "active_firms": self.active_firms.copy(),
            "consecutive_losses": self.consecutive_losses.copy(),
        }

        return observation, profits, terminated, truncated, info

    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass

    def close(self) -> None:
        """Close the environment (not implemented)."""
        pass

    def _calculate_dynamic_elasticity(
        self, market_price: float, total_demand: float
    ) -> float:
        """
        Calculate dynamic price elasticity based on market conditions.

        Args:
            market_price: Current market price
            total_demand: Current total demand

        Returns:
            Dynamic price elasticity value
        """
        if not self.use_dynamic_elasticity:
            return self.price_elasticity

        # Base elasticity
        base_elasticity = self.price_elasticity

        # Price factor: higher elasticity when prices are high (consumers more price-sensitive)
        baseline_price = self.demand_intercept / abs(self.demand_slope)
        price_factor = (
            1.0 + (market_price / baseline_price) * self.elasticity_sensitivity
        )

        # Demand factor: higher elasticity when demand is low (inelastic demand)
        baseline_demand = self.demand_intercept
        demand_factor = (
            1.0 + (1.0 - total_demand / baseline_demand) * self.elasticity_sensitivity
        )

        # Market volatility factor: higher elasticity during volatile periods
        volatility_factor = 1.0
        if len(self.price_history) >= 5:
            recent_prices = [p for p in self.price_history[-5:]]
            price_volatility = (
                np.std(recent_prices) / np.mean(recent_prices)
                if np.mean(recent_prices) > 0
                else 0
            )
            volatility_factor = (
                1.0 + float(price_volatility) * self.elasticity_sensitivity
            )

        # Calculate dynamic elasticity (more negative = more elastic)
        dynamic_elasticity = (
            base_elasticity * price_factor * demand_factor * volatility_factor
        )

        # Ensure elasticity stays within reasonable bounds
        min_elasticity = self.price_elasticity * 0.5  # At least 50% of base
        max_elasticity = self.price_elasticity * 2.0  # At most 200% of base

        return max(min_elasticity, min(max_elasticity, dynamic_elasticity))

    def _apply_capacity_constraints(self, quantities: np.ndarray) -> np.ndarray:
        """
        Apply production capacity constraints to firm quantities.

        Args:
            quantities: Array of quantities each firm wants to produce

        Returns:
            Array of quantities after applying capacity constraints
        """
        if not self.use_capacity_constraints:
            return quantities

        # Apply capacity constraints
        constrained_quantities = np.minimum(quantities, self.capacity)

        # If total constrained demand is less than original demand,
        # redistribute the shortfall proportionally among firms
        total_original = np.sum(quantities)
        total_constrained = np.sum(constrained_quantities)

        if total_constrained < total_original and total_constrained > 0:
            # Redistribute the shortfall proportionally
            shortfall = total_original - total_constrained
            redistribution = shortfall * (constrained_quantities / total_constrained)
            constrained_quantities += redistribution

        return constrained_quantities

    def _check_firm_viability(self, profits: np.ndarray) -> None:
        """
        Check if firms should exit due to sustained losses.

        Args:
            profits: Array of current period profits for each firm
        """
        if not self.use_market_entry_exit:
            return

        for i, profit in enumerate(profits):
            if not self.active_firms[i]:  # Skip inactive firms
                continue

            if profit < self.exit_threshold:
                self.consecutive_losses[i] += 1
                if self.consecutive_losses[i] >= self.max_consecutive_losses:
                    # Firm exits market
                    self.active_firms[i] = False
                    print(f"Firm {i} exited market due to sustained losses")
            else:
                self.consecutive_losses[i] = 0

    def _calculate_demand(self, prices: np.ndarray) -> float:
        """
        Calculate total market demand given prices with enhanced elasticity.

        Args:
            prices: Array of prices set by each firm

        Returns:
            Total market demand
        """
        market_price = np.mean(prices)

        # Calculate initial demand
        if self.use_enhanced_demand:
            # Constant elasticity demand: D(p) = A * p^(-ε)
            # where A is demand_intercept and ε is price_elasticity
            if market_price > 0:
                base_demand = self.demand_intercept * (
                    market_price**self.current_elasticity
                )
            else:
                base_demand = 0.0
        else:
            # Original linear demand curve
            base_demand = self.demand_intercept + self.demand_slope * market_price

            # Apply price elasticity effect for linear demand
            # If prices are higher than baseline, demand decreases more than linearly
            baseline_price = self.demand_intercept / abs(
                self.demand_slope
            )  # Price where demand = 0
            if market_price > baseline_price:
                # Apply elasticity penalty for high prices
                price_ratio = market_price / baseline_price
                elasticity_factor = price_ratio ** (
                    self.current_elasticity + 1
                )  # +1 to account for linear term
                base_demand *= elasticity_factor

        total_demand = base_demand + self.current_demand_shock
        total_demand = float(max(0, total_demand))

        # Update dynamic elasticity for next period
        if self.use_dynamic_elasticity:
            self.current_elasticity = self._calculate_dynamic_elasticity(
                market_price, total_demand
            )

        # Store history for dynamic elasticity calculation
        self.price_history.append(market_price)
        self.demand_history.append(total_demand)

        # Keep only recent history to avoid memory issues
        if len(self.price_history) > 20:
            self.price_history = self.price_history[-20:]
            self.demand_history = self.demand_history[-20:]

        return total_demand

    def _calculate_market_shares(
        self, prices: np.ndarray
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Calculate market shares based on price competition.

        Args:
            prices: Array of prices set by each firm

        Returns:
            Array of market shares for each firm
        """
        if self.use_logit_market_shares:
            # Logit-based market shares: s_i = exp(-λ*p_i) / Σ_j exp(-λ*p_j)
            # where λ is the price sensitivity parameter
            lambda_param = 0.1  # Price sensitivity (can be made configurable)
            utilities = -lambda_param * prices

            # Numerical stability: subtract max utility to prevent overflow
            max_utility = np.max(utilities)
            exp_utilities = np.exp(utilities - max_utility)

            # Normalize to get market shares
            total_utility = np.sum(exp_utilities)
            if total_utility > 0:
                market_shares: np.ndarray = exp_utilities / total_utility
            else:
                # Fallback to equal shares
                market_shares = np.ones(self.n_firms, dtype=np.float32) / self.n_firms
        else:
            # Original competitiveness-based market shares
            # Convert prices to competitiveness scores (lower price = higher competitiveness)
            # Add small epsilon to avoid division by zero
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

        return market_shares.astype(np.float32)

    def _calculate_costs(
        self, quantities: np.ndarray
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Calculate total costs for each firm including fixed costs and economies of scale.

        Args:
            quantities: Array of quantities sold by each firm

        Returns:
            Array of total costs for each firm
        """
        # Variable costs based on marginal costs
        variable_costs = self.marginal_costs * quantities

        if self.use_economies_of_scale:
            # Apply economies of scale: c(q) = c₀ * q^β where β < 1
            # Scale factor reduces costs at higher volumes
            scale_factor = np.power(quantities + 1e-6, self.scale_elasticity - 1.0)
            variable_costs = variable_costs * scale_factor

        # Add fixed costs if enabled
        if self.use_fixed_costs:
            fixed_costs = np.full(self.n_firms, self.fixed_cost, dtype=np.float32)
            total_costs = fixed_costs + variable_costs
        else:
            total_costs = variable_costs

        result: np.ndarray[Any, np.dtype[np.float32]] = total_costs.astype(np.float32)
        return result

    def _calculate_profits(
        self, prices: np.ndarray, quantities: np.ndarray
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Calculate profits for each firm using enhanced cost structure.

        Args:
            prices: Array of prices set by each firm
            quantities: Array of quantities sold by each firm

        Returns:
            Array of profits for each firm

        Note:
            Negative profits are allowed for economic realism. They can occur when:
            - Price < marginal_cost (firm selling at a loss)
            - Fixed costs exceed revenue
            - Very low demand leading to insufficient revenue
            - Fines applied by regulator (handled in penalty application)
        """
        revenues = prices * quantities
        costs = self._calculate_costs(quantities)
        profits = revenues - costs
        result: np.ndarray[Any, np.dtype[np.float32]] = profits.astype(np.float32)
        return result

    def _add_information_frictions(self, prices: np.ndarray) -> np.ndarray:
        """
        Add information frictions to price observations.

        Args:
            prices: Array of actual prices set by each firm

        Returns:
            Array of observed prices with noise and limited visibility
        """
        if not self.use_information_asymmetry:
            return prices.copy()

        # Add observation noise
        observation_noise = self.np_random.normal(
            0, self.observation_noise_std, prices.shape
        )
        observed_prices = prices + observation_noise

        # Apply limited visibility - firms may not observe all competitor prices
        visibility_mask = (
            self.np_random.random(prices.shape) < self.price_visibility_prob
        )
        # Always allow firms to observe their own prices
        visibility_mask = np.ones_like(visibility_mask, dtype=bool)

        # For competitors, apply visibility probability
        for i in range(self.n_firms):
            for j in range(self.n_firms):
                if i != j:  # Don't affect own price observation
                    if self.np_random.random() > self.price_visibility_prob:
                        observed_prices[i] = (
                            np.nan
                        )  # Cannot observe this competitor's price

        return observed_prices

    def get_observed_prices(self, prices: np.ndarray) -> np.ndarray:
        """
        Get the prices that firms can observe, accounting for information frictions.

        Args:
            prices: Array of actual prices set by each firm

        Returns:
            Array of observed prices with information frictions applied
        """
        return self._add_information_frictions(prices)
