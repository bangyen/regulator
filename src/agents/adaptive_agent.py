"""
Adaptive learning agent for the cartel environment.

This module implements an AdaptiveAgent class that learns from past outcomes
and adjusts its pricing strategy based on market conditions and performance.
"""

from typing import Any, Dict, Optional, Deque
import numpy as np
from collections import deque

from .firm_agents import BaseAgent
from cartel.cartel_env import CartelEnv


class AdaptiveAgent(BaseAgent):
    """
    Adaptive learning agent that adjusts pricing strategy based on past performance.

    This agent learns from market outcomes and adapts its pricing strategy to
    maximize profits while avoiding regulatory detection.
    """

    def __init__(
        self,
        agent_id: int,
        learning_rate: float = 0.1,
        memory_size: int = 50,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01,
        risk_aversion: float = 1.0,
        collusion_tendency: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the adaptive learning agent.

        Args:
            agent_id: Unique identifier for this agent
            learning_rate: Learning rate for strategy updates (0.0-1.0)
            memory_size: Number of past outcomes to remember
            exploration_rate: Initial exploration rate for price selection
            exploration_decay: Rate at which exploration decreases
            min_exploration_rate: Minimum exploration rate
            risk_aversion: Risk aversion parameter (higher = more risk averse)
            collusion_tendency: Tendency to engage in collusive behavior (0.0-1.0)
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, seed)

        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.risk_aversion = risk_aversion
        self.collusion_tendency = collusion_tendency

        # Learning state (use separate deques to avoid type conflicts with base class)
        self.learning_price_history: Deque[float] = deque(maxlen=memory_size)
        self.profit_history: Deque[float] = deque(maxlen=memory_size)
        self.market_price_history: Deque[float] = deque(maxlen=memory_size)
        self.regulator_history: Deque[bool] = deque(maxlen=memory_size)
        self.learning_rival_price_history: Deque[float] = deque(maxlen=memory_size)

        # Strategy parameters (learned over time)
        self.base_price = 50.0  # Base pricing strategy
        self.price_sensitivity = 1.0  # How much to adjust based on market conditions
        self.collusion_threshold = 0.7  # Threshold for collusive behavior
        self.defection_threshold = 0.3  # Threshold for defecting from collusion

        # Performance tracking
        self.total_profits = 0.0
        self.total_fines = 0.0
        self.collusion_attempts = 0
        self.successful_collusions = 0

    def choose_price(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose price using adaptive learning strategy.

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information from environment

        Returns:
            The chosen price
        """
        # Extract market information
        n_firms = env.n_firms
        rival_prices = observation[:n_firms]  # Previous period prices

        # Update rival price history
        if len(rival_prices) > 0:
            avg_rival_price = float(np.mean(rival_prices))
            self.learning_rival_price_history.append(avg_rival_price)

        # Calculate market conditions
        market_conditions = self._analyze_market_conditions(env)

        # Choose pricing strategy
        if self._should_explore():
            # Exploration: try different pricing strategies
            price = self._explore_price(env, market_conditions)
        else:
            # Exploitation: use learned strategy
            price = self._exploit_price(env, market_conditions)

        # Apply risk aversion
        price = self._apply_risk_aversion(price, env, market_conditions)

        # Clip to valid range
        price = max(env.price_min, min(env.price_max, price))

        return float(price)

    def _analyze_market_conditions(self, env: CartelEnv) -> Dict[str, Any]:
        """
        Analyze current market conditions.

        Args:
            env: The CartelEnv environment instance

        Returns:
            Dictionary containing market condition analysis
        """
        conditions = {
            "market_volatility": 0.0,
            "collusion_opportunity": 0.0,
            "competitive_pressure": 0.0,
            "regulatory_risk": 0.0,
        }

        # Market volatility
        if len(self.market_price_history) >= 5:
            recent_prices = list(self.market_price_history)[-5:]
            conditions["market_volatility"] = float(np.std(recent_prices)) / (
                float(np.mean(recent_prices)) + 1e-6
            )

        # Collusion opportunity (based on price similarity)
        if len(self.learning_rival_price_history) >= 3:
            recent_rival_prices = list(self.learning_rival_price_history)[-3:]
            price_similarity = 1.0 - (
                float(np.std(recent_rival_prices))
                / (float(np.mean(recent_rival_prices)) + 1e-6)
            )
            conditions["collusion_opportunity"] = max(0.0, min(1.0, price_similarity))

        # Competitive pressure (based on profit trends)
        if len(self.profit_history) >= 3:
            recent_profits = list(self.profit_history)[-3:]
            profit_trend = (
                float(np.polyfit(range(len(recent_profits)), recent_profits, 1)[0])
                if len(recent_profits) > 1
                else 0.0
            )
            conditions["competitive_pressure"] = max(
                0.0, min(1.0, -profit_trend / 100.0)
            )  # Normalize

        # Regulatory risk (based on past violations)
        if len(self.regulator_history) >= 5:
            recent_violations = list(self.regulator_history)[-5:]
            conditions["regulatory_risk"] = float(np.mean(recent_violations))

        return conditions

    def _should_explore(self) -> bool:
        """Determine if agent should explore or exploit."""
        return float(self.np_random.random()) < self.exploration_rate

    def _explore_price(
        self, env: CartelEnv, market_conditions: Dict[str, Any]
    ) -> float:
        """
        Explore different pricing strategies.

        Args:
            env: The CartelEnv environment instance
            market_conditions: Current market conditions

        Returns:
            Exploration price
        """
        # Choose exploration strategy
        strategy = self.np_random.choice(
            [
                "aggressive",  # Low price to gain market share
                "conservative",  # High price for profit margin
                "collusive",  # Match rival prices
                "defensive",  # Avoid regulatory detection
            ]
        )

        if strategy == "aggressive":
            # Price below market average
            base_price = env.marginal_cost * 1.2  # Slightly above marginal cost
            price = base_price + float(self.np_random.uniform(0, 10))

        elif strategy == "conservative":
            # Price above market average
            base_price = (
                env.demand_intercept / abs(env.demand_slope) * 0.8
            )  # High price
            price = base_price + float(self.np_random.uniform(-10, 10))

        elif strategy == "collusive":
            # Try to match rival prices
            if len(self.learning_rival_price_history) > 0:
                target_price = self.learning_rival_price_history[-1]
                price = target_price + float(self.np_random.uniform(-2, 2))
            else:
                price = self.base_price + float(self.np_random.uniform(-5, 5))

        else:  # defensive
            # Avoid regulatory detection by varying price
            if len(self.learning_price_history) > 0:
                last_price = self.learning_price_history[-1]
                # Vary price to avoid parallel pricing
                variation = float(self.np_random.uniform(-5, 5))
                price = last_price + variation
            else:
                price = self.base_price + float(self.np_random.uniform(-10, 10))

        return float(price)

    def _exploit_price(
        self, env: CartelEnv, market_conditions: Dict[str, Any]
    ) -> float:
        """
        Exploit learned pricing strategy.

        Args:
            env: The CartelEnv environment instance
            market_conditions: Current market conditions

        Returns:
            Exploitation price
        """
        # Base price from learned strategy
        price = self.base_price

        # Adjust based on market conditions
        if market_conditions["collusion_opportunity"] > self.collusion_threshold:
            # High collusion opportunity - try to coordinate
            if len(self.learning_rival_price_history) > 0:
                target_price = self.learning_rival_price_history[-1]
                # Gradually move toward target price
                price = price * 0.7 + target_price * 0.3
                self.collusion_attempts += 1

        elif market_conditions["competitive_pressure"] > 0.5:
            # High competitive pressure - be more aggressive
            price *= 0.9  # Reduce price by 10%

        elif market_conditions["regulatory_risk"] > 0.3:
            # High regulatory risk - be more defensive
            # Vary price to avoid detection
            if len(self.price_history) > 0:
                last_price = self.price_history[-1]
                variation = self.np_random.uniform(-3, 3)
                price = last_price + variation

        # Adjust based on demand conditions
        if hasattr(env, "current_demand_shock"):
            demand_shock = env.current_demand_shock
            if demand_shock > 0:  # Positive demand shock
                price *= 1.05  # Increase price slightly
            else:  # Negative demand shock
                price *= 0.95  # Decrease price slightly

        return price

    def _apply_risk_aversion(
        self, price: float, env: CartelEnv, market_conditions: Dict[str, Any]
    ) -> float:
        """
        Apply risk aversion to price choice.

        Args:
            price: Proposed price
            env: The CartelEnv environment instance
            market_conditions: Current market conditions

        Returns:
            Risk-adjusted price
        """
        # Calculate risk factors
        regulatory_risk = market_conditions["regulatory_risk"]
        market_volatility = market_conditions["market_volatility"]

        # Risk adjustment
        risk_factor = (regulatory_risk + market_volatility) * self.risk_aversion

        if risk_factor > 0.5:  # High risk
            # Move price toward safer range (closer to marginal cost)
            safe_price = env.marginal_cost * 1.5
            price = price * (1 - risk_factor) + safe_price * risk_factor

        return price

    def update_strategy(
        self,
        price: float,
        profit: float,
        market_price: float,
        was_violation: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update strategy based on outcome.

        Args:
            price: Price that was chosen
            profit: Profit earned
            market_price: Market average price
            was_violation: Whether a regulatory violation occurred
            info: Additional information
        """
        # Update history
        self.learning_price_history.append(price)
        self.profit_history.append(profit)
        self.market_price_history.append(market_price)
        self.regulator_history.append(was_violation)

        # Update performance tracking
        self.total_profits += profit
        if was_violation:
            self.total_fines += abs(profit)  # Estimate fine amount

        # Update strategy parameters based on performance
        self._update_base_price(profit, market_price)
        self._update_collusion_tendency(was_violation)
        self._update_price_sensitivity(profit)

        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )

    def _update_base_price(self, profit: float, market_price: float) -> None:
        """Update base pricing strategy based on performance."""
        if len(self.profit_history) < 2:
            return

        # Calculate performance relative to market
        recent_profits = list(self.profit_history)[-5:]
        avg_profit = np.mean(recent_profits)

        if avg_profit > 0:  # Profitable
            # If we're doing well, maintain current strategy
            pass
        else:  # Not profitable
            # Adjust base price toward market average
            self.base_price = self.base_price * 0.9 + market_price * 0.1

    def _update_collusion_tendency(self, was_violation: bool) -> None:
        """Update collusion tendency based on regulatory outcomes."""
        if was_violation:
            # Reduce collusion tendency after violation
            self.collusion_tendency *= 0.95
            self.collusion_threshold = min(0.9, self.collusion_threshold + 0.05)
        else:
            # Gradually increase collusion tendency if no violations
            self.collusion_tendency = min(1.0, self.collusion_tendency * 1.01)

    def _update_price_sensitivity(self, profit: float) -> None:
        """Update price sensitivity based on profit outcomes."""
        if len(self.profit_history) < 3:
            return

        # Calculate profit trend
        recent_profits = list(self.profit_history)[-3:]
        profit_trend = np.polyfit(range(len(recent_profits)), recent_profits, 1)[0]

        if profit_trend > 0:  # Improving profits
            # Increase sensitivity to market conditions
            self.price_sensitivity = min(2.0, self.price_sensitivity * 1.05)
        else:  # Declining profits
            # Decrease sensitivity to be more stable
            self.price_sensitivity = max(0.5, self.price_sensitivity * 0.95)

    def get_strategy_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's strategy and performance.

        Returns:
            Dictionary containing strategy statistics
        """
        stats = {
            "agent_id": self.agent_id,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "risk_aversion": self.risk_aversion,
            "collusion_tendency": self.collusion_tendency,
            "base_price": self.base_price,
            "price_sensitivity": self.price_sensitivity,
            "collusion_threshold": self.collusion_threshold,
            "total_profits": self.total_profits,
            "total_fines": self.total_fines,
            "collusion_attempts": self.collusion_attempts,
            "successful_collusions": self.successful_collusions,
        }

        if len(self.profit_history) > 0:
            stats.update(
                {
                    "avg_profit": float(np.mean(self.profit_history)),
                    "profit_volatility": float(np.std(self.profit_history)),
                    "recent_profit_trend": (
                        float(
                            np.polyfit(
                                range(len(self.profit_history)),
                                list(self.profit_history),
                                1,
                            )[0]
                        )
                        if len(self.profit_history) > 1
                        else 0.0
                    ),
                }
            )

        if len(self.regulator_history) > 0:
            stats["violation_rate"] = float(np.mean(self.regulator_history))

        return stats

    def reset(self) -> None:
        """Reset the agent's learning state."""
        super().reset()
        self.price_history.clear()
        self.profit_history.clear()
        self.market_price_history.clear()
        self.regulator_history.clear()
        self.rival_price_history.clear()

        # Reset performance tracking
        self.total_profits = 0.0
        self.total_fines = 0.0
        self.collusion_attempts = 0
        self.successful_collusions = 0

        # Reset strategy parameters
        self.base_price = 50.0
        self.price_sensitivity = 1.0
        self.collusion_threshold = 0.7
        self.defection_threshold = 0.3
        self.exploration_rate = 0.1
