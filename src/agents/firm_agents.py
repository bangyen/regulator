"""
Baseline firm agents for the CartelEnv environment.

This module implements three baseline firm agents:
- RandomAgent: Chooses random prices within bounds
- BestResponseAgent: Chooses price that maximizes profit against average of rivals
- TitForTatAgent: Copies previous rival average price

These agents serve as baselines for testing and benchmarking the CartelEnv.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from cartel.cartel_env import CartelEnv


class BaseAgent(ABC):
    """
    Abstract base class for firm agents.

    This class defines the interface that all firm agents must implement,
    providing a consistent way to interact with the CartelEnv environment.
    """

    def __init__(self, agent_id: int, seed: Optional[int] = None) -> None:
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for this agent
            seed: Random seed for reproducibility
        """
        self.agent_id = agent_id
        self.np_random = np.random.default_rng(seed)
        self.price_history: list[float] = []
        self.rival_price_history: list[float] = []

    @abstractmethod
    def choose_price(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose a price for the current step.

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information from the environment

        Returns:
            The price to set for this agent
        """
        pass

    def update_history(self, my_price: float, rival_prices: np.ndarray) -> None:
        """
        Update the agent's price history.

        Args:
            my_price: The price this agent chose
            rival_prices: Array of prices chosen by rival agents
        """
        self.price_history.append(my_price)
        rival_avg = float(np.mean(rival_prices))
        self.rival_price_history.append(rival_avg)

    def reset(self) -> None:
        """Reset the agent's internal state."""
        self.price_history.clear()
        self.rival_price_history.clear()


class RandomAgent(BaseAgent):
    """
    Agent that chooses random prices within the environment's bounds.

    This agent serves as a baseline that makes no strategic decisions,
    simply sampling prices uniformly from the valid price range.
    """

    def choose_price(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose a random price within the environment's bounds.

        Args:
            observation: Current environment observation (unused)
            env: The CartelEnv environment instance
            info: Additional information (unused)

        Returns:
            A random price within [price_min, price_max]
        """
        return float(self.np_random.uniform(env.price_min, env.price_max))


class BestResponseAgent(BaseAgent):
    """
    Agent that chooses the price that maximizes profit against the average of rivals.

    This agent assumes rivals will set prices equal to the average of their previous
    prices and chooses the optimal response to maximize its own profit.
    """

    def choose_price(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose the price that maximizes profit against rival average.

        For a 2-firm game, the best response to rival price p_r is:
        p* = (a + c + b*p_r) / (2*b)
        where a = demand_intercept, b = demand_slope, c = marginal_cost

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information (unused)

        Returns:
            The optimal price response
        """
        # If no rival history, use a reasonable default
        if not self.rival_price_history:
            # Use Nash equilibrium price as default
            nash_price = self._calculate_nash_equilibrium_price(env)
            return nash_price

        # Get the most recent rival average price
        rival_avg_price = self.rival_price_history[-1]

        # Calculate best response price
        # For the demand function D = a + b*p_market, where p_market = (p_self + p_rival) / 2
        # The best response is: p* = (a + c + b*p_rival) / (2*b)
        # where a = demand_intercept, b = demand_slope, c = marginal_cost

        a = env.demand_intercept
        b = env.demand_slope
        c = env.marginal_cost

        # Best response formula
        best_response = (a + c + b * rival_avg_price) / (2 * b)

        # Clip to valid price range
        best_response = max(env.price_min, min(env.price_max, best_response))

        return float(best_response)

    def _calculate_nash_equilibrium_price(self, env: CartelEnv) -> float:
        """
        Calculate the Nash equilibrium price for the static game.

        For symmetric firms with demand D = a + b*p_market and marginal cost c,
        the Nash equilibrium price is: p* = (a + c) / (2*b - b/n_firms)

        Args:
            env: The CartelEnv environment instance

        Returns:
            The Nash equilibrium price
        """
        a = env.demand_intercept
        b = env.demand_slope
        c = env.marginal_cost
        n = env.n_firms

        # Nash equilibrium price for symmetric firms
        nash_price = (a + c) / (2 * b - b / n)

        # Clip to valid price range
        nash_price = max(env.price_min, min(env.price_max, nash_price))

        return float(nash_price)


class TitForTatAgent(BaseAgent):
    """
    Agent that copies the previous average price of rival agents.

    This agent implements a simple tit-for-tat strategy, setting its price
    equal to the average price that rivals set in the previous period.
    """

    def choose_price(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose a price equal to the previous rival average price.

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information (unused)

        Returns:
            The previous rival average price, or a default if no history
        """
        # If no rival history, use a reasonable default
        if not self.rival_price_history:
            # Use a price slightly above marginal cost as default
            default_price = env.marginal_cost + 5.0
            return float(max(env.price_min, min(env.price_max, default_price)))

        # Copy the most recent rival average price
        tit_for_tat_price = self.rival_price_history[-1]

        # Clip to valid price range
        tit_for_tat_price = max(env.price_min, min(env.price_max, tit_for_tat_price))

        return float(tit_for_tat_price)
