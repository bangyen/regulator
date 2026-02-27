"""
Baseline firm agents for the CartelEnv environment.

This module implements three baseline firm agents:
- RandomAgent: Chooses random prices within bounds
- BestResponseAgent: Chooses price that maximizes profit against average of rivals
- TitForTatAgent: Copies previous rival average price

These agents serve as baselines for testing and benchmarking the CartelEnv.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

from regulator.agents.leniency import WhistleblowerAgent, LeniencyProgram


class BaseAgent(ABC):
    """
    Abstract base class for firm agents.
    """

    def __init__(
        self, agent_id: int, seed: Optional[int] = None, history_len: int = 10
    ) -> None:
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for this agent
            seed: Random seed for reproducibility
            history_len: Maximum length of price history to maintain
        """
        self.agent_id = agent_id
        self.np_random = np.random.default_rng(seed)
        # History tracking
        self.price_history: deque[float] = deque(maxlen=history_len)
        self.rival_price_history: deque[float] = deque(maxlen=history_len)

    @abstractmethod
    def choose_price(
        self,
        observation: np.ndarray,
        env: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose a price for the current step.

        Args:
            observation: Current environment observation
            env: Optional environment instance (deprecated, use info instead)
            info: Dictionary containing market parameters and other info
        """
        pass

    def update_history(self, my_price: float, rival_prices: np.ndarray) -> None:
        """Update the agent's price history."""
        self.price_history.append(my_price)
        rival_avg = float(np.mean(rival_prices)) if rival_prices.size > 0 else 0.0
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
        env: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose a random price within market bounds.
        """
        # Prioritize info['market_params']
        params = info.get("market_params", {}) if info else {}
        price_min = params.get("price_min", getattr(env, "price_min", 1.0))
        price_max = params.get("price_max", getattr(env, "price_max", 100.0))

        return float(self.np_random.uniform(price_min, price_max))


class BestResponseAgent(BaseAgent):
    """
    Agent that chooses the price that maximizes profit against the average of rivals.

    This agent assumes rivals will set prices equal to the average of their previous
    prices and chooses the optimal response to maximize its own profit.
    """

    def choose_price(
        self,
        observation: np.ndarray,
        env: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose the price that maximizes profit against rival average.
        """
        params = info.get("market_params", {}) if info else {}

        # If no rival history, use a reasonable default
        if not self.rival_price_history:
            # Use Nash equilibrium price as default
            nash_price = self._calculate_nash_equilibrium_price(env, params)
            return nash_price

        # Get the most recent rival average price
        rival_avg_price = self.rival_price_history[-1]

        # Get market parameters
        a = params.get("demand_intercept", getattr(env, "demand_intercept", 100.0))
        b = params.get("demand_slope", getattr(env, "demand_slope", -1.0))
        c = params.get("marginal_cost", getattr(env, "marginal_cost", 10.0))
        price_min = params.get("price_min", getattr(env, "price_min", 1.0))
        price_max = params.get("price_max", getattr(env, "price_max", 100.0))

        # Best response formula: p_self = (a + 2*c - b*p_rival) / (2*b)
        best_response = (a + 2 * c) / (2 * abs(b)) - rival_avg_price / 2

        # Clip to valid price range
        best_response = max(price_min, min(price_max, best_response))

        return float(best_response)

    def _calculate_nash_equilibrium_price(
        self, env: Optional[Any] = None, params: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate the Nash equilibrium price.
        """
        if params is None:
            params = {}

        a = params.get("demand_intercept", getattr(env, "demand_intercept", 100.0))
        b = params.get("demand_slope", getattr(env, "demand_slope", -1.0))
        c = params.get("marginal_cost", getattr(env, "marginal_cost", 10.0))
        n = params.get("n_firms", getattr(env, "n_firms", 3))
        price_min = params.get("price_min", getattr(env, "price_min", 1.0))
        price_max = params.get("price_max", getattr(env, "price_max", 100.0))

        # Enhanced Nash equilibrium calculation
        # For competitiveness-based market shares: s_i = (1/p_i^α) / Σ(1/p_j^α)
        # where α = competition_intensity

        # For symmetric Nash equilibrium, all firms set the same price p*
        # Market share for each firm = 1/n
        # Total demand = a + b*p*
        # Individual quantity = (a + b*p*) / n
        # Profit = (p* - c) * (a + b*p*) / n

        # Taking derivative with respect to p* and setting to zero:
        # dπ/dp* = (a + b*p*)/n + (p* - c)*b/n = 0
        # (a + b*p*) + (p* - c)*b = 0
        # a + b*p* + b*p* - b*c = 0
        # a + 2*b*p* - b*c = 0
        # 2*b*p* = b*c - a
        # p* = (b*c - a) / (2*b)

        # For b < 0 (downward sloping demand):
        # p* = (|b|*c + a) / (2*|b|)

        if b < 0:
            nash_price = (abs(b) * c + a) / (2 * abs(b))
        else:
            # This shouldn't happen with proper validation, but handle it
            nash_price = (a + n * c) / ((n + 1) * abs(b))

        # Ensure the price is economically reasonable
        # Price should be above marginal cost for positive profits
        nash_price = max(nash_price, c + 1.0)  # At least 1 unit above marginal cost

        # Clip to valid price range
        nash_price = max(price_min, min(price_max, nash_price))

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
        env: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose a price equal to the previous rival average price.
        """
        params = info.get("market_params", {}) if info else {}
        marginal_cost = params.get("marginal_cost", getattr(env, "marginal_cost", 10.0))
        price_min = params.get("price_min", getattr(env, "price_min", 1.0))
        price_max = params.get("price_max", getattr(env, "price_max", 100.0))

        # If no rival history, use a reasonable default
        if not self.rival_price_history:
            # Use a price slightly above marginal cost as default
            default_price = marginal_cost + 5.0
            return float(max(price_min, min(price_max, default_price)))

        # Copy the most recent rival average price
        tit_for_tat_price = self.rival_price_history[-1]

        # Clip to valid price range
        tit_for_tat_price = max(price_min, min(price_max, tit_for_tat_price))

        return float(tit_for_tat_price)


class CollusiveAgent(BaseAgent):
    """
    Agent that attempts to maintain collusive pricing behavior.

    This agent tries to set prices at a collusive level, coordinating with
    other firms to maintain high prices and avoid price wars.
    """

    def __init__(
        self,
        agent_id: int,
        collusive_price: float = 30.0,
        deviation_penalty: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the collusive agent.

        Args:
            agent_id: Unique identifier for this agent
            collusive_price: Target collusive price level
            deviation_penalty: Penalty factor for deviating from collusive price
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, seed)
        self.collusive_price = collusive_price
        self.deviation_penalty = deviation_penalty

    def choose_price(
        self,
        observation: np.ndarray,
        env: Optional[Any] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose a collusive price.
        """
        params = info.get("market_params", {}) if info else {}
        price_min = params.get("price_min", getattr(env, "price_min", 1.0))
        price_max = params.get("price_max", getattr(env, "price_max", 100.0))
        # Start with collusive price
        price = self.collusive_price

        # If we have rival price history, try to coordinate
        if self.rival_price_history:
            rival_avg = self.rival_price_history[-1]

            # If rivals are pricing below collusive level, match them to avoid price war
            if rival_avg < self.collusive_price * 0.9:
                price = rival_avg
            # If rivals are pricing above collusive level, stay at collusive level
            elif rival_avg > self.collusive_price * 1.1:
                price = self.collusive_price
            # Otherwise, set collusive price
            else:
                price = self.collusive_price

        # Add small random variation to avoid perfect coordination
        variation = self.np_random.normal(0, 0.5)
        price += variation

        # Clip to valid price range
        price = max(price_min, min(price_max, price))

        return float(price)


class WhistleblowerTitForTatAgent(TitForTatAgent, WhistleblowerAgent):
    """
    Agent that combines tit-for-tat pricing with strategic whistleblowing.

    This agent sets prices using a tit-for-tat strategy but can also
    strategically whistleblow when leniency incentives are sufficient.
    """

    def __init__(
        self,
        agent_id: int,
        leniency_program: LeniencyProgram,
        whistleblow_threshold: float = 10.0,
        risk_aversion: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the whistleblower tit-for-tat agent.

        Args:
            agent_id: Unique identifier for this agent
            leniency_program: The leniency program instance
            whistleblow_threshold: Minimum incentive to whistleblow
            risk_aversion: Risk aversion parameter
            seed: Random seed for reproducibility
        """
        TitForTatAgent.__init__(self, agent_id, seed)
        WhistleblowerAgent.__init__(
            self, agent_id, leniency_program, whistleblow_threshold, risk_aversion, seed
        )

    def evaluate_whistleblow_opportunity(
        self,
        current_fine: float,
        collusion_probability: float,
        step: int,
        rival_firms: List[int],
    ) -> Tuple[bool, float]:
        """
        Evaluate whether to whistleblow and submit a report if beneficial.

        Args:
            current_fine: Current fine amount if caught
            collusion_probability: Probability of being caught in collusion
            step: Current step number
            rival_firms: List of rival firm IDs to potentially report

        Returns:
            Tuple of (whistled, incentive_value)
        """
        should_whistleblow, incentive = self.evaluate_whistleblow_decision(
            current_fine, collusion_probability, step
        )

        if should_whistleblow and rival_firms:
            # Submit leniency report
            evidence_strength = min(
                0.9, collusion_probability + 0.2
            )  # Base evidence on detection probability
            success = self.leniency_program.submit_report(
                self.agent_id, rival_firms, evidence_strength, step
            )

            if success:
                return True, incentive

        return False, incentive

    def get_combined_statistics(self) -> Dict[str, Any]:
        """
        Get combined statistics from both pricing and whistleblowing behavior.

        Returns:
            Dictionary containing combined statistics
        """
        pricing_stats = {
            "price_history_length": len(self.price_history),
            "rival_price_history_length": len(self.rival_price_history),
        }

        whistleblow_stats = self.get_whistleblow_statistics()

        return {**pricing_stats, **whistleblow_stats}

    def reset(self) -> None:
        """Reset both pricing and whistleblowing history."""
        TitForTatAgent.reset(self)
        WhistleblowerAgent.reset(self)


class StrategicWhistleblowerAgent(BestResponseAgent, WhistleblowerAgent):
    """
    Agent that combines best response pricing with strategic whistleblowing.

    This agent chooses optimal prices against rivals but can also
    strategically whistleblow when leniency incentives are sufficient.
    """

    def __init__(
        self,
        agent_id: int,
        leniency_program: LeniencyProgram,
        whistleblow_threshold: float = 15.0,
        risk_aversion: float = 1.2,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the strategic whistleblower agent.

        Args:
            agent_id: Unique identifier for this agent
            leniency_program: The leniency program instance
            whistleblow_threshold: Minimum incentive to whistleblow
            risk_aversion: Risk aversion parameter
            seed: Random seed for reproducibility
        """
        BestResponseAgent.__init__(self, agent_id, seed)
        WhistleblowerAgent.__init__(
            self, agent_id, leniency_program, whistleblow_threshold, risk_aversion, seed
        )

    def evaluate_whistleblow_opportunity(
        self,
        current_fine: float,
        collusion_probability: float,
        step: int,
        rival_firms: List[int],
    ) -> Tuple[bool, float]:
        """
        Evaluate whether to whistleblow and submit a report if beneficial.

        Args:
            current_fine: Current fine amount if caught
            collusion_probability: Probability of being caught in collusion
            step: Current step number
            rival_firms: List of rival firm IDs to potentially report

        Returns:
            Tuple of (whistled, incentive_value)
        """
        should_whistleblow, incentive = self.evaluate_whistleblow_decision(
            current_fine, collusion_probability, step
        )

        if should_whistleblow and rival_firms:
            # Submit leniency report with higher evidence strength for strategic agent
            evidence_strength = min(0.95, collusion_probability + 0.3)
            success = self.leniency_program.submit_report(
                self.agent_id, rival_firms, evidence_strength, step
            )

            if success:
                return True, incentive

        return False, incentive

    def get_combined_statistics(self) -> Dict[str, Any]:
        """
        Get combined statistics from both pricing and whistleblowing behavior.

        Returns:
            Dictionary containing combined statistics
        """
        pricing_stats = {
            "price_history_length": len(self.price_history),
            "rival_price_history_length": len(self.rival_price_history),
        }

        whistleblow_stats = self.get_whistleblow_statistics()

        return {**pricing_stats, **whistleblow_stats}

    def reset(self) -> None:
        """Reset both pricing and whistleblowing history."""
        BestResponseAgent.reset(self)
        WhistleblowerAgent.reset(self)
