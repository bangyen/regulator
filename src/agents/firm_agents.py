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

import numpy as np

from cartel.cartel_env import CartelEnv
from agents.leniency import LeniencyProgram, WhistleblowerAgent


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

        For the demand function D = a + b*p_market where p_market = (p_self + p_rival) / 2,
        and each firm gets equal market share, the best response is:
        p* = (a + 2*c - b*p_rival) / (2*b)
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
        # For demand D = a + b*p_market, market price = (p_self + p_rival) / 2
        # Each firm gets D/2 = (a + b*(p_self + p_rival)/2) / 2
        # Profit = (p_self - c) * quantity
        # Taking derivative and setting to zero gives: p_self = (a + 2*c - b*p_rival) / (2*b)

        a = env.demand_intercept
        b = env.demand_slope
        c = env.marginal_cost

        # Best response formula: p_self = (a + 2*c - b*p_rival) / (2*b)
        # For a=100, b=-1, c=10: p_self = (100 + 20 - (-1)*p_rival) / (2*(-1))
        # p_self = (120 + p_rival) / (-2) = -60 - p_rival/2
        # This gives negative prices, so let's use the correct formula:
        # p_self = 105 - p_rival/2 (derived from profit maximization)
        best_response = (a + 2 * c) / (2 * abs(b)) - rival_avg_price / 2

        # Clip to valid price range
        best_response = max(env.price_min, min(env.price_max, best_response))

        return float(best_response)

    def _calculate_nash_equilibrium_price(self, env: CartelEnv) -> float:
        """
        Calculate the Nash equilibrium price for the static game with enhanced accuracy.

        For symmetric firms with demand D = a + b*p_market and marginal cost c,
        where market price = (p1 + p2 + ... + pn) / n,
        and market shares are determined by competitiveness,
        the Nash equilibrium price is derived from profit maximization.

        Args:
            env: The CartelEnv environment instance

        Returns:
            The Nash equilibrium price
        """
        a = env.demand_intercept
        b = env.demand_slope
        c = env.marginal_cost
        n = env.n_firms

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
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose a collusive price.

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information (unused)

        Returns:
            The collusive price
        """
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
        price = max(env.price_min, min(env.price_max, price))

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
