"""
Stealthy collusive agent for the CartelEnv environment.

This module implements a StealthCollusiveAgent that attempts to maintain
higher-than-competitive prices while avoiding detection through "jitter"
and awareness of regulatory risk.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.agents.firm_agents import BaseAgent
from src.cartel.cartel_env import CartelEnv


class StealthCollusiveAgent(BaseAgent):
    """
    Agent that attempts to collude while avoiding regulatory detection.

    It uses several tactics:
    1. Jitter: Adds random noise to prices to avoid perfect parallel pricing.
    2. Risk Awareness: Monitors its own "violation history" (if provided in info)
       to reduce prices when risk is high.
    3. Partial Coordination: Matches rival prices but only gradually.
    """

    def __init__(
        self,
        agent_id: int,
        target_collusive_price: float = 40.0,
        jitter_std: float = 2.0,
        risk_threshold: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the stealth collusive agent.

        Args:
            agent_id: Unique identifier for this agent.
            target_collusive_price: The price level the agent wants to maintain.
            jitter_std: Standard deviation of the noise added to prices.
            risk_threshold: Threshold for regulatory risk above which the agent
                            becomes more competitive.
            seed: Random seed for reproducibility.
        """
        super().__init__(agent_id, seed)
        self.target_collusive_price = target_collusive_price
        self.jitter_std = jitter_std
        self.risk_threshold = risk_threshold
        self.violation_history: List[bool] = []

    def choose_price(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose a price that balances collusion and stealth.

        Args:
            observation: Current environment observation.
            env: The CartelEnv environment instance.
            info: Additional information from the environment.

        Returns:
            The price to set for this agent.
        """
        # 1. Assess Regulatory Risk
        # In a real scenario, this might come from 'info' if the regulator provides feedback
        # or it might be inferred from past fines.
        risk_level = 0.0
        if info and "ml_collusion_probability" in info:
            risk_level = info["ml_collusion_probability"]
        elif self.violation_history:
            # Simple heuristic: violation rate in last 10 steps
            risk_level = float(np.mean(self.violation_history[-10:]))

        # 2. Determine Base Strategy
        if risk_level > self.risk_threshold:
            # High risk: move toward competitive price (slightly above marginal cost)
            base_price = env.marginal_cost * 1.2
        else:
            # Low risk: attempt collusion
            if self.rival_price_history:
                rival_avg = self.rival_price_history[-1]
                # Aim for a point between target and rival average to coordinate subtly
                base_price = 0.7 * self.target_collusive_price + 0.3 * rival_avg
            else:
                base_price = self.target_collusive_price

        # 3. Add Jitter to avoid perfect correlation
        jitter = float(self.np_random.normal(0, self.jitter_std))
        final_price = base_price + jitter

        # 4. Clip to valid range
        final_price = max(env.price_min, min(env.price_max, final_price))

        return float(final_price)

    def update_history(self, my_price: float, rival_prices: np.ndarray) -> None:
        """Update history and track if a fine was applied (mocked here or passed in)."""
        super().update_history(my_price, rival_prices)
        # Note: In the current runner, the agent doesn't directly know if it was fined
        # unless we explicitly pass that info in. For now, we'll assume choosing
        # a price doesn't update 'violation_history' here, but rather in a custom loop.

    def record_violation(self, was_violation: bool) -> None:
        """Explicitly record a regulatory violation."""
        self.violation_history.append(was_violation)

    def reset(self) -> None:
        """Reset internal state."""
        super().reset()
        self.violation_history.clear()
