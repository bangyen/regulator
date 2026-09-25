"""
Tests for the stealth collusive agent.

This module tests the StealthCollusiveAgent class including jittered pricing,
risk-aware retreat to competitive prices, and history management.
"""

import numpy as np

from regulator.agents.stealth_agent import StealthCollusiveAgent
from regulator.experiments.experiment_runner import create_agent

OBS = np.array([30.0, 30.0])
MARKET = {
    "market_params": {"price_min": 1.0, "price_max": 100.0, "marginal_cost": 10.0}
}


class TestStealthCollusiveAgent:
    """Test suite for StealthCollusiveAgent class."""

    def test_initialization(self) -> None:
        """Test default parameters and empty violation history."""
        agent = StealthCollusiveAgent(agent_id=3, seed=0)

        assert agent.agent_id == 3
        assert agent.target_collusive_price == 50.0
        assert agent.jitter_std == 2.0
        assert agent.risk_threshold == 0.5
        assert agent.violation_history == []

    def test_low_risk_prices_near_target(self) -> None:
        """Without risk signals, prices cluster around the collusive target."""
        agent = StealthCollusiveAgent(agent_id=0, seed=42)

        prices = [agent.choose_price(OBS, info=MARKET) for _ in range(200)]

        assert abs(np.mean(prices) - 50.0) < 1.0
        # Jitter keeps prices from being identical
        assert np.std(prices) > 0.5

    def test_no_jitter_is_deterministic(self) -> None:
        """With zero jitter the agent prices exactly at the target."""
        agent = StealthCollusiveAgent(agent_id=0, jitter_std=0.0, seed=1)

        assert agent.choose_price(OBS, info=MARKET) == 50.0

    def test_blends_toward_rival_price(self) -> None:
        """Once rival history exists, price blends target and rival average."""
        agent = StealthCollusiveAgent(agent_id=0, jitter_std=0.0, seed=1)
        agent.update_history(40.0, np.array([20.0, 20.0]))

        price = agent.choose_price(OBS, info=MARKET)

        assert price == 0.7 * 50.0 + 0.3 * 20.0

    def test_high_ml_probability_triggers_retreat(self) -> None:
        """A high collusion probability pushes price toward marginal cost."""
        agent = StealthCollusiveAgent(agent_id=0, jitter_std=0.0, seed=1)
        info = {**MARKET, "ml_collusion_probability": 0.9}

        assert agent.choose_price(OBS, info=info) == 10.0 * 1.2

    def test_violation_history_triggers_retreat(self) -> None:
        """Recorded violations above the threshold also trigger retreat."""
        agent = StealthCollusiveAgent(agent_id=0, jitter_std=0.0, seed=1)
        for _ in range(10):
            agent.record_violation(True)

        assert agent.choose_price(OBS, info=MARKET) == 10.0 * 1.2

    def test_old_violations_are_forgotten(self) -> None:
        """Only the last 10 violation records count toward risk."""
        agent = StealthCollusiveAgent(agent_id=0, jitter_std=0.0, seed=1)
        for _ in range(10):
            agent.record_violation(True)
        for _ in range(10):
            agent.record_violation(False)

        assert agent.choose_price(OBS, info=MARKET) == 50.0

    def test_price_clipped_to_bounds(self) -> None:
        """Prices are clipped to the market's price range."""
        agent = StealthCollusiveAgent(
            agent_id=0, target_collusive_price=500.0, jitter_std=0.0, seed=1
        )

        assert agent.choose_price(OBS, info=MARKET) == 100.0

    def test_falls_back_to_env_attributes(self) -> None:
        """Without market_params in info, bounds come from the env."""

        class Env:
            price_min = 5.0
            price_max = 30.0
            marginal_cost = 10.0

        agent = StealthCollusiveAgent(agent_id=0, jitter_std=0.0, seed=1)

        assert agent.choose_price(OBS, env=Env()) == 30.0

    def test_seed_reproducibility(self) -> None:
        """Same seed yields the same price sequence."""
        a = StealthCollusiveAgent(agent_id=0, seed=7)
        b = StealthCollusiveAgent(agent_id=0, seed=7)

        assert [a.choose_price(OBS, info=MARKET) for _ in range(5)] == [
            b.choose_price(OBS, info=MARKET) for _ in range(5)
        ]

    def test_reset_clears_histories(self) -> None:
        """Reset clears price and violation histories."""
        agent = StealthCollusiveAgent(agent_id=0, seed=1)
        agent.update_history(40.0, np.array([35.0]))
        agent.record_violation(True)

        agent.reset()

        assert len(agent.price_history) == 0
        assert len(agent.rival_price_history) == 0
        assert agent.violation_history == []

    def test_create_agent_stealth(self) -> None:
        """The experiment runner can build a stealth agent by name."""
        agent = create_agent("stealth", agent_id=2, seed=0)

        assert isinstance(agent, StealthCollusiveAgent)
        assert agent.agent_id == 2
