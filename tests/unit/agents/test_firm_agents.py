"""
Unit tests for the minimalist firm agents.
"""

import numpy as np
import pytest
from src.agents.firm_agents import (
    RandomAgent,
    BestResponseAgent,
    TitForTatAgent,
)
from src.cartel.cartel_env import CartelEnv


class TestFirmAgents:
    """Minimalist test suite for firm agents."""

    @pytest.fixture
    def env(self):
        return CartelEnv(n_firms=2)

    def test_random_agent(self, env):
        agent = RandomAgent(agent_id=0)
        obs, _ = env.reset()
        price = agent.choose_price(obs, env)
        assert 1.0 <= price <= 100.0

    def test_tit_for_tat_agent(self, env):
        agent = TitForTatAgent(agent_id=0)
        obs, _ = env.reset()

        # Initial price
        price1 = agent.choose_price(obs, env)
        assert price1 == 15.0  # Default initial price (MC=10 + 5)

        # Update history with rival price
        agent.update_history(price1, np.array([30.0]))
        price2 = agent.choose_price(obs, env)
        assert price2 == 30.0

    def test_best_response_agent(self, env):
        agent = BestResponseAgent(agent_id=0)
        obs, _ = env.reset()
        price = agent.choose_price(obs, env)
        assert price > 0

    def test_history_management(self):
        agent = RandomAgent(agent_id=0, history_len=2)
        agent.update_history(10.0, np.array([20.0]))
        agent.update_history(15.0, np.array([25.0]))
        agent.update_history(20.0, np.array([30.0]))

        assert len(agent.price_history) == 2
        assert list(agent.price_history) == [15.0, 20.0]
        assert list(agent.rival_price_history) == [25.0, 30.0]
