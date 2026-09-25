"""Tests for the tabular Q-learning agent and its training helpers."""

import numpy as np
import pytest

from regulator.agents.q_learning_agent import QLearningAgent, default_price_grid
from regulator.experiments.q_learning import collusion_index, train_q_learners


def test_default_grid_spans_nash_to_monopoly_with_margin() -> None:
    grid = default_price_grid()

    assert len(grid) == 15
    assert grid[0] == pytest.approx(38.5)
    assert grid[-1] == pytest.approx(56.5)


def test_prices_come_from_the_grid() -> None:
    agent = QLearningAgent(0, seed=0)
    info = {"prices": np.array([45.0, 50.0])}

    for _ in range(20):
        assert agent.choose_price(np.zeros(3), info=info) in agent.price_grid


def test_q_update_uses_reward_and_discounted_next_value() -> None:
    agent = QLearningAgent(0, learning_rate=0.5, discount=0.9, seed=0)
    info = {"prices": np.array([40.0, 40.0])}
    agent.choose_price(np.zeros(3), info=info)
    state, action = agent._state, agent._action
    agent.q_table[:] = 0.0
    agent.q_table[agent._state_index(np.array([50.0, 50.0])), 3] = 10.0

    agent.observe_outcome(100.0)
    agent.choose_price(np.zeros(3), info={"prices": np.array([50.0, 50.0])})

    # 0 + 0.5 * (100 + 0.9 * 10 - 0)
    assert agent.q_table[state, action] == pytest.approx(54.5)
    assert agent.t == 1


def test_no_update_without_a_reward() -> None:
    agent = QLearningAgent(0, seed=0)
    info = {"prices": np.array([40.0, 40.0])}
    agent.choose_price(np.zeros(3), info=info)
    agent.choose_price(np.zeros(3), info=info)

    assert agent.t == 0
    assert not agent.q_table.any()


def test_epsilon_decays_and_freeze_stops_learning() -> None:
    agent = QLearningAgent(0, exploration_decay=0.1, seed=0)
    assert agent.epsilon == 1.0
    agent.t = 50
    assert agent.epsilon == pytest.approx(np.exp(-5))

    agent.freeze(epsilon=0.2)
    agent.observe_outcome(1.0)
    agent.choose_price(np.zeros(3), info={"prices": np.array([40.0, 40.0])})

    assert agent.epsilon == 0.2
    assert agent.t == 50


def test_greedy_policy_follows_q_table() -> None:
    agent = QLearningAgent(0, seed=0)
    prices = np.array([47.5, 47.5])
    agent.q_table[agent._state_index(prices), 7] = 1.0
    agent.freeze()

    assert agent.greedy_price(prices) == agent.price_grid[7]
    assert (
        agent.choose_price(np.zeros(3), info={"prices": prices}) == agent.price_grid[7]
    )


def test_reset_keeps_learning() -> None:
    agent = QLearningAgent(0, seed=0)
    agent.q_table[0, 0] = 5.0
    agent.t = 10

    agent.reset()

    assert agent.q_table[0, 0] == 5.0
    assert agent.t == 10
    assert agent._state is None


def test_three_firm_state_space() -> None:
    agent = QLearningAgent(0, n_firms=3, seed=0)

    assert agent.q_table.shape == (15**3, 15)
    agent.choose_price(np.zeros(4), info={"prices": np.array([40.0, 45.0, 50.0])})


def test_train_q_learners_returns_frozen_agents() -> None:
    agents = train_q_learners(steps=2_000, seed=0)

    assert len(agents) == 2
    assert all(not a.learning and a.t > 1_000 for a in agents)
    assert all(a.q_table.any() for a in agents)


def test_collusion_index() -> None:
    assert collusion_index(40.0) == 0.0
    assert collusion_index(55.0) == 1.0
    assert collusion_index(47.5) == pytest.approx(0.5)
