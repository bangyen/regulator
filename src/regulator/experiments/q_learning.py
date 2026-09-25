"""
Training Q-learning firms in CartelEnv (tacit algorithmic collusion).

Q-learning needs far more periods than an episode has, so firms are trained
in one long run here and then frozen for evaluation episodes.
"""

from typing import Any

import numpy as np

from regulator.agents.q_learning_agent import (
    DEFAULT_MONOPOLY_PRICE,
    DEFAULT_NASH_PRICE,
    QLearningAgent,
)
from regulator.cartel.cartel_env import CartelEnv


def train_q_learners(
    n_firms: int = 2,
    discount: float = 0.95,
    steps: int = 200_000,
    seed: int = 0,
    env_kwargs: dict[str, Any] | None = None,
    **agent_kwargs: Any,
) -> list[QLearningAgent]:
    """
    Let `n_firms` Q-learners play each other for `steps` periods.

    Args:
        n_firms: Number of firms
        discount: Discount factor for every agent (0.95 patient, 0 myopic)
        steps: Training periods
        seed: Seed for the environment and the agents
        env_kwargs: Extra CartelEnv arguments
        **agent_kwargs: Extra QLearningAgent arguments

    Returns:
        The trained agents, frozen (greedy, no further learning)
    """
    env = CartelEnv(
        n_firms=n_firms, max_steps=steps + 1, seed=seed, **(env_kwargs or {})
    )
    agents = [
        QLearningAgent(
            i, n_firms=n_firms, discount=discount, seed=seed * 1000 + i, **agent_kwargs
        )
        for i in range(n_firms)
    ]
    observation, info = env.reset(seed=seed)
    for _ in range(steps):
        prices = [agent.choose_price(observation, info=info) for agent in agents]
        observation, rewards, _, _, info = env.step(np.array(prices))
        for agent, reward in zip(agents, rewards, strict=True):
            agent.observe_outcome(float(reward))

    for agent in agents:
        agent.freeze()
        agent.reset()
    return agents


def collusion_index(
    average_price: float,
    nash_price: float = DEFAULT_NASH_PRICE,
    monopoly_price: float = DEFAULT_MONOPOLY_PRICE,
) -> float:
    """Share of the way from the Nash to the monopoly price (0 = Nash, 1 = M)."""
    return (average_price - nash_price) / (monopoly_price - nash_price)
