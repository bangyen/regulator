"""
Offline training data for the MLRegulator's collusion classifier.

A regulator never observes ground truth during an episode, so the classifier
is trained beforehand on simulated price windows whose labels come from the
strategies the firms were given (colluding vs competing).
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np

from regulator.agents.firm_agents import (
    BaseAgent,
    BestResponseAgent,
    CollusiveAgent,
    NoisyAgent,
    RandomAgent,
    TitForTatAgent,
)
from regulator.agents.ml_regulator import MLRegulator
from regulator.agents.stealth_agent import StealthCollusiveAgent
from regulator.cartel.cartel_env import CartelEnv

Factory = Callable[[int, np.random.Generator], BaseAgent]


def _colluder(i: int, rng: np.random.Generator) -> BaseAgent:
    kind = rng.integers(3)
    seed = int(rng.integers(2**31))
    if kind == 0:
        return CollusiveAgent(i, collusive_price=float(rng.uniform(45, 60)), seed=seed)
    return StealthCollusiveAgent(
        i,
        target_collusive_price=float(rng.uniform(44, 60)),
        jitter_std=float(rng.uniform(0.5, 6.0)),
        seed=seed,
    )


def _competitor(i: int, rng: np.random.Generator) -> BaseAgent:
    kind = rng.integers(4)
    seed = int(rng.integers(2**31))
    if kind == 0:
        return RandomAgent(i, seed=seed)
    if kind == 1:
        return TitForTatAgent(i, seed=seed)
    inner = BestResponseAgent(i, seed=seed)
    if kind == 2:
        return inner
    return NoisyAgent(inner, float(rng.uniform(1.0, 6.0)), seed=seed + 1)


def _simulate_prices(
    agents: list[BaseAgent], steps: int, seed: int
) -> list[np.ndarray]:
    env = CartelEnv(n_firms=len(agents), max_steps=steps, seed=seed)
    obs, info = env.reset(seed=seed)
    history = []
    for _ in range(steps):
        prices = [agent.choose_price(obs, env, info) for agent in agents]
        obs, _, terminated, truncated, info = env.step(np.array(prices))
        for i, agent in enumerate(agents):
            agent.update_history(prices[i], np.delete(np.array(prices), i))
        history.append(np.asarray(info["prices"], dtype=float))
        if terminated or truncated:
            break
    return history


def simulate_labeled_windows(
    n_episodes: int = 60,
    steps: int = 60,
    window: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate episodes and cut them into labeled feature windows.

    Half the episodes use only colluding firms (label 1); the rest use only
    competing firms (label 0). Firm counts vary between 2 and 3.

    Returns:
        (X, y): one row of ``MLRegulator._extract_features`` per window
    """
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[int] = []

    for episode in range(n_episodes):
        label = episode % 2
        make = _colluder if label else _competitor
        n_firms = int(rng.integers(2, 4))
        agents = [make(i, rng) for i in range(n_firms)]
        history = _simulate_prices(agents, steps, int(rng.integers(2**31)))

        for end in range(window, len(history) + 1):
            features.append(MLRegulator._extract_features(history[end - window : end]))
            labels.append(label)

    return np.array(features), np.array(labels)


@lru_cache(maxsize=4)
def train_collusion_classifier(
    seed: int = 0, n_episodes: int = 60, steps: int = 60, window: int = 10
) -> Any:
    """
    Fit (and cache) a collusion classifier on simulated labeled windows.

    Returns:
        A fitted scikit-learn pipeline with ``predict_proba``
    """
    X, y = simulate_labeled_windows(n_episodes, steps, window, seed)
    regulator = MLRegulator(seed=seed, feature_window_size=window)
    regulator.fit_classifier(X, y)
    return regulator.collusion_classifier
