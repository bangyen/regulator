"""
Screening study: how well do label-free screens separate collusion from
competition, and how often do they flag competitive markets?

Protocol:

1. Calibrate every screen's threshold on competitive episodes only (the
   competitive null a regulator can simulate from known demand and costs).
2. On fresh seeds, and for Q-learners on pairs never used in calibration,
   measure the share of episodes flagged for each competitive population
   (false-positive rate) and each collusive population (detection rate).
3. Report each screen's ROC AUC over all competitive vs all collusive test
   episodes. Labels are used only for this evaluation, never for fitting.

The built-in rule-based Regulator is scored the same way (score = number of
steps it flags), so its false positives are directly comparable.
"""

import copy
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from regulator.agents.firm_agents import (
    BaseAgent,
    BestResponseAgent,
    CollusiveAgent,
    NoisyAgent,
)
from regulator.agents.q_learning_agent import QLearningAgent
from regulator.agents.regulator import Regulator
from regulator.agents.stealth_agent import StealthCollusiveAgent
from regulator.cartel.cartel_env import CartelEnv
from regulator.experiments.q_learning import train_q_learners
from regulator.screens import (
    SCREENS,
    MarketBenchmarks,
    calibrate_thresholds,
    market_benchmarks,
    score_episode,
)

# The rule-based Regulator with its shipped behavior (fine on any flag)
DEFAULT_RULE = "rule_based_regulator (as shipped)"

# Makes an agent for a firm index; the second argument is a seed
Factory = Callable[[int, int], BaseAgent]


@dataclass
class Population:
    """A kind of market: how to build its firms, and whether it colludes."""

    name: str
    colludes: bool
    make_lineups: Callable[[int], list[Factory]]  # seed -> one factory per firm


def _pair(factory: Factory) -> Callable[[int], list[Factory]]:
    return lambda seed: [factory, factory]


def _best_response(i: int, s: int) -> BaseAgent:
    return BestResponseAgent(i, seed=s)


def _noisy_best_response(i: int, s: int) -> BaseAgent:
    noise = np.random.default_rng(s // 10).uniform(1.0, 4.0)
    return NoisyAgent(BestResponseAgent(i, seed=s), noise, seed=s + 1)


def _explicit_cartel(i: int, s: int) -> BaseAgent:
    return CollusiveAgent(i, seed=s)


def _stealth_cartel(i: int, s: int) -> BaseAgent:
    return StealthCollusiveAgent(i, seed=s)


def _frozen_pairs(
    pairs: list[list[QLearningAgent]], explore: float
) -> Callable[[int], list[Factory]]:
    """Episodes cycle through trained pairs; each gets a fresh RNG."""

    def lineup(seed: int) -> list[Factory]:
        trained = pairs[seed % len(pairs)]

        def make(i: int, s: int) -> BaseAgent:
            agent = copy.deepcopy(trained[i])
            agent.np_random = np.random.default_rng(s)
            agent.freeze(epsilon=explore)
            agent.reset()
            return agent

        return [make, make]

    return lineup


def simulate_prices(
    factories: list[Factory], steps: int, seed: int, env_kwargs: dict[str, Any]
) -> np.ndarray:
    """Run one episode and return realized prices (steps x firms)."""
    env = CartelEnv(n_firms=len(factories), max_steps=steps, seed=seed, **env_kwargs)
    agents = [make(i, seed * 10 + i) for i, make in enumerate(factories)]
    observation, info = env.reset(seed=seed)
    history = []
    for _ in range(steps):
        prices = [agent.choose_price(observation, env, info) for agent in agents]
        observation, rewards, terminated, truncated, info = env.step(np.array(prices))
        for i, agent in enumerate(agents):
            agent.update_history(prices[i], np.delete(np.array(prices), i))
            agent.observe_outcome(float(rewards[i]))
        history.append(np.asarray(info["prices"], dtype=float))
        if terminated or truncated:
            break
    return np.array(history)


def rule_based_score(prices: np.ndarray) -> float:
    """Number of steps the default Regulator flags (parallel or break)."""
    regulator = Regulator(seed=0)
    flagged = 0
    for step, step_prices in enumerate(prices):
        result = regulator.monitor_step(step_prices, step)
        flagged += bool(
            result["parallel_violation"] or result["structural_break_violation"]
        )
    return float(flagged)


def _scores(
    prices: np.ndarray, market: MarketBenchmarks, burn_in: int
) -> dict[str, float]:
    # Episodes start from zero prices; screens look at behavior after the
    # start-up transient, as screens on real data use settled periods
    settled = prices[burn_in:]
    scores = score_episode(settled, market)
    scores["rule_based_regulator"] = rule_based_score(settled)
    scores[DEFAULT_RULE] = scores["rule_based_regulator"]
    return scores


def _train(job: tuple[float, int, int]) -> list[QLearningAgent]:
    discount, steps, seed = job
    return train_q_learners(discount=discount, steps=steps, seed=seed)


@dataclass
class StudyResult:
    """Flag rates per population and AUC per screen."""

    thresholds: dict[str, float]
    flag_rates: dict[str, dict[str, float]]  # population -> screen -> rate
    auc: dict[str, float]
    mean_prices: dict[str, float]
    market: MarketBenchmarks
    populations: list[Population] = field(default_factory=list)


def run_study(
    episodes: int = 40,
    calibration_episodes: int = 60,
    steps: int = 100,
    q_pairs: int = 4,
    q_train_steps: int = 150_000,
    alpha: float = 0.05,
    explore: float = 0.05,
    burn_in: int = 10,
    seed: int = 0,
    env_kwargs: dict[str, Any] | None = None,
) -> StudyResult:
    """
    Calibrate screens on competitive episodes and evaluate on fresh ones.

    Args:
        episodes: Test episodes per population
        calibration_episodes: Competitive episodes per competitive population
            used to set thresholds
        steps: Steps per episode
        q_pairs: Q-learning pairs trained per discount factor (myopic pairs are
            split between calibration and test)
        q_train_steps: Training periods per Q-learning pair
        alpha: Target false-positive rate on the competitive null
        explore: Exploration rate of frozen Q-learners during episodes
        burn_in: Initial steps each screen ignores
        seed: Base seed
        env_kwargs: Extra CartelEnv arguments
    """
    env_kwargs = env_kwargs or {}
    market = market_benchmarks(env_kwargs)

    jobs = [(0.0, q_train_steps, seed + 1000 + k) for k in range(2 * q_pairs)]
    jobs += [(0.95, q_train_steps, seed + 2000 + k) for k in range(q_pairs)]
    with ProcessPoolExecutor() as pool:
        trained = list(pool.map(_train, jobs))
    myopic_calibration = trained[:q_pairs]
    myopic_test = trained[q_pairs : 2 * q_pairs]
    patient = trained[2 * q_pairs :]

    calibration_populations = [
        Population("best response", False, _pair(_best_response)),
        Population("noisy best response", False, _pair(_noisy_best_response)),
        Population(
            "myopic Q-learning", False, _frozen_pairs(myopic_calibration, explore)
        ),
    ]
    test_populations = [
        Population("best response", False, _pair(_best_response)),
        Population("noisy best response", False, _pair(_noisy_best_response)),
        Population("myopic Q-learning", False, _frozen_pairs(myopic_test, explore)),
        Population("explicit cartel", True, _pair(_explicit_cartel)),
        Population("stealth cartel", True, _pair(_stealth_cartel)),
        Population("patient Q-learning", True, _frozen_pairs(patient, explore)),
    ]

    null_scores = []
    for p, population in enumerate(calibration_populations):
        for e in range(calibration_episodes):
            episode_seed = seed + 10_000 + 1000 * p + e
            prices = simulate_prices(
                population.make_lineups(e), steps, episode_seed, env_kwargs
            )
            null_scores.append(_scores(prices, market, burn_in))
    thresholds = calibrate_thresholds(null_scores, alpha)
    # The Regulator as shipped fines on any flagged step: threshold 0
    thresholds[DEFAULT_RULE] = 0.0

    flag_rates: dict[str, dict[str, float]] = {}
    mean_prices: dict[str, float] = {}
    all_scores: list[dict[str, float]] = []
    labels: list[int] = []
    for p, population in enumerate(test_populations):
        flags: dict[str, list[bool]] = {name: [] for name in thresholds}
        prices_seen = []
        for e in range(episodes):
            episode_seed = seed + 50_000 + 1000 * p + e
            prices = simulate_prices(
                population.make_lineups(e), steps, episode_seed, env_kwargs
            )
            scores = _scores(prices, market, burn_in)
            for name, threshold in thresholds.items():
                flags[name].append(scores[name] > threshold)
            all_scores.append(scores)
            labels.append(int(population.colludes))
            prices_seen.append(prices.mean())
        flag_rates[population.name] = {
            name: float(np.mean(values)) for name, values in flags.items()
        }
        mean_prices[population.name] = float(np.mean(prices_seen))

    y = np.array(labels)
    auc = {
        name: float(roc_auc_score(y, [s[name] for s in all_scores]))
        for name in thresholds
    }
    return StudyResult(
        thresholds, flag_rates, auc, mean_prices, market, test_populations
    )


SCREEN_ORDER = [*SCREENS, "rule_based_regulator", DEFAULT_RULE]


def format_study(result: StudyResult) -> str:
    """Markdown table: flag rate per population (FPR / detection) and AUC."""
    pops = result.populations
    header = (
        "| Screen | "
        + " | ".join(f"{p.name} ({'detect' if p.colludes else 'FPR'})" for p in pops)
        + " | AUC |"
    )
    lines = [header, "|" + "---|" * (len(pops) + 2)]
    for name in SCREEN_ORDER:
        cells = [f"{result.flag_rates[p.name][name]:.0%}" for p in pops]
        lines.append(
            f"| {name} | " + " | ".join(cells) + f" | {result.auc[name]:.2f} |"
        )
    prices = ", ".join(f"{p.name} {result.mean_prices[p.name]:.1f}" for p in pops)
    lines.append("")
    lines.append(
        f"Mean prices: {prices} (Nash {result.market.nash_price:.1f}, "
        f"monopoly {result.market.monopoly_price:.1f})"
    )
    return "\n".join(lines)
