"""
Label-free collusion screens.

A screen maps an episode's price path to a suspicion score (higher = more
suspicious) without ever seeing collusion labels. Thresholds are calibrated
on a *competitive null*: episodes simulated from the market's known demand
and costs with firms that compete (best response, myopic learners). Setting
the threshold at the null's (1 - alpha) quantile targets a false-positive
rate of alpha on competitive markets.

Screens follow the empirical screening literature:

- variance: collusive prices are more stable (Abrantes-Metz et al. 2006)
- rigidity: collusive prices change less often
- markup: prices sit above the static Nash price (structural; needs demand
  and costs)
- parallel: firms' prices move together and stay close, which is the rule
  the built-in Regulator uses
- retaliation: unilateral cuts are answered by rivals and then reversed
  (punish-and-return, the signature of reward-punishment schemes)
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from regulator.cartel.cartel_env import CartelEnv
from regulator.detectors.ml_detector import STRATEGIC_FEATURE_NAMES, _strategic_features


@dataclass(frozen=True)
class MarketBenchmarks:
    """Static reference prices of a market."""

    nash_price: float
    monopoly_price: float
    marginal_cost: float
    demand_intercept: float


def market_benchmarks(
    env_kwargs: dict[str, Any] | None = None, grid_step: float = 0.25
) -> MarketBenchmarks:
    """
    One-shot symmetric Nash price and joint-profit-maximizing price.

    Computed on a price grid by best-response iteration (Nash) and by
    maximizing joint profit over symmetric prices (monopoly), with demand
    shocks switched off.
    """
    kwargs = {"n_firms": 2, **(env_kwargs or {}), "shock_std": 0.0}
    env = CartelEnv(**kwargs)
    env.reset(seed=0)
    env.current_demand_shock = 0.0
    n = env.n_firms
    grid = np.arange(env.marginal_cost, env.price_max + grid_step, grid_step)

    def profits(prices: np.ndarray) -> np.ndarray:
        demand = env._calculate_demand(prices)
        quantities = env._calculate_market_shares(prices) * demand
        return np.asarray(env._calculate_profits(prices, quantities))

    def best_response(rival: float) -> float:
        own = [profits(np.array([p] + [rival] * (n - 1)))[0] for p in grid]
        return float(grid[int(np.argmax(own))])

    price = float(grid[len(grid) // 2])
    for _ in range(200):
        new = best_response(price)
        if abs(new - price) < grid_step / 2:
            break
        price = new

    joint = [profits(np.full(n, p)).sum() for p in grid]
    return MarketBenchmarks(
        nash_price=price,
        monopoly_price=float(grid[int(np.argmax(joint))]),
        marginal_cost=float(env.marginal_cost),
        demand_intercept=float(env.demand_intercept),
    )


Screen = Callable[[np.ndarray, MarketBenchmarks], float]


def variance_screen(prices: np.ndarray, market: MarketBenchmarks) -> float:
    """Negative coefficient of variation of the market price over time."""
    market_price = prices.mean(axis=1)
    return -float(market_price.std() / max(market_price.mean(), 1e-9))


def rigidity_screen(prices: np.ndarray, market: MarketBenchmarks) -> float:
    """Share of periods in which the market price moves by less than 0.5%."""
    market_price = prices.mean(axis=1)
    changes = np.abs(np.diff(market_price)) / np.maximum(market_price[:-1], 1e-9)
    return float(np.mean(changes < 0.005)) if len(changes) else 0.0


def markup_screen(prices: np.ndarray, market: MarketBenchmarks) -> float:
    """Mean price position between Nash (0) and monopoly (1)."""
    span = max(market.monopoly_price - market.nash_price, 1e-9)
    return float((prices.mean() - market.nash_price) / span)


def parallel_screen(
    prices: np.ndarray, market: MarketBenchmarks, tolerance: float = 5.0
) -> float:
    """Share of periods in which all firms price within `tolerance` of each other."""
    spread = prices.max(axis=1) - prices.min(axis=1)
    return float(np.mean(spread <= tolerance))


def retaliation_screen(prices: np.ndarray, market: MarketBenchmarks) -> float:
    """
    Punish-and-return: how strongly rivals cut after an unprovoked cut, times
    how often the cutter then restores its price. 0 when no cuts occur.
    """
    features = dict(
        zip(
            STRATEGIC_FEATURE_NAMES,
            _strategic_features(prices, market.marginal_cost, market.demand_intercept),
            strict=True,
        )
    )
    retaliation = max(0.0, -features["rival_response_to_cut"])
    return float(retaliation * features["cut_recovery_rate"])


SCREENS: dict[str, Screen] = {
    "variance": variance_screen,
    "rigidity": rigidity_screen,
    "markup": markup_screen,
    "parallel": parallel_screen,
    "retaliation": retaliation_screen,
}


def score_episode(
    prices: np.ndarray, market: MarketBenchmarks, screens: dict[str, Screen] = SCREENS
) -> dict[str, float]:
    """Score one price path (steps x firms) with every screen."""
    return {name: screen(prices, market) for name, screen in screens.items()}


def calibrate_thresholds(
    null_scores: list[dict[str, float]], alpha: float = 0.05
) -> dict[str, float]:
    """
    Per-screen threshold at the (1 - alpha) quantile of competitive scores.

    An episode is flagged when its score is strictly above the threshold, so
    screens with many tied scores (e.g. rigidity 0) flag less than alpha.
    """
    names = null_scores[0].keys()
    return {
        name: float(np.quantile([s[name] for s in null_scores], 1 - alpha))
        for name in names
    }


def flag(scores: dict[str, float], thresholds: dict[str, float]) -> dict[str, bool]:
    """Which screens flag an episode."""
    return {name: scores[name] > thresholds[name] for name in thresholds}
