"""Tests for label-free collusion screens and the screening study."""

import numpy as np
import pytest

from regulator.experiments.screening import (
    DEFAULT_RULE,
    SCREEN_ORDER,
    format_study,
    rule_based_score,
    run_study,
    simulate_prices,
)
from regulator.screens import (
    SCREENS,
    MarketBenchmarks,
    calibrate_thresholds,
    flag,
    market_benchmarks,
    markup_screen,
    parallel_screen,
    retaliation_screen,
    rigidity_screen,
    variance_screen,
)

MARKET = MarketBenchmarks(
    nash_price=40.0, monopoly_price=55.0, marginal_cost=10.0, demand_intercept=100.0
)


def test_market_benchmarks_default_market() -> None:
    market = market_benchmarks()

    assert market.nash_price == 40.0
    assert market.monopoly_price == 55.0


def test_market_benchmarks_respond_to_parameters() -> None:
    three = market_benchmarks({"n_firms": 3})
    costly = market_benchmarks({"marginal_cost": 20.0})

    assert three.nash_price < 40.0  # more competition
    assert costly.nash_price > 40.0 and costly.monopoly_price == 60.0


def test_variance_screen_prefers_stable_prices() -> None:
    stable = np.full((20, 2), 50.0)
    volatile = np.column_stack([np.linspace(30, 60, 20)] * 2)

    assert variance_screen(stable, MARKET) > variance_screen(volatile, MARKET)
    assert variance_screen(stable, MARKET) == 0.0


def test_rigidity_screen() -> None:
    prices = np.array([[50, 50], [50, 50], [55, 55], [55, 55], [55, 55]], float)

    assert rigidity_screen(prices, MARKET) == pytest.approx(3 / 4)


def test_markup_screen_is_the_collusion_index() -> None:
    assert markup_screen(np.full((5, 2), 40.0), MARKET) == 0.0
    assert markup_screen(np.full((5, 2), 55.0), MARKET) == 1.0
    assert markup_screen(np.full((5, 2), 47.5), MARKET) == pytest.approx(0.5)


def test_parallel_screen() -> None:
    prices = np.array([[40, 42], [40, 50], [50, 51], [30, 60]], float)

    assert parallel_screen(prices, MARKET) == 0.5


def test_retaliation_screen_rewards_punish_and_return() -> None:
    punish = np.array(
        [[50, 50], [50, 50], [42, 50], [42, 42], [50, 46], [50, 50], [50, 50]], float
    )
    ignore = np.array(
        [[50, 50], [50, 50], [42, 50], [42, 50], [42, 50], [42, 50], [42, 50]], float
    )

    assert retaliation_screen(punish, MARKET) > 0.1
    assert retaliation_screen(ignore, MARKET) == 0.0


def test_calibration_and_flagging() -> None:
    null = [{"markup": float(x)} for x in np.linspace(0, 1, 101)]

    thresholds = calibrate_thresholds(null, alpha=0.1)

    assert thresholds["markup"] == pytest.approx(0.9)
    assert flag({"markup": 0.95}, thresholds) == {"markup": True}
    assert flag({"markup": 0.9}, thresholds) == {"markup": False}


def test_rule_based_score_counts_flagged_steps() -> None:
    identical = np.full((12, 2), 40.0)
    dispersed = np.column_stack([np.full(12, 20.0), np.full(12, 80.0)])

    assert rule_based_score(identical) >= 8  # parallel pricing flagged
    assert rule_based_score(dispersed) == 0


def test_simulate_prices_shape() -> None:
    from regulator.agents.firm_agents import BestResponseAgent

    def make(i: int, s: int) -> BestResponseAgent:
        return BestResponseAgent(i, seed=s)

    prices = simulate_prices([make, make], steps=15, seed=0, env_kwargs={})

    assert prices.shape == (15, 2)


def test_small_study_runs_end_to_end() -> None:
    result = run_study(
        episodes=4,
        calibration_episodes=4,
        steps=30,
        q_pairs=1,
        q_train_steps=1_000,
        seed=0,
    )

    assert set(result.auc) == set(SCREEN_ORDER)
    assert set(result.flag_rates) == {p.name for p in result.populations}
    assert all(
        0.0 <= r <= 1.0 for rates in result.flag_rates.values() for r in rates.values()
    )
    # The shipped regulator fines on any flag, so best responders are flagged
    assert result.flag_rates["best response"][DEFAULT_RULE] == 1.0
    assert result.mean_prices["explicit cartel"] > result.mean_prices["best response"]
    table = format_study(result)
    assert table.splitlines()[0].startswith("| Screen |")
    assert all(name in table for name in SCREENS)


def test_cli_screen_command() -> None:
    from click.testing import CliRunner

    from regulator.cli import main

    result = CliRunner().invoke(
        main,
        [
            "screen",
            "--episodes",
            "3",
            "--steps",
            "25",
            "--q-pairs",
            "1",
            "--q-train-steps",
            "500",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "| markup |" in result.output
