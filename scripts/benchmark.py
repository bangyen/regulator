#!/usr/bin/env python3
"""
Reproducible detection benchmark.

Generates labeled episodes with known ground truth and reports accuracy,
precision, recall, F1 and ROC AUC for the supervised episode detector
(CollusionDetector), across scenarios of increasing difficulty (see
SCENARIOS). The label comes from the firms' strategies. Label-free screens
are evaluated separately by scripts/screen_study.py.

Usage:
    python scripts/benchmark.py                      # defaults, seed 42
    python scripts/benchmark.py --episodes 200 --model lightgbm --json out.json
"""

import argparse
import copy
import json
import logging
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from regulator.agents.adaptive_agent import AdaptiveAgent
from regulator.agents.firm_agents import (
    BaseAgent,
    BestResponseAgent,
    CollusiveAgent,
    NoisyAgent,
    RandomAgent,
    TitForTatAgent,
)
from regulator.agents.regulator import Regulator
from regulator.agents.stealth_agent import StealthCollusiveAgent
from regulator.cartel.cartel_env import CartelEnv
from regulator.detectors.ml_detector import CollusionDetector, FeatureExtractor
from regulator.episode_logging.episode_runner import (
    run_episode_with_regulator_logging,
)
from regulator.experiments.q_learning import collusion_index, train_q_learners

# An agent factory takes (agent_id, seed); a line-up is one factory per firm.
Factory = Callable[[int, int], BaseAgent]
Lineup = tuple[Factory, ...]


def _stealth(**kwargs: Any) -> Factory:
    return lambda i, s: StealthCollusiveAgent(agent_id=i, seed=s, **kwargs)


def _adaptive(tendency: float) -> Factory:
    return lambda i, s: AdaptiveAgent(agent_id=i, seed=s, collusion_tendency=tendency)


def _cls(agent_cls: type[BaseAgent]) -> Factory:
    return lambda i, s: agent_cls(agent_id=i, seed=s)


def _noisy_best() -> Factory:
    """Best response plus noise; the noise level varies per episode."""

    def make(i: int, s: int) -> BaseAgent:
        noise = np.random.default_rng(s // 10).uniform(2.0, 6.0)
        return NoisyAgent(BestResponseAgent(agent_id=i, seed=s), noise, s + 1)

    return make


def _noisy_colluder() -> Factory:
    """Stealth colluder whose markup (0-6 over ~40) and jitter vary per episode."""

    def make(i: int, s: int) -> BaseAgent:
        rng = np.random.default_rng(s // 10)  # shared by both firms in an episode
        return StealthCollusiveAgent(
            agent_id=i,
            seed=s,
            target_collusive_price=40.0 + rng.uniform(0.0, 6.0),
            jitter_std=rng.uniform(2.0, 6.0),
        )

    return make


COLLUDER, STEALTH = _cls(CollusiveAgent), _stealth()
RANDOM, BEST, TFT = _cls(RandomAgent), _cls(BestResponseAgent), _cls(TitForTatAgent)
NOISY_BEST, NOISY_COLLUDER = _noisy_best(), _noisy_colluder()

# Each scenario: (collusive line-ups, competitive line-ups). Ground truth is the
# strategy the firms were given, not a heuristic on the resulting prices.
SCENARIOS: dict[str, tuple[list[Lineup], list[Lineup]]] = {
    # Fixed-price colluders vs erratic competitors; trivially separable
    "baseline": (
        [(COLLUDER, COLLUDER), (STEALTH, STEALTH), (COLLUDER, STEALTH)],
        [(RANDOM, RANDOM), (BEST, BEST), (RANDOM, BEST), (TFT, RANDOM)],
    ),
    # Comparable noise on both sides; colluders sit 0-6 above the
    # best-response price level (~40 in this market)
    "noisy": (
        [(NOISY_COLLUDER,) * 2],
        [(NOISY_BEST,) * 2],
    ),
    # A single colluder is not a cartel: pairs vs one colluder + one competitor
    "mixed": (
        [(NOISY_COLLUDER,) * 2],
        [(NOISY_COLLUDER, NOISY_BEST)],
    ),
    # Learning agents that differ only in their disposition to collude
    "adaptive": (
        [(_adaptive(0.9), _adaptive(0.9))],
        [(_adaptive(0.1), _adaptive(0.1))],
    ),
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both_classes else None,
        "n_test": int(len(y_true)),
    }


def _simulate(lineup: Lineup, seed: int, steps: int, log_dir: str) -> Path:
    env = CartelEnv(n_firms=len(lineup), max_steps=steps, seed=seed)
    agents = [make(i, seed * 10 + i) for i, make in enumerate(lineup)]
    result = run_episode_with_regulator_logging(
        env=env,
        agents=agents,
        regulator=Regulator(seed=seed),
        log_dir=log_dir,
        episode_id=f"bench_{seed}",
        agent_types=[type(a).__name__ for a in agents],
    )
    return Path(result["log_file"])


def benchmark_ml(
    n_episodes: int,
    steps: int,
    model_type: str,
    seed: int,
    scenario: str = "baseline",
) -> dict[str, Any]:
    """Train on one scenario's simulated episodes; evaluate on a held-out split."""
    collusive, competitive = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    log_files: list[str | Path] = []
    labels: list[int] = []

    with tempfile.TemporaryDirectory() as log_dir:
        for i in range(n_episodes):
            label = i % 2
            pool = collusive if label else competitive
            lineup = pool[int(rng.integers(len(pool)))]
            log_files.append(_simulate(lineup, seed + i, steps, log_dir))
            labels.append(label)

        X = FeatureExtractor().extract_features_batch(log_files)

    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    detector = CollusionDetector(model_type=model_type, random_state=seed)
    detector.train(X_train, y_train, validation_split=False)

    y_score = detector.predict_proba(X_test)[:, 1]
    return _metrics(y_test, detector.predict(X_test), y_score)


def _train_pair(job: tuple[float, int, int]) -> list[BaseAgent]:
    discount, train_steps, seed = job
    return list(train_q_learners(discount=discount, steps=train_steps, seed=seed))


def benchmark_tacit(
    n_pairs: int,
    train_steps: int,
    episodes_per_pair: int,
    steps: int,
    model_type: str,
    seed: int,
    explore: float = 0.05,
) -> dict[str, Any]:
    """
    Tacit collusion: patient (discount 0.95, label 1) vs myopic (discount 0,
    label 0) Q-learning pairs, trained in long runs and then frozen.

    Nobody tells these firms to collude, so the label is the condition under
    which collusion is known to emerge (Calvano et al. 2020). Train/test are
    split by trained pair, so test episodes come from pairs never seen in
    training.
    """
    jobs = [
        (discount, train_steps, seed + 100 * k + (0 if discount else 50))
        for discount in (0.95, 0.0)
        for k in range(n_pairs)
    ]
    with ProcessPoolExecutor() as pool:
        pairs = list(pool.map(_train_pair, jobs))

    rng = np.random.default_rng(seed)
    log_files: list[str | Path] = []
    labels: list[int] = []
    groups: list[int] = []
    prices: dict[int, list[float]] = {0: [], 1: []}

    with tempfile.TemporaryDirectory() as log_dir:
        for pair_id, ((discount, _, _), trained) in enumerate(
            zip(jobs, pairs, strict=True)
        ):
            label = int(discount > 0)
            for _ in range(episodes_per_pair):
                episode_seed = int(rng.integers(2**31))

                def clone(i: int, s: int, trained: list = trained) -> BaseAgent:
                    agent = copy.deepcopy(trained[i])
                    agent.np_random = np.random.default_rng(s)
                    agent.freeze(epsilon=explore)
                    return agent

                log = _simulate((clone, clone), episode_seed, steps, log_dir)
                log_files.append(log)
                labels.append(label)
                groups.append(pair_id)
                episode_prices = [
                    np.mean(json.loads(line)["prices"])
                    for line in log.read_text().splitlines()
                    if '"type": "step"' in line
                ]
                prices[label].append(float(np.mean(episode_prices)))

        X = FeatureExtractor().extract_features_batch(log_files)

    y, groups_arr = np.array(labels), np.array(groups)
    test_pairs = {g for g in range(len(jobs)) if g % 2 == 1}
    test = np.isin(groups_arr, list(test_pairs))
    detector = CollusionDetector(model_type=model_type, random_state=seed)
    detector.train(X[~test], y[~test], validation_split=False)

    y_score = detector.predict_proba(X[test])[:, 1]
    metrics = _metrics(y[test], detector.predict(X[test]), y_score)
    metrics["mean_price_patient"] = float(np.mean(prices[1]))
    metrics["mean_price_myopic"] = float(np.mean(prices[0]))
    metrics["collusion_index_patient"] = collusion_index(metrics["mean_price_patient"])
    metrics["collusion_index_myopic"] = collusion_index(metrics["mean_price_myopic"])
    return metrics


def _fmt(value: float | None, pct: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if pct else f"{value:.3f}"


def _table(results: dict[str, dict[str, Any]]) -> str:
    rows = [
        "| Detector | Accuracy | Precision | Recall | F1 | ROC AUC | Test size |",
        "|----------|----------|-----------|--------|----|---------|-----------|",
    ]
    for name, m in results.items():
        rows.append(
            f"| {name} | {_fmt(m['accuracy'])} | {_fmt(m['precision'])} | "
            f"{_fmt(m['recall'])} | {_fmt(m['f1'])} | "
            f"{_fmt(m['roc_auc'], pct=False)} | {m['n_test']} |"
        )
    return "\n".join(rows)


def main() -> None:
    """Run the benchmark and print a markdown table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--episodes", type=int, default=200, help="ML episodes per scenario"
    )
    parser.add_argument("--steps", type=int, default=50, help="Steps per episode")
    parser.add_argument("--model", choices=["logistic", "lightgbm"], default="logistic")
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help=f"Comma-separated ML scenarios (default: all of {', '.join(SCENARIOS)})",
    )
    parser.add_argument(
        "--tacit-pairs",
        type=int,
        default=4,
        help="Q-learning pairs trained per class for the tacit scenario (0 skips it)",
    )
    parser.add_argument(
        "--tacit-train-steps",
        type=int,
        default=150_000,
        help="Training periods per Q-learning pair",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path, help="Also write metrics to this file")
    args = parser.parse_args()

    # Episode runs log at INFO; keep the benchmark output to the table
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    results: dict[str, dict[str, Any]] = {}
    for scenario in args.scenarios.split(","):
        results[f"ML ({args.model}) — {scenario}"] = benchmark_ml(
            args.episodes, args.steps, args.model, args.seed, scenario
        )

    if args.tacit_pairs:
        tacit = benchmark_tacit(
            n_pairs=args.tacit_pairs,
            train_steps=args.tacit_train_steps,
            episodes_per_pair=max(1, args.episodes // (2 * args.tacit_pairs)),
            steps=args.steps,
            model_type=args.model,
            seed=args.seed,
        )
        results[f"ML ({args.model}) — tacit (Q-learning)"] = tacit

    print(_table(results))
    for name, m in results.items():
        if "collusion_index_patient" in m:
            print(
                f"\n{name}: mean price patient {m['mean_price_patient']:.2f} "
                f"(collusion index {m['collusion_index_patient']:.2f}), myopic "
                f"{m['mean_price_myopic']:.2f} "
                f"({m['collusion_index_myopic']:.2f}); Nash 40, monopoly 55"
            )
    print(f"\nseed={args.seed} episodes={args.episodes}/scenario steps={args.steps}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
