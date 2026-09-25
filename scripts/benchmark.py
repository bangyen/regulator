#!/usr/bin/env python3
"""
Reproducible detection benchmark.

Generates labeled data with known ground truth, then reports accuracy,
precision, recall, F1 and ROC AUC for:

- the ML episode detector (CollusionDetector), on simulated episodes where the
  label comes from the agents' strategies (collusive vs competitive firms)
- the LLM message detector (LLMDetector), on chat messages drawn from the
  collusive and non-collusive templates of the chat agents

Usage:
    python scripts/benchmark.py                      # defaults, seed 42
    python scripts/benchmark.py --episodes 200 --model lightgbm --json out.json
"""

import argparse
import json
import logging
import sys
import tempfile
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

from regulator.agents.chat_firm import CollusiveChatAgent, CompetitiveChatAgent
from regulator.agents.firm_agents import (
    BaseAgent,
    BestResponseAgent,
    CollusiveAgent,
    RandomAgent,
    TitForTatAgent,
)
from regulator.agents.regulator import Regulator
from regulator.agents.stealth_agent import StealthCollusiveAgent
from regulator.cartel.cartel_env import CartelEnv
from regulator.detectors.llm_detector import LLMDetector
from regulator.detectors.ml_detector import CollusionDetector, FeatureExtractor
from regulator.episode_logging.episode_runner import (
    run_episode_with_regulator_logging,
)

# Firm line-ups per class. Ground truth is the strategy, not a heuristic.
COLLUSIVE_LINEUPS = [
    (CollusiveAgent, CollusiveAgent),
    (StealthCollusiveAgent, StealthCollusiveAgent),
    (CollusiveAgent, StealthCollusiveAgent),
]
COMPETITIVE_LINEUPS = [
    (RandomAgent, RandomAgent),
    (BestResponseAgent, BestResponseAgent),
    (RandomAgent, BestResponseAgent),
    (TitForTatAgent, RandomAgent),
]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "n_test": int(len(y_true)),
    }


def _simulate(
    lineup: tuple[type[BaseAgent], ...], seed: int, steps: int, log_dir: str
) -> Path:
    env = CartelEnv(n_firms=len(lineup), max_steps=steps, seed=seed)
    agents = [cls(agent_id=i, seed=seed * 10 + i) for i, cls in enumerate(lineup)]
    result = run_episode_with_regulator_logging(
        env=env,
        agents=agents,
        regulator=Regulator(seed=seed),
        log_dir=log_dir,
        episode_id=f"bench_{seed}",
        agent_types=[cls.__name__ for cls in lineup],
    )
    return Path(result["log_file"])


def benchmark_ml(
    n_episodes: int, steps: int, model_type: str, seed: int
) -> dict[str, Any]:
    """Train on simulated episodes and evaluate on a held-out split."""
    rng = np.random.default_rng(seed)
    log_files: list[str | Path] = []
    labels: list[int] = []

    with tempfile.TemporaryDirectory() as log_dir:
        for i in range(n_episodes):
            label = i % 2
            pool = COLLUSIVE_LINEUPS if label else COMPETITIVE_LINEUPS
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


def benchmark_llm(n_messages: int, seed: int) -> dict[str, Any]:
    """Classify labeled template messages with the stubbed LLM detector."""
    collusive = CollusiveChatAgent(agent_id=0, collusion_intensity=1.0, seed=seed)
    normal = CollusiveChatAgent(agent_id=1, collusion_intensity=0.0, seed=seed + 1)
    competitive = CompetitiveChatAgent(agent_id=2, seed=seed + 2)
    obs = np.zeros(2)
    env = CartelEnv(n_firms=2, seed=seed)

    messages: list[tuple[str, int]] = []
    for i in range(n_messages):
        if i % 2:
            messages.append((collusive._generate_base_message(obs, env), 1))
        elif i % 4 == 0:
            messages.append((normal._generate_base_message(obs, env), 0))
        else:
            messages.append((competitive._generate_base_message(obs, env), 0))

    detector = LLMDetector(model_type="stubbed")
    results = [
        detector.classify_message(text, sender_id=0, receiver_id=1, step=i)
        for i, (text, _) in enumerate(messages)
    ]
    y_true = np.array([label for _, label in messages])
    y_pred = np.array([int(r["is_collusive"]) for r in results])
    y_score = np.array([r["collusive_probability"] for r in results])
    return _metrics(y_true, y_pred, y_score)


def _table(results: dict[str, dict[str, Any]]) -> str:
    rows = [
        "| Detector | Accuracy | Precision | Recall | F1 | ROC AUC | Test size |",
        "|----------|----------|-----------|--------|----|---------|-----------|",
    ]
    for name, m in results.items():
        rows.append(
            f"| {name} | {m['accuracy']:.1%} | {m['precision']:.1%} | "
            f"{m['recall']:.1%} | {m['f1']:.1%} | {m['roc_auc']:.3f} | {m['n_test']} |"
        )
    return "\n".join(rows)


def main() -> None:
    """Run the benchmark and print a markdown table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episodes", type=int, default=200, help="ML episodes")
    parser.add_argument("--steps", type=int, default=50, help="Steps per episode")
    parser.add_argument("--messages", type=int, default=400, help="LLM messages")
    parser.add_argument("--model", choices=["logistic", "lightgbm"], default="logistic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path, help="Also write metrics to this file")
    args = parser.parse_args()

    # Episode runs log at INFO; keep the benchmark output to the table
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    results = {
        f"ML ({args.model})": benchmark_ml(
            args.episodes, args.steps, args.model, args.seed
        ),
        "LLM (stubbed)": benchmark_llm(args.messages, args.seed),
    }

    print(_table(results))
    print(
        f"\nseed={args.seed} episodes={args.episodes} steps={args.steps} "
        f"messages={args.messages}"
    )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
