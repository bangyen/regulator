#!/usr/bin/env python3
"""
Reproducible detection benchmark.

Generates labeled data with known ground truth, then reports accuracy,
precision, recall, F1 and ROC AUC for:

- the ML episode detector (CollusionDetector), on simulated episodes where the
  label comes from the firms' strategies, across scenarios of increasing
  difficulty (see SCENARIOS)
- the LLM message detector (LLMDetector), on chat-agent template messages and
  on a hand-written set of paraphrased / indirect messages (HARD_MESSAGES)

Usage:
    python scripts/benchmark.py                      # defaults, seed 42
    python scripts/benchmark.py --episodes 200 --model lightgbm --json out.json
    OPENAI_API_KEY=... python scripts/benchmark.py --llm-model gpt-4o-mini
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from collections.abc import Callable
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
from regulator.agents.chat_firm import CollusiveChatAgent, CompetitiveChatAgent
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
from regulator.detectors.llm_detector import LLMDetector
from regulator.detectors.ml_detector import CollusionDetector, FeatureExtractor
from regulator.episode_logging.episode_runner import (
    run_episode_with_regulator_logging,
)

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

# Messages whose intent is not carried by obvious keywords. 1 = collusive.
HARD_MESSAGES: list[tuple[str, int]] = [
    ("It would be a shame if anyone started a race to the bottom.", 1),
    ("We've all seen what happens when someone breaks ranks.", 1),
    ("I'll be holding at 48 next quarter, in case that's useful to anyone.", 1),
    ("Let's keep things civilised in the northern region, it's yours.", 1),
    ("Nobody benefits from those discounts; I'm not planning to repeat them.", 1),
    ("If you stay above 45 I won't have a reason to go lower.", 1),
    ("Same arrangement as last year works for me.", 1),
    ("Happy to leave the hospital contracts to you if the schools stay with us.", 1),
    ("Our list price goes up 8% on the first; you'll see the circular soon.", 1),
    ("There's plenty of margin for everyone if we're all sensible.", 1),
    ("We're cutting prices aggressively to win share this quarter.", 0),
    ("Our new plant lowers unit costs, so expect sharper pricing from us.", 0),
    ("We price independently based on our own costs and demand.", 0),
    ("Customers keep asking for discounts; we'll probably match the market.", 0),
    ("Raw material costs rose 8%, and we're passing that through.", 0),
    ("We don't discuss pricing with competitors.", 0),
    ("Holding prices steady isn't something we can promise anyone.", 0),
    ("We're bidding on every hospital contract this year.", 0),
    ("Demand in the north is soft, so we're running a promotion.", 0),
    ("Our margins are thin; efficiency is the only way forward.", 0),
]


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


def template_messages(n_messages: int, seed: int) -> list[tuple[str, int]]:
    """Labeled messages drawn from the chat agents' templates."""
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
    return messages


def benchmark_llm(
    messages: list[tuple[str, int]], detector: LLMDetector
) -> dict[str, Any]:
    """Classify labeled messages; for a real model also report latency/tokens."""
    results = [
        detector.classify_message(text, sender_id=0, receiver_id=1, step=i)
        for i, (text, _) in enumerate(messages)
    ]
    y_true = np.array([label for _, label in messages])
    y_pred = np.array([int(r["is_collusive"]) for r in results])
    y_score = np.array([r["collusive_probability"] for r in results])
    metrics = _metrics(y_true, y_pred, y_score)

    if detector.model_type == "llm":
        answered = [r for r in results if not r.get("llm_fallback")]
        metrics["fallbacks"] = len(results) - len(answered)
        if answered:
            metrics["mean_latency_s"] = float(
                np.mean([r["latency_s"] for r in answered])
            )
            for key in ("prompt_tokens", "completion_tokens"):
                metrics[key] = int(sum(r["usage"][key] for r in answered))
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
    parser.add_argument("--messages", type=int, default=400, help="Template messages")
    parser.add_argument("--model", choices=["logistic", "lightgbm"], default="logistic")
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help=f"Comma-separated ML scenarios (default: all of {', '.join(SCENARIOS)})",
    )
    parser.add_argument(
        "--llm-model",
        help="Also evaluate this OpenAI model (needs OPENAI_API_KEY and the "
        "llm extra); makes paid API calls",
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

    templates = template_messages(args.messages, args.seed)
    stub = LLMDetector(model_type="stubbed", seed=args.seed)
    results["LLM (stub) — templates"] = benchmark_llm(templates, stub)
    results["LLM (stub) — hard"] = benchmark_llm(HARD_MESSAGES, stub)

    if args.llm_model:
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("--llm-model needs OPENAI_API_KEY")
        llm = LLMDetector(model_type="llm", model_name=args.llm_model)
        # Templates repeat, so the distinct ones are enough for a paid model
        distinct = list(dict.fromkeys(templates))
        results[f"LLM ({args.llm_model}) — templates"] = benchmark_llm(distinct, llm)
        results[f"LLM ({args.llm_model}) — hard"] = benchmark_llm(HARD_MESSAGES, llm)

    print(_table(results))
    for name, m in results.items():
        if "fallbacks" in m:
            print(
                f"\n{name}: {m['fallbacks']} fallbacks, "
                f"mean latency {m.get('mean_latency_s', 0):.2f}s, "
                f"{m.get('prompt_tokens', 0)} prompt + "
                f"{m.get('completion_tokens', 0)} completion tokens"
            )
    print(
        f"\nseed={args.seed} episodes={args.episodes}/scenario steps={args.steps} "
        f"messages={args.messages}"
    )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
