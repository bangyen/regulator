#!/usr/bin/env python3
"""
Screening study: label-free collusion screens vs competitive and collusive
markets, including tacitly colluding Q-learners.

Thresholds are calibrated on competitive simulations only (target
false-positive rate --alpha); flag rates and AUC are measured on fresh
episodes. See regulator.experiments.screening for the protocol.

Usage:
    python scripts/screen_study.py                 # ~1-2 min on 4 cores
    python scripts/screen_study.py --episodes 100 --json study.json
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from regulator.experiments.screening import format_study, run_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--episodes", type=int, default=40, help="Test episodes per population"
    )
    parser.add_argument(
        "--calibration-episodes",
        type=int,
        default=60,
        help="Competitive episodes per population for thresholds",
    )
    parser.add_argument("--steps", type=int, default=100, help="Steps per episode")
    parser.add_argument(
        "--q-pairs", type=int, default=4, help="Q-learning pairs per discount"
    )
    parser.add_argument(
        "--q-train-steps", type=int, default=150_000, help="Training periods per pair"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Target false-positive rate"
    )
    parser.add_argument(
        "--burn-in", type=int, default=10, help="Initial steps screens ignore"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="Also write results to this file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    result = run_study(
        episodes=args.episodes,
        calibration_episodes=args.calibration_episodes,
        steps=args.steps,
        q_pairs=args.q_pairs,
        q_train_steps=args.q_train_steps,
        alpha=args.alpha,
        burn_in=args.burn_in,
        seed=args.seed,
    )
    print(format_study(result))
    print(
        f"\nalpha={args.alpha} episodes={args.episodes}/population "
        f"steps={args.steps} seed={args.seed}"
    )
    if args.json:
        data = asdict(result)
        data["populations"] = [
            {"name": p.name, "colludes": p.colludes} for p in result.populations
        ]
        args.json.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
