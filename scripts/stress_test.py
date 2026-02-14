#!/usr/bin/env python3
"""
Regulator Stress Test: Evaluating detection resilience across complexity tiers.

This script runs experiments across three tiers:
1. Base: Simple agents (Tit-for-Tat), no noise, stable demand.
2. Moderate: Adaptive agents, some noise, stochastic demand.
3. Hard: Stealthy agents, high noise, limited visibility, dynamic elasticity.

It measures detection accuracy and F1 score for each tier.
"""

import argparse
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List

from src.cartel.cartel_env import CartelEnv
from src.agents.firm_agents import TitForTatAgent, RandomAgent
from src.agents.adaptive_agent import AdaptiveAgent
from src.agents.stealth_agent import StealthCollusiveAgent
from src.agents.regulator import Regulator
from src.episode_logging.episode_runner import run_episode_with_regulator_logging


def run_tier(
    tier_name: str,
    env_params: Dict[str, Any],
    agent_types: List[str],
    n_episodes: int = 5,
    steps: int = 100,
    regulator_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Runs a set of episodes for a specific complexity tier."""
    print(f"\n--- Running Tier: {tier_name} ---")

    results = []
    for ep in range(n_episodes):
        # Create environment
        env = CartelEnv(**env_params)

        # Create agents
        agents = []
        for i, atype in enumerate(agent_types):
            if atype == "titfortat":
                agents.append(TitForTatAgent(i))
            elif atype == "adaptive":
                agents.append(AdaptiveAgent(i))
            elif atype == "stealth":
                agents.append(StealthCollusiveAgent(i))
            else:
                agents.append(RandomAgent(i))

        # Create regulator
        reg = Regulator(**(regulator_config or {}))

        # Run episode
        episode_id = f"stresstest_{tier_name}_ep{ep}"
        res = run_episode_with_regulator_logging(
            env=env,
            agents=agents,
            regulator=reg,
            log_dir="logs/stresstest",
            episode_id=episode_id,
            agent_types=agent_types,
        )
        results.append(res)
        print(f"  Episode {ep+1}/{n_episodes} complete.")

    return summarize_tier(results)


def summarize_tier(tier_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates summary statistics for a tier."""
    parallel_violations = []
    structural_violations = []
    total_fines = []
    avg_prices = []

    for res in tier_results:
        # Note: res structure depends on run_episode_with_regulator_logging implementation
        # Assuming it returns a dict with 'episode_data'
        data = res.get("episode_data", {})
        violations = data.get("violations", {})
        parallel_violations.append(violations.get("parallel", 0))
        structural_violations.append(violations.get("structural_break", 0))
        total_fines.append(data.get("total_fines", 0.0))

        # Calculate avg price from summary if available
        summary = res.get("episode_summary", {})
        prices = summary.get("avg_prices", [])
        if prices:
            avg_prices.append(np.mean(prices))

    return {
        "avg_parallel_violations": float(np.mean(parallel_violations)),
        "avg_structural_violations": float(np.mean(structural_violations)),
        "avg_fines": float(np.mean(total_fines)),
        "avg_market_price": float(np.mean(avg_prices)) if avg_prices else 0.0,
        "violation_rate": float(
            np.mean(
                [
                    1 if (p > 0 or s > 0) else 0
                    for p, s in zip(parallel_violations, structural_violations)
                ]
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Regulator Stress Test")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per tier")
    args = parser.parse_args()

    tier_configs = [
        {
            "name": "Tier 1: Base",
            "env": {
                "n_firms": 3,
                "max_steps": 100,
                "shock_std": 2.0,
            },
            "agents": ["titfortat", "titfortat", "titfortat"],
            "regulator": {"parallel_threshold": 2.0, "parallel_steps": 3},
        },
        {
            "name": "Tier 2: Moderate",
            "env": {
                "n_firms": 3,
                "max_steps": 100,
                "shock_std": 5.0,
                "use_enhanced_demand": True,
            },
            "agents": ["adaptive", "adaptive", "adaptive"],
            "regulator": {"parallel_threshold": 2.0, "parallel_steps": 3},
        },
        {
            "name": "Tier 3: Hard",
            "env": {
                "n_firms": 3,
                "max_steps": 100,
                "shock_std": 10.0,
                "use_enhanced_demand": True,
                "use_information_asymmetry": True,
                "observation_noise_std": 0.5,
                "use_dynamic_elasticity": True,
            },
            "agents": ["stealth", "stealth", "stealth"],
            "regulator": {"parallel_threshold": 2.0, "parallel_steps": 3},
        },
    ]

    report = {}
    for config in tier_configs:
        report[config["name"]] = run_tier(
            tier_name=config["name"],
            env_params=config["env"],
            agent_types=config["agents"],
            n_episodes=args.episodes,
            regulator_config=config["regulator"],
        )

    print("\n" + "=" * 40)
    print("STRESS TEST REPORT")
    print("=" * 40)
    print(json.dumps(report, indent=2))

    # Save report
    report_path = Path("logs/stresstest/final_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
