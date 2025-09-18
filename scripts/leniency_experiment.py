#!/usr/bin/env python3
"""
Experiment to compare collusion frequency with and without leniency program.

This script runs simulations comparing market outcomes when the leniency program
is enabled versus disabled, measuring effects on collusion frequency, welfare,
and firm behavior.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.agents.regulator import Regulator
from src.agents.firm_agents import (
    TitForTatAgent,
    WhistleblowerTitForTatAgent,
)
from src.cartel.cartel_env import CartelEnv
from src.agents.leniency import LeniencyProgram


def run_episode_with_leniency(
    env: CartelEnv,
    regulator: Regulator,
    agents: List[Any],
    max_steps: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a single episode with leniency program enabled.

    Args:
        env: The cartel environment
        regulator: The regulator with leniency program
        agents: List of firm agents
        max_steps: Maximum number of steps
        seed: Random seed

    Returns:
        Dictionary containing episode results
    """
    if seed is not None:
        np.random.seed(seed)
        env.np_random = np.random.default_rng(seed)

    # Reset environment and agents
    observation, info = env.reset(seed=seed)
    regulator.reset(n_firms=env.n_firms)
    for agent in agents:
        agent.reset()

    # Episode tracking
    episode_data: Dict[str, Any] = {
        "prices": [],
        "rewards": [],
        "fines": [],
        "parallel_violations": [],
        "structural_break_violations": [],
        "whistleblow_events": [],
        "leniency_reports": [],
        "total_welfare": 0.0,
        "consumer_surplus": 0.0,
        "producer_surplus": 0.0,
    }

    for step in range(max_steps):
        # Get actions from agents
        actions = []
        for i, agent in enumerate(agents):
            action = agent.choose_price(observation, env, info)
            actions.append(action)

        actions_array = np.array(actions, dtype=np.float32)

        # Step environment
        observation, rewards, terminated, truncated, info = env.step(actions_array)

        # Monitor with regulator
        detection_results = regulator.monitor_step(actions_array, step, info)

        # Apply penalties
        modified_rewards = regulator.apply_penalties(rewards, detection_results)

        # Update agent histories
        for i, agent in enumerate(agents):
            rival_prices = np.concatenate([actions_array[:i], actions_array[i + 1 :]])
            agent.update_history(actions_array[i], rival_prices)

        # Check for whistleblowing opportunities
        whistleblow_events = []
        for i, agent in enumerate(agents):
            if hasattr(agent, "evaluate_whistleblow_opportunity"):
                # Calculate collusion probability based on recent violations
                collusion_prob = 0.0
                if detection_results["parallel_violation"]:
                    collusion_prob += 0.4
                if detection_results["structural_break_violation"]:
                    collusion_prob += 0.3

                if collusion_prob > 0:
                    rival_firms = [j for j in range(env.n_firms) if j != i]
                    whistled, incentive = agent.evaluate_whistleblow_opportunity(
                        regulator.fine_amount, collusion_prob, step, rival_firms
                    )

                    if whistled:
                        whistleblow_events.append(
                            {
                                "step": step,
                                "firm_id": i,
                                "incentive": incentive,
                                "collusion_prob": collusion_prob,
                            }
                        )

        # Record episode data
        episode_data["prices"].append(actions_array.tolist())
        episode_data["rewards"].append(modified_rewards.tolist())
        episode_data["fines"].append(detection_results["fines_applied"].tolist())
        episode_data["parallel_violations"].append(
            detection_results["parallel_violation"]
        )
        episode_data["structural_break_violations"].append(
            detection_results["structural_break_violation"]
        )
        episode_data["whistleblow_events"].extend(whistleblow_events)

        # Calculate welfare metrics
        market_price = np.mean(actions_array)
        total_demand = info["total_demand"]
        consumer_surplus = (
            0.5
            * total_demand
            * (env.demand_intercept / (-env.demand_slope) - market_price)
        )
        producer_surplus: float = np.sum(modified_rewards)
        total_welfare = consumer_surplus + producer_surplus

        episode_data["consumer_surplus"] += consumer_surplus
        episode_data["producer_surplus"] += producer_surplus
        episode_data["total_welfare"] += total_welfare

        if terminated or truncated:
            break

    # Get final leniency reports
    episode_data["leniency_reports"] = regulator.get_leniency_reports()

    # Add summary statistics
    episode_data["summary"] = {
        "total_steps": len(episode_data["prices"]),
        "total_parallel_violations": sum(episode_data["parallel_violations"]),
        "total_structural_break_violations": sum(
            episode_data["structural_break_violations"]
        ),
        "total_whistleblow_events": len(episode_data["whistleblow_events"]),
        "total_leniency_reports": len(episode_data["leniency_reports"]),
        "avg_price": float(
            np.mean([np.mean(prices) for prices in episode_data["prices"]])
        ),
        "price_std": float(
            np.std([np.mean(prices) for prices in episode_data["prices"]])
        ),
        "total_fines": float(
            np.sum([np.sum(fines) for fines in episode_data["fines"]])
        ),
        "regulator_summary": regulator.get_violation_summary(),
        "leniency_summary": regulator.get_leniency_summary(),
    }

    return episode_data


def run_episode_without_leniency(
    env: CartelEnv,
    regulator: Regulator,
    agents: List[Any],
    max_steps: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a single episode without leniency program.

    Args:
        env: The cartel environment
        regulator: The regulator without leniency program
        agents: List of firm agents (without whistleblowing capabilities)
        max_steps: Maximum number of steps
        seed: Random seed

    Returns:
        Dictionary containing episode results
    """
    if seed is not None:
        np.random.seed(seed)
        env.np_random = np.random.default_rng(seed)

    # Reset environment and agents
    observation, info = env.reset(seed=seed)
    regulator.reset()  # No n_firms needed when leniency disabled
    for agent in agents:
        agent.reset()

    # Episode tracking
    episode_data: Dict[str, Any] = {
        "prices": [],
        "rewards": [],
        "fines": [],
        "parallel_violations": [],
        "structural_break_violations": [],
        "whistleblow_events": [],
        "leniency_reports": [],
        "total_welfare": 0.0,
        "consumer_surplus": 0.0,
        "producer_surplus": 0.0,
    }

    for step in range(max_steps):
        # Get actions from agents
        actions = []
        for i, agent in enumerate(agents):
            action = agent.choose_price(observation, env, info)
            actions.append(action)

        actions_array = np.array(actions, dtype=np.float32)

        # Step environment
        observation, rewards, terminated, truncated, info = env.step(actions_array)

        # Monitor with regulator
        detection_results = regulator.monitor_step(actions_array, step, info)

        # Apply penalties
        modified_rewards = regulator.apply_penalties(rewards, detection_results)

        # Update agent histories
        for i, agent in enumerate(agents):
            rival_prices = np.concatenate([actions_array[:i], actions_array[i + 1 :]])
            agent.update_history(actions_array[i], rival_prices)

        # Record episode data
        episode_data["prices"].append(actions_array.tolist())
        episode_data["rewards"].append(modified_rewards.tolist())
        episode_data["fines"].append(detection_results["fines_applied"].tolist())
        episode_data["parallel_violations"].append(
            detection_results["parallel_violation"]
        )
        episode_data["structural_break_violations"].append(
            detection_results["structural_break_violation"]
        )

        # Calculate welfare metrics
        market_price = np.mean(actions_array)
        total_demand = info["total_demand"]
        consumer_surplus = (
            0.5
            * total_demand
            * (env.demand_intercept / (-env.demand_slope) - market_price)
        )
        producer_surplus: float = np.sum(modified_rewards)
        total_welfare = consumer_surplus + producer_surplus

        episode_data["consumer_surplus"] += consumer_surplus
        episode_data["producer_surplus"] += producer_surplus
        episode_data["total_welfare"] += total_welfare

        if terminated or truncated:
            break

    # Add summary statistics
    episode_data["summary"] = {
        "total_steps": len(episode_data["prices"]),
        "total_parallel_violations": sum(episode_data["parallel_violations"]),
        "total_structural_break_violations": sum(
            episode_data["structural_break_violations"]
        ),
        "total_whistleblow_events": 0,
        "total_leniency_reports": 0,
        "avg_price": float(
            np.mean([np.mean(prices) for prices in episode_data["prices"]])
        ),
        "price_std": float(
            np.std([np.mean(prices) for prices in episode_data["prices"]])
        ),
        "total_fines": float(
            np.sum([np.sum(fines) for fines in episode_data["fines"]])
        ),
        "regulator_summary": regulator.get_violation_summary(),
        "leniency_summary": {"leniency_enabled": False},
    }

    return episode_data


def run_leniency_experiment(
    n_episodes: int = 50,
    n_firms: int = 3,
    max_steps: int = 100,
    leniency_reduction: float = 0.5,
    fine_amount: float = 50.0,
    output_dir: str = "logs",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the leniency experiment comparing with and without leniency.

    Args:
        n_episodes: Number of episodes to run for each condition
        n_firms: Number of firms in the market
        max_steps: Maximum steps per episode
        leniency_reduction: Fine reduction for whistleblowers
        fine_amount: Base fine amount
        output_dir: Directory to save results
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing experiment results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Initialize environment
    env = CartelEnv(
        n_firms=n_firms,
        max_steps=max_steps,
        marginal_cost=10.0,
        demand_intercept=100.0,
        demand_slope=-1.0,
        shock_std=5.0,
        price_min=1.0,
        price_max=100.0,
        seed=seed,
    )

    # Initialize regulators
    regulator_with_leniency = Regulator(
        parallel_threshold=2.0,
        parallel_steps=3,
        structural_break_threshold=10.0,
        fine_amount=fine_amount,
        leniency_enabled=True,
        leniency_reduction=leniency_reduction,
        seed=seed,
    )

    regulator_without_leniency = Regulator(
        parallel_threshold=2.0,
        parallel_steps=3,
        structural_break_threshold=10.0,
        fine_amount=fine_amount,
        leniency_enabled=False,
        seed=seed,
    )

    # Initialize leniency program for whistleblower agents
    leniency_program = LeniencyProgram(
        leniency_reduction=leniency_reduction,
        seed=seed,
    )

    # Initialize agents
    agents_with_leniency = [
        WhistleblowerTitForTatAgent(i, leniency_program, seed=seed)
        for i in range(n_firms)
    ]

    agents_without_leniency = [TitForTatAgent(i, seed=seed) for i in range(n_firms)]

    # Run experiments
    print(f"Running leniency experiment with {n_episodes} episodes per condition...")

    # With leniency
    print("Running episodes with leniency program...")
    episodes_with_leniency = []
    for episode in range(n_episodes):
        if episode % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}")

        episode_seed = seed + episode if seed is not None else None
        episode_data = run_episode_with_leniency(
            env, regulator_with_leniency, agents_with_leniency, max_steps, episode_seed
        )
        episodes_with_leniency.append(episode_data)

    # Without leniency
    print("Running episodes without leniency program...")
    episodes_without_leniency = []
    for episode in range(n_episodes):
        if episode % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}")

        episode_seed = seed + episode + n_episodes if seed is not None else None
        episode_data = run_episode_without_leniency(
            env,
            regulator_without_leniency,
            agents_without_leniency,
            max_steps,
            episode_seed,
        )
        episodes_without_leniency.append(episode_data)

    # Analyze results
    print("Analyzing results...")

    # Calculate aggregate statistics
    def calculate_aggregate_stats(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
        summaries = [ep["summary"] for ep in episodes]

        return {
            "avg_parallel_violations": float(
                np.mean([s["total_parallel_violations"] for s in summaries])
            ),
            "avg_structural_break_violations": float(
                np.mean([s["total_structural_break_violations"] for s in summaries])
            ),
            "avg_whistleblow_events": float(
                np.mean([s["total_whistleblow_events"] for s in summaries])
            ),
            "avg_leniency_reports": float(
                np.mean([s["total_leniency_reports"] for s in summaries])
            ),
            "avg_price": float(np.mean([s["avg_price"] for s in summaries])),
            "avg_price_std": float(np.mean([s["price_std"] for s in summaries])),
            "avg_total_fines": float(np.mean([s["total_fines"] for s in summaries])),
            "avg_welfare": float(np.mean([ep["total_welfare"] for ep in episodes])),
            "avg_consumer_surplus": float(
                np.mean([ep["consumer_surplus"] for ep in episodes])
            ),
            "avg_producer_surplus": np.mean(
                [ep["producer_surplus"] for ep in episodes]
            ),
        }

    stats_with_leniency = calculate_aggregate_stats(episodes_with_leniency)
    stats_without_leniency = calculate_aggregate_stats(episodes_without_leniency)

    # Calculate differences
    differences = {}
    for key in stats_with_leniency:
        differences[f"{key}_difference"] = (
            stats_with_leniency[key] - stats_without_leniency[key]
        )
        differences[f"{key}_percent_change"] = (
            (stats_with_leniency[key] - stats_without_leniency[key])
            / stats_without_leniency[key]
            * 100
            if stats_without_leniency[key] != 0
            else 0
        )

    # Compile results
    results = {
        "experiment_config": {
            "n_episodes": n_episodes,
            "n_firms": n_firms,
            "max_steps": max_steps,
            "leniency_reduction": leniency_reduction,
            "fine_amount": fine_amount,
            "seed": seed,
        },
        "stats_with_leniency": stats_with_leniency,
        "stats_without_leniency": stats_without_leniency,
        "differences": differences,
        "episodes_with_leniency": episodes_with_leniency,
        "episodes_without_leniency": episodes_without_leniency,
    }

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"leniency_experiment_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("LENIENCY EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Episodes per condition: {n_episodes}")
    print(f"Firms: {n_firms}")
    print(f"Leniency reduction: {leniency_reduction:.1%}")
    print()

    print("KEY METRICS:")
    print("  Collusion violations (parallel pricing):")
    print(f"    With leniency:    {stats_with_leniency['avg_parallel_violations']:.2f}")
    print(
        f"    Without leniency: {stats_without_leniency['avg_parallel_violations']:.2f}"
    )
    print(
        f"    Change:           {differences['avg_parallel_violations_percent_change']:+.1f}%"
    )
    print()

    print("  Whistleblow events:")
    print(f"    With leniency:    {stats_with_leniency['avg_whistleblow_events']:.2f}")
    print(
        f"    Without leniency: {stats_without_leniency['avg_whistleblow_events']:.2f}"
    )
    print()

    print("  Average market price:")
    print(f"    With leniency:    ${stats_with_leniency['avg_price']:.2f}")
    print(f"    Without leniency: ${stats_without_leniency['avg_price']:.2f}")
    print(f"    Change:           {differences['avg_price_percent_change']:+.1f}%")
    print()

    print("  Total welfare:")
    print(f"    With leniency:    ${stats_with_leniency['avg_welfare']:.2f}")
    print(f"    Without leniency: ${stats_without_leniency['avg_welfare']:.2f}")
    print(f"    Change:           {differences['avg_welfare_percent_change']:+.1f}%")
    print()

    print("  Consumer surplus:")
    print(f"    With leniency:    ${stats_with_leniency['avg_consumer_surplus']:.2f}")
    print(
        f"    Without leniency: ${stats_without_leniency['avg_consumer_surplus']:.2f}"
    )
    print(
        f"    Change:           {differences['avg_consumer_surplus_percent_change']:+.1f}%"
    )
    print()

    # Validation checks
    print("VALIDATION CHECKS:")

    # Check 1: Whistleblowing reduces fine for reporting firm
    whistleblow_events = sum(
        len(ep["whistleblow_events"]) for ep in episodes_with_leniency
    )
    leniency_reports = sum(len(ep["leniency_reports"]) for ep in episodes_with_leniency)
    print(f"  ✓ Whistleblow events: {whistleblow_events}")
    print(f"  ✓ Leniency reports: {leniency_reports}")

    # Check 2: Collusion probability drops
    collusion_reduction = differences["avg_parallel_violations_percent_change"]
    if collusion_reduction < 0:
        print(f"  ✓ Collusion frequency reduced by {abs(collusion_reduction):.1f}%")
    else:
        print(f"  ✗ Collusion frequency increased by {collusion_reduction:.1f}%")

    # Check 3: Welfare improves
    welfare_change = differences["avg_welfare_percent_change"]
    if welfare_change > 0:
        print(f"  ✓ Welfare improved by {welfare_change:.1f}%")
    else:
        print(f"  ✗ Welfare decreased by {abs(welfare_change):.1f}%")

    print("=" * 60)

    return results


def main() -> Dict[str, Any]:
    """Main function to run the leniency experiment."""
    parser = argparse.ArgumentParser(description="Run leniency program experiment")
    parser.add_argument(
        "--episodes", type=int, default=50, help="Number of episodes per condition"
    )
    parser.add_argument("--firms", type=int, default=3, help="Number of firms")
    parser.add_argument(
        "--steps", type=int, default=100, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--leniency-reduction",
        type=float,
        default=0.5,
        help="Leniency reduction factor",
    )
    parser.add_argument(
        "--fine-amount", type=float, default=50.0, help="Base fine amount"
    )
    parser.add_argument(
        "--output-dir", type=str, default="logs", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    results = run_leniency_experiment(
        n_episodes=args.episodes,
        n_firms=args.firms,
        max_steps=args.steps,
        leniency_reduction=args.leniency_reduction,
        fine_amount=args.fine_amount,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    return results


if __name__ == "__main__":
    main()
