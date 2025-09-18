#!/usr/bin/env python3
"""
CLI demo script for testing the Regulator with 3 RandomAgents.

This script runs a 20-step simulation with 3 RandomAgents and a Regulator,
demonstrating cartel detection and penalty application. The script confirms
that fines trigger appropriately when violations are detected.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.agents.firm_agents import RandomAgent
from src.agents.regulator import Regulator
from src.cartel.cartel_env import CartelEnv


def run_regulator_demo(
    n_steps: int = 20,
    n_firms: int = 3,
    parallel_threshold: float = 2.0,
    parallel_steps: int = 3,
    structural_break_threshold: float = 10.0,
    fine_amount: float = 50.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run a demonstration of the Regulator with RandomAgents.

    Args:
        n_steps: Number of steps to run the simulation
        n_firms: Number of firms in the market
        parallel_threshold: Threshold for parallel pricing detection
        parallel_steps: Minimum steps for parallel pricing detection
        structural_break_threshold: Threshold for structural break detection
        fine_amount: Fine amount to apply for violations
        seed: Random seed for reproducibility
        verbose: Whether to print detailed output

    Returns:
        Dictionary containing simulation results and statistics
    """
    # Initialize environment
    env = CartelEnv(
        n_firms=n_firms,
        max_steps=n_steps,
        price_min=10.0,
        price_max=100.0,
        seed=seed,
    )

    # Initialize agents
    agents = [RandomAgent(agent_id=i, seed=seed + i) for i in range(n_firms)]

    # Initialize regulator
    regulator = Regulator(
        parallel_threshold=parallel_threshold,
        parallel_steps=parallel_steps,
        structural_break_threshold=structural_break_threshold,
        fine_amount=fine_amount,
        seed=seed,
    )

    # Reset environment and agents
    obs, _ = env.reset(seed=seed)
    for agent in agents:
        agent.reset()
    regulator.reset()

    # Track simulation results
    results = {
        "step_data": [],
        "total_rewards": np.zeros(n_firms),
        "total_fines": 0.0,
        "violations": {"parallel": 0, "structural_break": 0},
        "final_stats": {},
    }

    if verbose:
        print(
            f"Running {n_steps}-step simulation with {n_firms} RandomAgents + Regulator"
        )
        print(
            f"Regulator settings: parallel_threshold={parallel_threshold}, "
            f"parallel_steps={parallel_steps}, structural_break_threshold={structural_break_threshold}, "
            f"fine_amount={fine_amount}"
        )
        print("-" * 80)

    # Run simulation
    for step in range(n_steps):
        # Agents choose prices
        prices = []
        for i, agent in enumerate(agents):
            price = agent.choose_price(obs, env)
            prices.append(price)

        prices_array = np.array(prices, dtype=np.float32)

        # Regulator monitors the step
        detection_results = regulator.monitor_step(prices_array, step)

        # Environment step
        obs, rewards, terminated, truncated, info = env.step(prices_array)

        # Apply regulator penalties
        modified_rewards = regulator.apply_penalties(rewards, detection_results)

        # Update agent histories
        for i, agent in enumerate(agents):
            rival_prices = np.array([prices[j] for j in range(len(prices)) if j != i])
            agent.update_history(prices[i], rival_prices)

        # Track results
        step_data = {
            "step": step,
            "prices": prices_array.copy(),
            "original_rewards": rewards.copy(),
            "modified_rewards": modified_rewards.copy(),
            "fines_applied": detection_results["fines_applied"].copy(),
            "parallel_violation": detection_results["parallel_violation"],
            "structural_break_violation": detection_results[
                "structural_break_violation"
            ],
            "violation_details": detection_results["violation_details"].copy(),
        }
        results["step_data"].append(step_data)

        # Update totals
        results["total_rewards"] += modified_rewards
        results["total_fines"] += np.sum(detection_results["fines_applied"])

        if detection_results["parallel_violation"]:
            results["violations"]["parallel"] += 1
        if detection_results["structural_break_violation"]:
            results["violations"]["structural_break"] += 1

        # Print step details if verbose
        if verbose:
            print(
                f"Step {step:2d}: Prices={prices_array}, "
                f"Rewards={rewards}→{modified_rewards}, "
                f"Fines={detection_results['fines_applied']}"
            )
            if detection_results["violation_details"]:
                for detail in detection_results["violation_details"]:
                    print(f"         ⚠️  {detail}")

        if truncated:
            break

    # Get final statistics
    violation_summary = regulator.get_violation_summary()
    price_stats = regulator.get_price_statistics()
    parallel_ratio = regulator.get_parallel_pricing_ratio()

    results["final_stats"] = {
        "violation_summary": violation_summary,
        "price_statistics": price_stats,
        "parallel_pricing_ratio": parallel_ratio,
        "total_steps": len(results["step_data"]),
    }

    if verbose:
        print("-" * 80)
        print("SIMULATION COMPLETE")
        print(f"Total steps: {len(results['step_data'])}")
        print(f"Total fines applied: ${results['total_fines']:.2f}")
        print(f"Parallel pricing violations: {results['violations']['parallel']}")
        print(
            f"Structural break violations: {results['violations']['structural_break']}"
        )
        print(f"Parallel pricing ratio: {parallel_ratio:.3f}")
        print(f"Final firm rewards: {results['total_rewards']}")
        print(
            f"Price statistics: mean={price_stats.get('mean_price', 0):.2f}, "
            f"std={price_stats.get('std_price', 0):.2f}, "
            f"range={price_stats.get('price_range', 0):.2f}"
        )

    return results


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Demo script for Regulator with RandomAgents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of simulation steps"
    )
    parser.add_argument("--firms", type=int, default=3, help="Number of firms")
    parser.add_argument(
        "--parallel-threshold",
        type=float,
        default=2.0,
        help="Threshold for parallel pricing detection",
    )
    parser.add_argument(
        "--parallel-steps",
        type=int,
        default=3,
        help="Minimum steps for parallel pricing detection",
    )
    parser.add_argument(
        "--structural-break-threshold",
        type=float,
        default=10.0,
        help="Threshold for structural break detection",
    )
    parser.add_argument(
        "--fine-amount", type=float, default=50.0, help="Fine amount for violations"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    try:
        results = run_regulator_demo(
            n_steps=args.steps,
            n_firms=args.firms,
            parallel_threshold=args.parallel_threshold,
            parallel_steps=args.parallel_steps,
            structural_break_threshold=args.structural_break_threshold,
            fine_amount=args.fine_amount,
            seed=args.seed,
            verbose=not args.quiet,
        )

        # Check if fines were triggered (validation requirement)
        if results["total_fines"] > 0:
            print(
                f"\n✅ SUCCESS: Fines were triggered! Total fines: ${results['total_fines']:.2f}"
            )
            return 0
        else:
            print("\n⚠️  WARNING: No fines were triggered in this run.")
            print("This may be normal for random agents, but try adjusting parameters:")
            print(
                "  --parallel-threshold 1.0 --parallel-steps 2 --structural-break-threshold 5.0"
            )
            return 1

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
