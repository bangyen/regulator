#!/usr/bin/env python3
"""
Economic Validation Demonstration

This script demonstrates the economic validation capabilities by:
1. Running a sample episode
2. Validating the economic data
3. Showing plausibility checks
4. Identifying any economic inconsistencies
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cartel.cartel_env import CartelEnv
from agents.firm_agents import RandomAgent, BestResponseAgent, TitForTatAgent
from episode_logging.episode_runner import run_episode_with_logging
from economic_validation import validate_economic_data, check_economic_plausibility


def run_validation_demo():
    """Run economic validation demonstration."""
    print("🔍 Economic Validation Demonstration")
    print("=" * 50)

    # Create environment
    env = CartelEnv(
        n_firms=3,
        max_steps=20,
        marginal_cost=10.0,
        demand_intercept=100.0,
        demand_slope=-1.0,
        shock_std=5.0,
        price_min=1.0,
        price_max=100.0,
        seed=42,
    )

    # Create agents
    agents = [
        RandomAgent(0, seed=42),
        BestResponseAgent(1, seed=43),
        TitForTatAgent(2, seed=44),
    ]

    agent_types = ["random", "bestresponse", "titfortat"]

    print("Running episode with economic validation...")

    # Run episode with logging
    result = run_episode_with_logging(
        env=env,
        agents=agents,
        log_dir="logs",
        episode_id="validation_demo",
        agent_types=agent_types,
    )

    print(f"Episode completed: {len(result.get('episode_prices', []))} steps")
    print(f"Final total profits: {result.get('total_rewards', 'N/A')}")

    # Load episode data for validation
    log_file = Path("logs/validation_demo.jsonl")
    if log_file.exists():
        from episode_logging.logger import Logger

        episode_data = Logger.load_episode_data(log_file)

        print("\n" + "=" * 50)
        print("ECONOMIC VALIDATION RESULTS")
        print("=" * 50)

        # Validate economic data
        is_valid, errors = validate_economic_data(episode_data)

        if is_valid:
            print("✅ Economic data is VALID - all constraints satisfied")
        else:
            print("❌ Economic data has ERRORS:")
            for error in errors:
                print(f"  - {error}")

        # Check economic plausibility
        print("\n" + "-" * 30)
        print("ECONOMIC PLAUSIBILITY CHECKS")
        print("-" * 30)

        plausibility = check_economic_plausibility(episode_data)

        if plausibility.get("overall_plausible", False):
            print("✅ Data is economically plausible")
        else:
            print("⚠️  Data shows some economic implausibilities")

        # Show detailed statistics
        print("\nPrice Statistics:")
        price_stats = plausibility["price_stats"]
        print(f"  Mean: {price_stats['mean']:.2f}")
        print(f"  Std:  {price_stats['std']:.2f}")
        print(f"  Range: {price_stats['min']:.2f} - {price_stats['max']:.2f}")

        print("\nProfit Statistics:")
        profit_stats = plausibility["profit_stats"]
        print(f"  Mean: {profit_stats['mean']:.2f}")
        print(f"  Negative profits: {profit_stats['negative_profits']}")
        print(f"  Total negative: {profit_stats['total_negative_profits']:.2f}")

        print("\nDemand Statistics:")
        demand_stats = plausibility["demand_stats"]
        print(f"  Mean: {demand_stats['mean']:.2f}")
        print(f"  Zero demand steps: {demand_stats['zero_demand_steps']}")

        print("\nPlausibility Checks:")
        checks = plausibility["plausibility_checks"]
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check_name}: {result}")

        # Show sample step validation
        if episode_data["steps"]:
            print("\n" + "-" * 30)
            print("SAMPLE STEP VALIDATION")
            print("-" * 30)

            from economic_validation import EconomicValidator

            validator = EconomicValidator()

            sample_step = episode_data["steps"][0]
            is_valid, errors = validator.validate_step_data(
                prices=sample_step["prices"],
                market_price=sample_step["market_price"],
                total_demand=sample_step["total_demand"],
                individual_quantities=sample_step.get("individual_quantity", []),
                market_shares=sample_step.get("market_shares", []),
                profits=sample_step["profits"],
                demand_shock=sample_step["demand_shock"],
            )

            print(f"Step 1 validation: {'✅ VALID' if is_valid else '❌ INVALID'}")
            if errors:
                for error in errors:
                    print(f"  - {error}")

    else:
        print("❌ Log file not found - validation cannot be performed")


if __name__ == "__main__":
    run_validation_demo()
