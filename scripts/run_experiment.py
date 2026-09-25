#!/usr/bin/env python3
"""
CLI for running regulator experiments.

Thin argparse wrapper around regulator.experiments.experiment_runner that also
exposes the environment parameters as flags.
"""

import argparse
import logging
import sys

from regulator.experiments.experiment_runner import (
    AGENT_TYPES,
    REGULATOR_CONFIGS,
    run_experiment,
)


def main() -> None:
    """Main entry point for the CLI."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Run regulator experiments with various agent types and configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_experiment.py --firms random,bestresponse,titfortat --steps 100 --regulator ml
  python scripts/run_experiment.py --firms random,random --steps 50 --regulator rule_based --seed 123
  python scripts/run_experiment.py --firms stealth,stealth --steps 100 --regulator enhanced
  python scripts/run_experiment.py --firms titfortat,titfortat --steps 200 --regulator none  # Unregulated
        """,
    )

    parser.add_argument(
        "--firms",
        type=str,
        required=True,
        help=f"Comma-separated list of agent types ({', '.join(AGENT_TYPES)})",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps to run (default: 100)",
    )

    parser.add_argument(
        "--regulator",
        type=str,
        default="rule_based",
        choices=[*REGULATOR_CONFIGS, "disabled"],
        help="Regulator configuration (default: rule_based). 'none' (alias "
        "'disabled') runs without a regulator.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save log files (default: logs)",
    )

    parser.add_argument(
        "--episode-id",
        type=str,
        help="Custom episode ID (auto-generated if not provided)",
    )

    # Environment parameters
    parser.add_argument(
        "--marginal-cost",
        type=float,
        default=10.0,
        help="Marginal cost for all firms (default: 10.0)",
    )

    parser.add_argument(
        "--demand-intercept",
        type=float,
        default=100.0,
        help="Demand curve intercept (default: 100.0)",
    )

    parser.add_argument(
        "--demand-slope",
        type=float,
        default=-1.0,
        help="Demand curve slope (default: -1.0)",
    )

    parser.add_argument(
        "--shock-std",
        type=float,
        default=5.0,
        help="Standard deviation of demand shocks (default: 5.0)",
    )

    parser.add_argument(
        "--price-min",
        type=float,
        default=1.0,
        help="Minimum allowed price (default: 1.0)",
    )

    parser.add_argument(
        "--price-max",
        type=float,
        default=100.0,
        help="Maximum allowed price (default: 100.0)",
    )

    args = parser.parse_args()

    # Parse firms list
    firms = [firm.strip() for firm in args.firms.split(",")]

    # Validate agent types
    for firm in firms:
        if firm.lower().replace("_", "").replace("-", "") not in AGENT_TYPES:
            print(
                f"Error: Unknown agent type '{firm}'. Valid types: {', '.join(AGENT_TYPES)}"
            )
            sys.exit(1)

    # Create environment parameters
    env_params = {
        "marginal_cost": args.marginal_cost,
        "demand_intercept": args.demand_intercept,
        "demand_slope": args.demand_slope,
        "shock_std": args.shock_std,
        "price_min": args.price_min,
        "price_max": args.price_max,
    }

    try:
        # Run experiment
        results = run_experiment(
            firms=firms,
            steps=args.steps,
            regulator_config=args.regulator,
            seed=args.seed,
            log_dir=args.log_dir,
            episode_id=args.episode_id,
            env_params=env_params,
        )

        print("\nExperiment completed successfully!")
        print(f"Results saved to: {results['log_file']}")

    except Exception as e:
        print(f"Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
