#!/usr/bin/env python3
"""
CLI for running regulator experiments.

This script provides a command-line interface for running cartel detection
experiments with various agent types, regulator configurations, and analysis.
"""

import argparse
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Import from the package
from agents.firm_agents import (
    BaseAgent,
    RandomAgent,
    BestResponseAgent,
    TitForTatAgent,
)
from agents.regulator import Regulator
from cartel.cartel_env import CartelEnv
from episode_logging.episode_runner import (
    run_episode_with_regulator_logging,
)


def create_agent(
    agent_type: str, agent_id: int, seed: Optional[int] = None
) -> BaseAgent:
    """
    Create an agent of the specified type.

    Args:
        agent_type: Type of agent to create ('random', 'bestresponse', 'titfortat')
        agent_id: Unique identifier for the agent
        seed: Random seed for reproducibility

    Returns:
        Agent instance
    """
    agent_type = agent_type.lower()

    if agent_type == "random":
        return RandomAgent(agent_id=agent_id, seed=seed)
    elif agent_type == "bestresponse":
        return BestResponseAgent(agent_id=agent_id, seed=seed)
    elif agent_type == "titfortat":
        return TitForTatAgent(agent_id=agent_id, seed=seed)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_regulator(regulator_config: str, seed: Optional[int] = None) -> Regulator:
    """
    Create a regulator with the specified configuration.

    Args:
        regulator_config: Regulator configuration ('ml', 'rule_based', 'none')
        seed: Random seed for reproducibility

    Returns:
        Regulator instance
    """
    regulator_config = regulator_config.lower()

    if regulator_config == "ml":
        # ML-based regulator with stricter thresholds
        return Regulator(
            parallel_threshold=1.5,
            parallel_steps=2,
            structural_break_threshold=8.0,
            fine_amount=75.0,
            seed=seed,
        )
    elif regulator_config == "rule_based":
        # Standard rule-based regulator
        return Regulator(
            parallel_threshold=2.0,
            parallel_steps=3,
            structural_break_threshold=10.0,
            fine_amount=50.0,
            seed=seed,
        )
    elif regulator_config == "none":
        # No regulator (monitoring only, no fines)
        return Regulator(
            parallel_threshold=2.0,
            parallel_steps=3,
            structural_break_threshold=10.0,
            fine_amount=0.0,
            seed=seed,
        )
    elif regulator_config == "disabled":
        # Completely disable regulator
        return None
    else:
        raise ValueError(f"Unknown regulator config: {regulator_config}")


def calculate_welfare_metrics(
    episode_data: Dict[str, Any], env: CartelEnv
) -> Dict[str, float]:
    """
    Calculate welfare metrics from episode data.

    Args:
        episode_data: Episode data containing prices and profits
        env: Environment instance for parameters

    Returns:
        Dictionary containing welfare metrics
    """
    episode_prices = episode_data.get("episode_prices", [])
    episode_profits = episode_data.get("episode_profits", [])

    if not episode_prices or not episode_profits:
        return {
            "consumer_surplus": 0.0,
            "producer_surplus": 0.0,
            "total_welfare": 0.0,
            "deadweight_loss": 0.0,
        }

    # Convert to numpy arrays
    prices_array = np.array(episode_prices)
    profits_array = np.array(episode_profits)

    # Check if arrays are empty or have unexpected shapes
    if prices_array.size == 0 or profits_array.size == 0:
        return {
            "consumer_surplus": 0.0,
            "producer_surplus": 0.0,
            "total_welfare": 0.0,
            "deadweight_loss": 0.0,
        }

    # Calculate market prices and quantities
    market_prices = np.mean(prices_array, axis=1)
    quantities = np.maximum(
        0.0,
        (env.demand_intercept + env.demand_slope * market_prices) / env.n_firms,
    )

    # Consumer surplus: area under demand curve above market price
    # For linear demand D = a + b*p, CS = 0.5 * (a - p) * q
    consumer_surplus = 0.5 * (env.demand_intercept - market_prices) * quantities
    total_consumer_surplus: float = np.sum(consumer_surplus)

    # Producer surplus: total profits
    total_producer_surplus: float = np.sum(profits_array)

    # Total welfare
    total_welfare = total_consumer_surplus + total_producer_surplus

    # Deadweight loss: difference from competitive equilibrium
    # Competitive price = marginal cost
    competitive_price = env.marginal_cost
    competitive_quantity = max(
        0.0, (env.demand_intercept + env.demand_slope * competitive_price) / env.n_firms
    )
    competitive_welfare = (
        0.5 * (env.demand_intercept - competitive_price) * competitive_quantity
        + (competitive_price - env.marginal_cost) * competitive_quantity
    ) * env.n_firms

    deadweight_loss = max(0.0, competitive_welfare - total_welfare)

    return {
        "consumer_surplus": float(total_consumer_surplus),
        "producer_surplus": float(total_producer_surplus),
        "total_welfare": float(total_welfare),
        "deadweight_loss": float(deadweight_loss),
    }


def print_experiment_summary(
    results: Dict[str, Any],
    episode_data: Dict[str, Any],
    welfare_metrics: Dict[str, float],
) -> None:
    """
    Print a summary of the experiment results.

    Args:
        results: Results from episode runner
        episode_data: Episode data
        welfare_metrics: Calculated welfare metrics
    """
    episode_summary = results.get("episode_summary", {})

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    # Basic episode info
    print(f"Episode ID: {results.get('episode_id', 'N/A')}")
    print(f"Total Steps: {episode_data.get('total_steps', 0)}")
    print(f"Agent Types: {', '.join(episode_summary.get('agent_types', []))}")
    print(f"Number of Firms: {len(episode_summary.get('agent_types', []))}")

    # Price statistics
    avg_prices = episode_summary.get("avg_prices", [])
    if avg_prices and len(avg_prices) > 0:
        print(f"\nAverage Prices: {[f'{p:.2f}' for p in avg_prices]}")
        print(f"Overall Average Price: {np.mean(avg_prices):.2f}")
        print(f"Price Standard Deviation: {np.std(avg_prices):.2f}")

    # Profit statistics
    total_profits = episode_summary.get("total_profits", [])
    if total_profits and len(total_profits) > 0:
        print(f"\nTotal Profits: {[f'{p:.2f}' for p in total_profits]}")
        print(f"Total Industry Profits: {sum(total_profits):.2f}")
        print(f"Average Profit per Firm: {np.mean(total_profits):.2f}")

    # Welfare metrics
    print("\nWELFARE METRICS:")
    print(f"  Consumer Surplus: {welfare_metrics['consumer_surplus']:.2f}")
    print(f"  Producer Surplus: {welfare_metrics['producer_surplus']:.2f}")
    print(f"  Total Welfare: {welfare_metrics['total_welfare']:.2f}")
    print(f"  Deadweight Loss: {welfare_metrics['deadweight_loss']:.2f}")

    # Regulator results
    total_fines = episode_data.get("total_fines", 0.0)
    violations = episode_data.get("violations", {})

    print("\nREGULATOR RESULTS:")
    print(f"  Total Fines Applied: {total_fines:.2f}")
    print(f"  Parallel Pricing Violations: {violations.get('parallel', 0)}")
    print(f"  Structural Break Violations: {violations.get('structural_break', 0)}")

    # Log file info
    print(f"\nLog File: {results.get('log_file', 'N/A')}")
    print("=" * 80)


def run_experiment(
    firms: List[str],
    steps: int = 100,
    regulator_config: str = "rule_based",
    seed: int = 42,
    log_dir: str = "logs",
    episode_id: Optional[str] = None,
    env_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a complete experiment with the specified parameters.

    Args:
        firms: List of agent types for each firm
        steps: Number of steps to run
        regulator_config: Regulator configuration ('ml', 'rule_based', 'none')
        seed: Random seed for reproducibility
        log_dir: Directory to save log files
        episode_id: Unique identifier for this episode
        env_params: Additional environment parameters

    Returns:
        Dictionary containing experiment results
    """
    n_firms = len(firms)

    # Default environment parameters
    default_env_params = {
        "n_firms": n_firms,
        "max_steps": steps,
        "marginal_cost": 10.0,
        "demand_intercept": 100.0,
        "demand_slope": -1.0,
        "shock_std": 5.0,
        "price_min": 1.0,
        "price_max": 100.0,
        "seed": seed,
    }

    if env_params:
        default_env_params.update(env_params)

    # Create environment
    env = CartelEnv(
        n_firms=int(default_env_params["n_firms"]),
        max_steps=int(default_env_params["max_steps"]),
        marginal_cost=float(default_env_params["marginal_cost"]),
        demand_intercept=float(default_env_params["demand_intercept"]),
        demand_slope=float(default_env_params["demand_slope"]),
        shock_std=float(default_env_params["shock_std"]),
        price_min=float(default_env_params["price_min"]),
        price_max=float(default_env_params["price_max"]),
        seed=(
            int(default_env_params["seed"])
            if default_env_params["seed"] is not None
            else None
        ),
    )

    # Create agents
    agents = []
    for i, agent_type in enumerate(firms):
        agent = create_agent(agent_type, agent_id=i, seed=seed + i)
        agents.append(agent)

    # Create regulator (if enabled)
    regulator = create_regulator(regulator_config, seed=seed)

    # Generate episode ID if not provided
    if episode_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_id = f"experiment_{timestamp}"

    print(f"Running experiment: {episode_id}")
    print(f"Firms: {', '.join(firms)}")
    print(f"Steps: {steps}")
    print(f"Regulator: {regulator_config}")
    print(f"Seed: {seed}")
    print("-" * 60)

    # Run episode (with or without regulator)
    if regulator is not None:
        results = run_episode_with_regulator_logging(
            env=env,
            agents=agents,
            regulator=regulator,
            log_dir=log_dir,
            episode_id=episode_id,
            agent_types=firms,
        )
    else:
        # Run basic episode without regulator (like run_episode.py)
        from episode_logging.logger import Logger

        logger = Logger(
            log_dir=log_dir,
            episode_id=episode_id,
            n_firms=len(agents),
        )

        # Reset environment and agents
        obs, info = env.reset(seed=seed)
        for agent in agents:
            agent.reset()

        step = 0
        while step < default_env_params["max_steps"]:
            # Each agent chooses a price
            prices = []
            for i, agent in enumerate(agents):
                price = agent.choose_price(obs, env, info)
                prices.append(price)

            action = np.array(prices, dtype=np.float32)

            # Take step in environment
            next_obs, rewards, terminated, truncated, step_info = env.step(action)

            # Update agent histories
            for i, agent in enumerate(agents):
                rival_prices = np.array(
                    [prices[j] for j in range(len(prices)) if j != i]
                )
                agent.update_history(prices[i], rival_prices)

            # Log step data
            logger.log_step(
                step=step + 1,
                prices=step_info["prices"],
                profits=rewards,
                demand_shock=step_info["demand_shock"],
                market_price=step_info["market_price"],
                total_demand=step_info["total_demand"],
                individual_quantity=step_info["individual_quantities"],
                total_profits=step_info["total_profits"],
                additional_info={
                    "agent_types": firms,
                    "agent_prices": prices,
                },
            )

            # Update for next iteration
            obs = next_obs
            info = step_info
            step += 1

            # Check termination
            if terminated or truncated:
                break

        # Log episode end
        episode_summary = {
            "total_steps": step,
            "final_market_price": float(step_info["market_price"]),
            "total_profits": [float(p) for p in step_info["total_profits"].tolist()],
            "agent_types": firms,
            "environment_params": default_env_params,
        }

        logger.log_episode_end(
            terminated=terminated,
            truncated=truncated,
            final_rewards=step_info["total_profits"],
            episode_summary=episode_summary,
        )

        # Create results structure compatible with regulator version
        results = {
            "log_file": str(logger.get_log_file_path()),
            "episode_data": {
                "total_steps": step,
                "violations": {"parallel": 0, "structural_break": 0},
                "total_fines": 0.0,
                "episode_prices": [step_info["prices"] for _ in range(step)],
                "episode_profits": [step_info["total_profits"] for _ in range(step)],
            },
            "episode_summary": episode_summary,
        }

    # Add experiment metadata
    results["episode_id"] = episode_id
    results["experiment_params"] = {
        "firms": firms,
        "steps": steps,
        "regulator_config": regulator_config,
        "seed": seed,
        "env_params": default_env_params,
    }

    # Ensure all numpy types are converted to Python native types for JSON serialization
    def convert_numpy_types(obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    # Convert numpy types in results
    results = convert_numpy_types(results)  # type: ignore

    # Calculate welfare metrics
    welfare_metrics = calculate_welfare_metrics(results["episode_data"], env)
    results["welfare_metrics"] = welfare_metrics

    # Print summary
    print_experiment_summary(results, results["episode_data"], welfare_metrics)

    return results  # type: ignore


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Run regulator experiments with various agent types and configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_experiment.py --firms random,bestresponse,titfortat --steps 100 --regulator ml
  python scripts/run_experiment.py --firms random,random --steps 50 --regulator rule_based --seed 123
  python scripts/run_experiment.py --firms titfortat,titfortat --steps 200 --regulator none
  python scripts/run_experiment.py --firms random,titfortat --steps 50 --regulator disabled  # Basic episode without regulator
        """,
    )

    parser.add_argument(
        "--firms",
        type=str,
        required=True,
        help="Comma-separated list of agent types (random, bestresponse, titfortat)",
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
        choices=["ml", "rule_based", "none", "disabled"],
        help="Regulator configuration (default: rule_based). Use 'disabled' for basic episodes without regulator.",
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
    valid_types = {"random", "bestresponse", "titfortat"}
    for firm in firms:
        if firm.lower() not in valid_types:
            print(
                f"Error: Unknown agent type '{firm}'. Valid types: {', '.join(valid_types)}"
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
