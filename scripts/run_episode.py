#!/usr/bin/env python3
"""
Run a single episode with structured data logging.

This script runs a CartelEnv episode with specified agents and logs all
step-by-step data to a JSONL file for analysis and reproducibility.
"""

import argparse
import sys
from typing import List, Optional

import numpy as np

# Import from the package
from agents.firm_agents import (
    BaseAgent,
    RandomAgent,
    BestResponseAgent,
    TitForTatAgent,
)
from cartel.cartel_env import CartelEnv
from episode_logging.logger import Logger


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


def run_episode(
    firms: List[str],
    steps: int = 50,
    n_firms: Optional[int] = None,
    seed: int = 42,
    log_dir: str = "logs",
    episode_id: Optional[str] = None,
    env_params: Optional[dict] = None,
) -> str:
    """
    Run a single episode with structured data logging.

    Args:
        firms: List of agent types for each firm
        steps: Number of steps to run
        n_firms: Number of firms (inferred from firms list if None)
        seed: Random seed for reproducibility
        log_dir: Directory to save log files
        episode_id: Unique identifier for this episode
        env_params: Additional environment parameters

    Returns:
        Path to the created log file
    """
    # Determine number of firms
    if n_firms is None:
        n_firms = len(firms)
    elif n_firms != len(firms):
        raise ValueError(
            f"Number of firms ({n_firms}) must match number of agent types ({len(firms)})"
        )

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
        # Convert string values to appropriate types
        for key, value in env_params.items():
            if key in ["n_firms", "max_steps"]:
                default_env_params[key] = int(str(value))
            elif key in [
                "marginal_cost",
                "demand_intercept",
                "demand_slope",
                "shock_std",
                "price_min",
                "price_max",
            ]:
                default_env_params[key] = float(str(value))
            elif key == "seed":
                if value is not None:
                    default_env_params[key] = int(str(value))
                else:
                    default_env_params[key] = None  # type: ignore
            else:
                default_env_params[key] = value

    # Create environment with explicit type conversion
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

    # Create logger
    logger = Logger(
        log_dir=log_dir,
        episode_id=episode_id,
        n_firms=n_firms,
    )

    print(f"Running episode with {n_firms} firms: {', '.join(firms)}")
    print(f"Steps: {steps}, Seed: {seed}")
    print(f"Log file: {logger.get_log_file_path()}")
    print("-" * 60)

    # Reset environment and agents
    obs, info = env.reset(seed=seed)
    for agent in agents:
        agent.reset()

    # Run episode
    step = 0
    while step < steps:
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
            rival_prices = np.array([prices[j] for j in range(len(prices)) if j != i])
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

        # Print step summary
        print(
            f"Step {step + 1:3d}: Prices={[f'{p:.1f}' for p in prices]}, "
            f"Profits={[f'{r:.1f}' for r in rewards]}, "
            f"Market={step_info['market_price']:.1f}, "
            f"Shock={step_info['demand_shock']:.1f}"
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

    print("-" * 60)
    print(f"Episode completed: {step} steps")
    print(f"Final total profits: {[f'{p:.1f}' for p in step_info['total_profits']]}")
    print(f"Log file: {logger.get_log_file_path()}")

    return str(logger.get_log_file_path())


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run a CartelEnv episode with structured data logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_episode.py --firms random,titfortat --steps 50
  python scripts/run_episode.py --firms random,bestresponse,titfortat --steps 100 --seed 123
  python scripts/run_episode.py --firms random,random --steps 25 --log-dir custom_logs
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
        default=50,
        help="Number of steps to run (default: 50)",
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
        # Run episode
        log_file = run_episode(
            firms=firms,
            steps=args.steps,
            seed=args.seed,
            log_dir=args.log_dir,
            episode_id=args.episode_id,
            env_params=env_params,
        )

        print("\nEpisode completed successfully!")
        print(f"Log file: {log_file}")

    except Exception as e:
        print(f"Error running episode: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
