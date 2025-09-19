"""
Experiment execution functions for the Regulator package.

This module contains the core functions for running experiments,
training models, and executing episodes.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import numpy as np

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
    if regulator_config == "none":
        # Create a dummy regulator that doesn't do anything
        return Regulator(seed=seed)
    else:
        return Regulator(seed=seed)


def calculate_welfare_metrics(
    episode_data: List[Dict[str, Any]], env: CartelEnv
) -> Dict[str, Any]:
    """
    Calculate welfare metrics from episode data.

    Args:
        episode_data: List of step data from the episode
        env: The environment used for the episode

    Returns:
        Dictionary containing welfare metrics
    """
    if not episode_data:
        return {
            "consumer_surplus": 0.0,
            "producer_surplus": 0.0,
            "total_welfare": 0.0,
            "deadweight_loss": 0.0,
        }

    # Calculate total consumer surplus and producer surplus
    total_consumer_surplus = 0.0
    total_producer_surplus = 0.0

    for step_data in episode_data:
        market_price = step_data.get("market_price", 0.0)
        total_quantity = step_data.get("total_quantity", 0.0)

        # Consumer surplus = 0.5 * (demand_intercept - market_price) * total_quantity
        consumer_surplus = 0.5 * (env.demand_intercept - market_price) * total_quantity
        total_consumer_surplus += max(0, consumer_surplus)

        # Producer surplus = total profits
        total_profits = sum(step_data.get("profits", [0.0]))
        total_producer_surplus += total_profits

    total_welfare = total_consumer_surplus + total_producer_surplus

    # Calculate deadweight loss (simplified)
    # In a competitive market, welfare would be higher
    competitive_welfare = (
        total_welfare * 1.1
    )  # Assume 10% higher welfare in competitive market
    deadweight_loss = max(0, competitive_welfare - total_welfare)

    return {
        "consumer_surplus": total_consumer_surplus,
        "producer_surplus": total_producer_surplus,
        "total_welfare": total_welfare,
        "deadweight_loss": deadweight_loss,
    }


def print_experiment_summary(
    results: Dict[str, Any],
    episode_data: List[Dict[str, Any]],
    welfare_metrics: Dict[str, Any],
) -> None:
    """Print a summary of the experiment results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Episode ID: {results.get('episode_id', 'unknown')}")
    print(f"Total Steps: {len(episode_data)}")
    print(
        f"Agent Types: {', '.join(results.get('experiment_params', {}).get('firms', []))}"
    )
    print(
        f"Number of Firms: {len(results.get('experiment_params', {}).get('firms', []))}"
    )
    print()

    if episode_data:
        # Calculate average prices
        avg_prices = []
        for step_data in episode_data:
            prices = step_data.get("prices", [])
            if prices:
                avg_prices.append(np.mean(prices))

        if avg_prices:
            print(f"Average Prices: {[f'{p:.2f}' for p in avg_prices]}")
            print(f"Overall Average Price: {np.mean(avg_prices):.2f}")
            print(f"Price Standard Deviation: {np.std(avg_prices):.2f}")
            print()

        # Calculate total profits
        total_profits = []
        for step_data in episode_data:
            profits = step_data.get("profits", [])
            if profits:
                total_profits.append(sum(profits))

        if total_profits:
            print(f"Total Profits: {[f'{p:.2f}' for p in total_profits]}")
            print(f"Total Industry Profits: {sum(total_profits):.2f}")
            print(
                f"Average Profit per Firm: {sum(total_profits) / len(results.get('experiment_params', {}).get('firms', [1])):.2f}"
            )
            print()

    # Print welfare metrics
    print("WELFARE METRICS:")
    print(f"  Consumer Surplus: {welfare_metrics.get('consumer_surplus', 0):.2f}")
    print(f"  Producer Surplus: {welfare_metrics.get('producer_surplus', 0):.2f}")
    print(f"  Total Welfare: {welfare_metrics.get('total_welfare', 0):.2f}")
    print(f"  Deadweight Loss: {welfare_metrics.get('deadweight_loss', 0):.2f}")
    print()

    # Print regulator results
    if "regulator_results" in results:
        regulator_results = results["regulator_results"]
        print("REGULATOR RESULTS:")
        print(f"  Total Fines Applied: {regulator_results.get('total_fines', 0):.2f}")
        print(
            f"  Parallel Pricing Violations: {regulator_results.get('parallel_pricing_violations', 0)}"
        )
        print(
            f"  Structural Break Violations: {regulator_results.get('structural_break_violations', 0)}"
        )
        print()

    print(f"Log File: {results.get('log_file', 'unknown')}")
    print("=" * 80)
    print()
    print("Experiment completed successfully!")
    print(f"Results saved to: {results.get('log_file', 'unknown')}")


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

    # Default environment parameters with enhanced economic features
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
        # Enhanced economic features enabled by default
        "use_logit_market_shares": True,  # More realistic market shares
        "use_enhanced_market_shares": True,  # Multi-factor market share model
        "use_capacity_constraints": True,  # Production capacity limits
        "use_economies_of_scale": True,  # Cost advantages at scale
        "use_dynamic_elasticity": True,  # Varying price sensitivity
        "use_fixed_costs": True,  # Realistic cost structure
        "use_information_asymmetry": True,  # Market frictions
        "use_market_entry_exit": True,  # Dynamic market structure
        "capacity": [80.0, 120.0, 100.0] if n_firms == 3 else [100.0] * n_firms,
        "fixed_cost": 50.0,
        "scale_elasticity": 0.8,
        "elasticity_sensitivity": 0.3,
        "exit_threshold": -50.0,
        "max_consecutive_losses": 3,
    }

    if env_params:
        default_env_params.update(env_params)

    # Create environment with enhanced economic features
    env = CartelEnv(
        n_firms=int(cast(int, default_env_params["n_firms"])),
        max_steps=int(cast(int, default_env_params["max_steps"])),
        marginal_cost=float(cast(float, default_env_params["marginal_cost"])),
        demand_intercept=float(cast(float, default_env_params["demand_intercept"])),
        demand_slope=float(cast(float, default_env_params["demand_slope"])),
        shock_std=float(cast(float, default_env_params["shock_std"])),
        price_min=float(cast(float, default_env_params["price_min"])),
        price_max=float(cast(float, default_env_params["price_max"])),
        seed=(
            int(cast(int, default_env_params["seed"]))
            if default_env_params["seed"] is not None
            else None
        ),
        # Enhanced economic features
        use_logit_market_shares=bool(
            default_env_params.get("use_logit_market_shares", True)
        ),
        use_enhanced_market_shares=bool(
            default_env_params.get("use_enhanced_market_shares", True)
        ),
        use_capacity_constraints=bool(
            default_env_params.get("use_capacity_constraints", True)
        ),
        use_economies_of_scale=bool(
            default_env_params.get("use_economies_of_scale", True)
        ),
        use_dynamic_elasticity=bool(
            default_env_params.get("use_dynamic_elasticity", True)
        ),
        use_fixed_costs=bool(default_env_params.get("use_fixed_costs", True)),
        use_information_asymmetry=bool(
            default_env_params.get("use_information_asymmetry", True)
        ),
        use_market_entry_exit=bool(
            default_env_params.get("use_market_entry_exit", True)
        ),
        capacity=cast(Optional[List[float]], default_env_params.get("capacity", None)),
        fixed_cost=float(cast(float, default_env_params.get("fixed_cost", 50.0))),
        scale_elasticity=float(
            cast(float, default_env_params.get("scale_elasticity", 0.8))
        ),
        elasticity_sensitivity=float(
            cast(float, default_env_params.get("elasticity_sensitivity", 0.3))
        ),
        exit_threshold=float(
            cast(float, default_env_params.get("exit_threshold", -50.0))
        ),
        max_consecutive_losses=int(
            cast(int, default_env_params.get("max_consecutive_losses", 3))
        ),
    )

    # Create agents
    agents = []
    for i, agent_type in enumerate(firms):
        agent = create_agent(agent_type, agent_id=i, seed=seed + i)
        agents.append(agent)

    # Create regulator
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

    # Run episode with regulator
    results = run_episode_with_regulator_logging(
        env=env,
        agents=agents,
        regulator=regulator,
        log_dir=log_dir,
        episode_id=episode_id,
        agent_types=firms,
    )

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
