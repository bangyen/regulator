"""
Episode runner with integrated structured data logging.

This module provides utilities for running CartelEnv episodes with automatic
structured data logging, making it easy to integrate logging into existing
environment loops.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from cartel.cartel_env import CartelEnv
from episode_logging.logger import Logger


def run_episode_with_logging(
    env: CartelEnv,
    agents: List[Any],
    logger: Optional[Logger] = None,
    log_dir: str = "logs",
    episode_id: Optional[str] = None,
    agent_types: Optional[List[str]] = None,
    regulator_flags: Optional[Dict[str, Any]] = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a CartelEnv episode with integrated structured data logging.

    This function provides a drop-in replacement for manual episode loops,
    automatically logging all step data to a JSONL file.

    Args:
        env: CartelEnv environment instance
        agents: List of agent instances
        logger: Optional Logger instance (created if None)
        log_dir: Directory to save log files
        episode_id: Unique identifier for this episode
        agent_types: List of agent type names for logging
        regulator_flags: Optional regulator flags to log each step
        additional_info: Optional additional information to log each step

    Returns:
        Dictionary containing episode results and logger information
    """
    # Create logger if not provided
    if logger is None:
        logger = Logger(
            log_dir=log_dir,
            episode_id=episode_id,
            n_firms=env.n_firms,
        )

    # Reset environment and agents
    obs, info = env.reset()
    for agent in agents:
        if hasattr(agent, "reset"):
            agent.reset()

    # Track episode data
    episode_data: Dict[str, Any] = {
        "total_steps": 0,
        "total_rewards": np.zeros(env.n_firms),
        "episode_prices": [],
        "episode_profits": [],
        "episode_demand_shocks": [],
        "terminated": False,
        "truncated": False,
    }

    # Run episode
    step = 0
    while step < env.max_steps:
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
            if hasattr(agent, "update_history"):
                rival_prices = np.array(
                    [prices[j] for j in range(len(prices)) if j != i]
                )
                agent.update_history(prices[i], rival_prices)

        # Prepare additional info for logging
        step_additional_info = additional_info.copy() if additional_info else {}
        if agent_types:
            step_additional_info["agent_types"] = agent_types
        step_additional_info["agent_prices"] = prices

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
            regulator_flags=regulator_flags,
            additional_info=step_additional_info,
        )

        # Update episode tracking
        episode_data["total_steps"] = step + 1
        episode_data["total_rewards"] += rewards
        prices_list = episode_data["episode_prices"]
        if isinstance(prices_list, list):
            prices_list.append(prices.copy())
        profits_list = episode_data["episode_profits"]
        if isinstance(profits_list, list):
            profits_list.append(rewards.copy().tolist())
        demand_shocks_list = episode_data["episode_demand_shocks"]
        if isinstance(demand_shocks_list, list):
            demand_shocks_list.append(step_info["demand_shock"])
        episode_data["terminated"] = terminated
        episode_data["truncated"] = truncated

        # Update for next iteration
        obs = next_obs
        info = step_info
        step += 1

        # Check termination
        if terminated or truncated:
            break

    # Create episode summary
    episode_prices = episode_data["episode_prices"]
    episode_demand_shocks = episode_data["episode_demand_shocks"]

    # Type hints for MyPy
    prices_array = np.array(episode_prices) if episode_prices else np.array([])
    demand_shocks_array = (
        np.array(episode_demand_shocks) if episode_demand_shocks else np.array([])
    )

    episode_summary = {
        "total_steps": episode_data["total_steps"],
        "final_market_price": step_info["market_price"],
        "total_profits": (
            episode_data["total_rewards"].tolist()
            if isinstance(episode_data["total_rewards"], np.ndarray)
            else []
        ),
        "avg_prices": (
            np.mean(prices_array, axis=0).tolist() if len(prices_array) > 0 else []
        ),
        "price_std": (
            np.std(prices_array, axis=0).tolist() if len(prices_array) > 0 else []
        ),
        "avg_demand_shock": (
            np.mean(demand_shocks_array) if len(demand_shocks_array) > 0 else 0.0
        ),
        "demand_shock_std": (
            np.std(demand_shocks_array) if len(demand_shocks_array) > 0 else 0.0
        ),
        "agent_types": agent_types,
        "environment_params": {
            "n_firms": env.n_firms,
            "max_steps": env.max_steps,
            "marginal_cost": env.marginal_cost,
            "demand_intercept": env.demand_intercept,
            "demand_slope": env.demand_slope,
            "shock_std": env.shock_std,
            "price_min": env.price_min,
            "price_max": env.price_max,
        },
    }

    # Log episode end
    logger.log_episode_end(
        terminated=bool(episode_data["terminated"]),
        truncated=bool(episode_data["truncated"]),
        final_rewards=(
            episode_data["total_rewards"]
            if isinstance(episode_data["total_rewards"], np.ndarray)
            else None
        ),
        episode_summary=episode_summary,
    )

    # Return results
    results = {
        "logger": logger,
        "log_file": str(logger.get_log_file_path()),
        "episode_data": episode_data,
        "episode_summary": episode_summary,
    }

    return results


def run_episode_with_regulator_logging(
    env: CartelEnv,
    agents: List[Any],
    regulator: Any,
    logger: Optional[Logger] = None,
    log_dir: str = "logs",
    episode_id: Optional[str] = None,
    agent_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a CartelEnv episode with regulator monitoring and integrated logging.

    This function extends the basic episode runner to include regulator
    monitoring and penalty application, logging all relevant data.

    Args:
        env: CartelEnv environment instance
        agents: List of agent instances
        regulator: Regulator instance for monitoring
        logger: Optional Logger instance (created if None)
        log_dir: Directory to save log files
        episode_id: Unique identifier for this episode
        agent_types: List of agent type names for logging

    Returns:
        Dictionary containing episode results and logger information
    """
    # Create logger if not provided
    if logger is None:
        logger = Logger(
            log_dir=log_dir,
            episode_id=episode_id,
            n_firms=env.n_firms,
        )

    # Reset environment and agents
    obs, info = env.reset()
    for agent in agents:
        if hasattr(agent, "reset"):
            agent.reset()

    # Track episode data
    episode_data = {
        "total_steps": 0,
        "total_rewards": np.zeros(env.n_firms),
        "total_fines": 0.0,
        "violations": {"parallel": 0, "structural_break": 0},
        "episode_prices": [],
        "episode_profits": [],
        "episode_fines": [],
        "episode_demand_shocks": [],
        "terminated": False,
        "truncated": False,
    }

    # Run episode
    step = 0
    while step < env.max_steps:
        # Each agent chooses a price
        prices = []
        for i, agent in enumerate(agents):
            price = agent.choose_price(obs, env, info)
            prices.append(price)

        action = np.array(prices, dtype=np.float32)

        # Regulator monitors the step
        detection_results = regulator.monitor_step(action, step)

        # Take step in environment
        next_obs, rewards, terminated, truncated, step_info = env.step(action)

        # Apply regulator penalties
        modified_rewards = regulator.apply_penalties(rewards, detection_results)

        # Update agent histories
        for i, agent in enumerate(agents):
            if hasattr(agent, "update_history"):
                rival_prices = np.array(
                    [prices[j] for j in range(len(prices)) if j != i]
                )
                agent.update_history(prices[i], rival_prices)

        # Prepare regulator flags for logging
        regulator_flags = {
            "parallel_violation": detection_results["parallel_violation"],
            "structural_break_violation": detection_results[
                "structural_break_violation"
            ],
            "fines_applied": detection_results["fines_applied"].tolist(),
            "violation_details": detection_results["violation_details"],
        }

        # Prepare additional info for logging
        additional_info = {
            "agent_types": agent_types,
            "agent_prices": prices,
            "original_rewards": rewards.tolist(),
            "modified_rewards": modified_rewards.tolist(),
        }

        # Log step data
        logger.log_step(
            step=step + 1,
            prices=step_info["prices"],
            profits=modified_rewards,  # Log modified rewards (after penalties)
            demand_shock=step_info["demand_shock"],
            market_price=step_info["market_price"],
            total_demand=step_info["total_demand"],
            individual_quantity=step_info["individual_quantities"],
            total_profits=step_info["total_profits"],
            regulator_flags=regulator_flags,
            additional_info=additional_info,
        )

        # Update episode tracking
        episode_data["total_steps"] = step + 1
        episode_data["total_rewards"] += modified_rewards
        episode_data["total_fines"] += np.sum(detection_results["fines_applied"])
        prices_list = episode_data["episode_prices"]
        if isinstance(prices_list, list):
            prices_list.append(prices.copy())
        profits_list = episode_data["episode_profits"]
        if isinstance(profits_list, list):
            profits_list.append(modified_rewards.copy().tolist())
        fines_list = episode_data["episode_fines"]
        if isinstance(fines_list, list):
            fines_list.append(detection_results["fines_applied"].copy().tolist())
        demand_shocks_list = episode_data["episode_demand_shocks"]
        if isinstance(demand_shocks_list, list):
            demand_shocks_list.append(step_info["demand_shock"])
        episode_data["terminated"] = terminated
        episode_data["truncated"] = truncated

        violations_dict = episode_data["violations"]
        if isinstance(violations_dict, dict):
            if detection_results["parallel_violation"]:
                violations_dict["parallel"] += 1
            if detection_results["structural_break_violation"]:
                violations_dict["structural_break"] += 1

        # Update for next iteration
        obs = next_obs
        info = step_info
        step += 1

        # Check termination
        if terminated or truncated:
            break

    # Create episode summary
    episode_prices = episode_data["episode_prices"]
    episode_demand_shocks = episode_data["episode_demand_shocks"]

    # Type hints for MyPy
    prices_array = np.array(episode_prices) if episode_prices else np.array([])
    demand_shocks_array = (
        np.array(episode_demand_shocks) if episode_demand_shocks else np.array([])
    )

    episode_summary = {
        "total_steps": episode_data["total_steps"],
        "final_market_price": step_info["market_price"],
        "total_profits": (
            episode_data["total_rewards"].tolist()
            if isinstance(episode_data["total_rewards"], np.ndarray)
            else []
        ),
        "total_fines": episode_data["total_fines"],
        "violations": episode_data["violations"],
        "avg_prices": (
            np.mean(prices_array, axis=0).tolist() if len(prices_array) > 0 else []
        ),
        "price_std": (
            np.std(prices_array, axis=0).tolist() if len(prices_array) > 0 else []
        ),
        "avg_demand_shock": (
            np.mean(demand_shocks_array) if len(demand_shocks_array) > 0 else 0.0
        ),
        "demand_shock_std": (
            np.std(demand_shocks_array) if len(demand_shocks_array) > 0 else 0.0
        ),
        "agent_types": agent_types,
        "environment_params": {
            "n_firms": env.n_firms,
            "max_steps": env.max_steps,
            "marginal_cost": env.marginal_cost,
            "demand_intercept": env.demand_intercept,
            "demand_slope": env.demand_slope,
            "shock_std": env.shock_std,
            "price_min": env.price_min,
            "price_max": env.price_max,
        },
    }

    # Log episode end
    logger.log_episode_end(
        terminated=bool(episode_data["terminated"]),
        truncated=bool(episode_data["truncated"]),
        final_rewards=(
            episode_data["total_rewards"]
            if isinstance(episode_data["total_rewards"], np.ndarray)
            else None
        ),
        episode_summary=episode_summary,
    )

    # Return results
    results = {
        "logger": logger,
        "log_file": str(logger.get_log_file_path()),
        "episode_data": episode_data,
        "episode_summary": episode_summary,
    }

    return results
