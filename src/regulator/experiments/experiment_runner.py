"""
Experiment execution functions for the Regulator package.

This module contains the core functions for running experiments,
training models, and executing episodes.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np

from regulator.agents.chat_firm import CollusiveChatAgent, CompetitiveChatAgent
from regulator.agents.enhanced_regulator import EnhancedRegulator
from regulator.agents.firm_agents import (
    BaseAgent,
    BestResponseAgent,
    RandomAgent,
    TitForTatAgent,
)
from regulator.agents.ml_regulator import MLRegulator
from regulator.agents.regulator import Regulator
from regulator.agents.stealth_agent import StealthCollusiveAgent
from regulator.cartel.cartel_env import CartelEnv
from regulator.detectors.llm_detector import ChatRegulator, LLMDetector
from regulator.economic_validation import EconomicValidator
from regulator.episode_logging.episode_runner import (
    run_episode_with_regulator_logging,
)
from regulator.episode_logging.logger import Logger
from regulator.experiments.ml_training import train_collusion_classifier

logger = logging.getLogger(__name__)

# Agent types accepted by create_agent (after normalization)
AGENT_TYPES = (
    "random",
    "bestresponse",
    "titfortat",
    "stealth",
    "chatcolluder",
    "chatcompetitor",
)

# Regulator configurations accepted by create_regulator
REGULATOR_CONFIGS = ("rule_based", "ml", "enhanced", "none")


def create_agent(agent_type: str, agent_id: int, seed: int | None = None) -> BaseAgent:
    """
    Create an agent of the specified type.

    Args:
        agent_type: Type of agent to create (one of AGENT_TYPES)
        agent_id: Unique identifier for the agent
        seed: Random seed for reproducibility

    Returns:
        Agent instance
    """
    # Normalize so "tit_for_tat", "tit-for-tat" and "titfortat" are all accepted
    # (the CLI's --firms default uses the underscored spelling).
    agent_type = agent_type.lower().replace("_", "").replace("-", "")

    if agent_type == "random":
        return RandomAgent(agent_id=agent_id, seed=seed)
    elif agent_type == "bestresponse":
        return BestResponseAgent(agent_id=agent_id, seed=seed)
    elif agent_type == "titfortat":
        return TitForTatAgent(agent_id=agent_id, seed=seed)
    elif agent_type == "stealth":
        return StealthCollusiveAgent(agent_id=agent_id, seed=seed)
    elif agent_type == "chatcolluder":
        return CollusiveChatAgent(agent_id=agent_id, seed=seed)
    elif agent_type == "chatcompetitor":
        return CompetitiveChatAgent(agent_id=agent_id, seed=seed)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_regulator(
    regulator_config: str, seed: int | None = None
) -> Regulator | None:
    """
    Create a regulator with the specified configuration.

    Args:
        regulator_config: One of REGULATOR_CONFIGS:
            'rule_based' - parallel-pricing and structural-break rules
            'ml' - rule-based detection plus ML anomaly/collusion models
            'enhanced' - graduated penalties and market-aware monitoring
            'none' - no regulator ('disabled' is accepted as an alias)
        seed: Random seed for reproducibility

    Returns:
        Regulator instance, or None when regulation is off
    """
    config = regulator_config.lower()

    if config == "rule_based":
        return Regulator(seed=seed)
    if config == "ml":
        # Classifier is trained once per process on simulated labeled windows
        return MLRegulator(seed=seed, collusion_classifier=train_collusion_classifier())
    if config == "enhanced":
        return EnhancedRegulator(seed=seed)
    if config in ("none", "disabled"):
        return None
    raise ValueError(
        f"Unknown regulator config: {regulator_config}. "
        f"Valid configs: {', '.join(REGULATOR_CONFIGS)}"
    )


def calculate_welfare_metrics(
    episode_data: dict[str, Any], env: CartelEnv
) -> dict[str, float]:
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
    results: dict[str, Any],
    episode_data: dict[str, Any],
    welfare_metrics: dict[str, float],
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
    if episode_data.get("messages_sent"):
        print(f"  Messages Sent: {episode_data['messages_sent']}")
        print(f"  Collusive Messages Fined: {episode_data['message_violations']}")
        print(f"  Chat Fines: {episode_data['chat_fines']:.2f}")

    # Economic consistency checks
    validation = results.get("economic_validation")
    if validation is not None:
        if validation["valid"]:
            print("\nECONOMIC VALIDATION: passed")
        else:
            print(f"\nECONOMIC VALIDATION: {validation['n_issues']} issue(s)")
            for issue in validation["issues"][:3]:
                print(f"  - {issue}")

    # Log file info
    print(f"\nLog File: {results.get('log_file', 'N/A')}")
    print("=" * 80)


def validate_episode(log_file: str, env: CartelEnv) -> dict[str, Any]:
    """
    Run EconomicValidator over a logged episode.

    Args:
        log_file: Path to the episode's JSONL log
        env: Environment the episode ran in (supplies demand/cost parameters)

    Returns:
        {"valid": bool, "n_issues": int, "issues": first 20 issue strings}
    """
    steps = Logger.load_episode_data(log_file)["steps"]
    validator = EconomicValidator(
        demand_intercept=env.demand_intercept,
        demand_slope=env.demand_slope,
        marginal_cost=env.marginal_cost,
        price_min=env.price_min,
        price_max=env.price_max,
    )
    is_valid, issues = validator.validate_episode_consistency({"steps": steps})
    return {"valid": is_valid, "n_issues": len(issues), "issues": issues[:20]}


def run_experiment(
    firms: list[str],
    steps: int = 100,
    regulator_config: str = "rule_based",
    seed: int = 42,
    log_dir: str = "logs",
    episode_id: str | None = None,
    env_params: dict[str, Any] | None = None,
    chat_monitoring: bool = False,
    llm_model: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run a complete experiment with the specified parameters.

    Args:
        firms: List of agent types for each firm
        steps: Number of steps to run
        regulator_config: One of REGULATOR_CONFIGS
        seed: Random seed for reproducibility
        log_dir: Directory to save log files
        episode_id: Unique identifier for this episode
        env_params: Additional environment parameters
        chat_monitoring: Classify chat messages (from chatcolluder /
            chatcompetitor firms) and fine senders of collusive ones
        llm_model: OpenAI model for chat monitoring; the keyword stub if None
        verbose: Print the experiment summary

    Returns:
        Dictionary containing experiment results
    """
    n_firms = len(firms)

    # Default environment parameters; env_params overrides any of them
    default_env_params: dict[str, Any] = {
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

    env = CartelEnv(**default_env_params)

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

    logger.info("Running experiment: %s", episode_id)
    logger.info("Firms: %s", ", ".join(firms))
    logger.info("Steps: %d", steps)
    logger.info("Regulator: %s", regulator_config)
    logger.info("Seed: %s", seed)

    # Run episode with regulator
    chat_regulator = None
    if chat_monitoring:
        detector = (
            LLMDetector(model_type="llm", model_name=llm_model, seed=seed)
            if llm_model
            else LLMDetector(model_type="stubbed", seed=seed)
        )
        chat_regulator = ChatRegulator(llm_detector=detector)

    results = run_episode_with_regulator_logging(
        env=env,
        agents=agents,
        regulator=regulator,
        log_dir=log_dir,
        episode_id=episode_id,
        agent_types=firms,
        chat_regulator=chat_regulator,
    )

    # Add experiment metadata
    results["episode_id"] = episode_id
    results["experiment_params"] = {
        "firms": firms,
        "steps": steps,
        "regulator_config": regulator_config,
        "seed": seed,
        "env_params": default_env_params,
        "chat_monitoring": chat_monitoring,
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

    # Check the logged episode for economic consistency
    results["economic_validation"] = validate_episode(results["log_file"], env)
    validation = results["economic_validation"]
    if not validation["valid"]:
        logger.warning(
            "Economic validation found %d issue(s); first: %s",
            validation["n_issues"],
            validation["issues"][0],
        )

    if verbose:
        print_experiment_summary(results, results["episode_data"], welfare_metrics)

    return results  # type: ignore
