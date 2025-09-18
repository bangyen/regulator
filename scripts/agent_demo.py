#!/usr/bin/env python3
"""
Demo script showing RandomAgent vs BestResponseAgent in CartelEnv.

This script demonstrates the baseline firm agents by running a single episode
where a RandomAgent competes against a BestResponseAgent, logging their prices
and profits throughout the episode.
"""

import logging
from typing import List

import sys
from pathlib import Path

import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents.firm_agents import BestResponseAgent, RandomAgent
from src.cartel.cartel_env import CartelEnv


def setup_logging() -> None:
    """Set up logging configuration for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def run_agent_demo() -> None:
    """
    Run a demo episode with RandomAgent vs BestResponseAgent.

    This function creates a CartelEnv with 2 firms, initializes the agents,
    and runs a single episode while logging prices and profits.
    """
    logger = logging.getLogger(__name__)

    # Set up environment
    env = CartelEnv(
        n_firms=2,
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
    random_agent = RandomAgent(agent_id=0, seed=123)
    best_response_agent = BestResponseAgent(agent_id=1, seed=456)
    agents = [random_agent, best_response_agent]

    logger.info("Starting agent demo: RandomAgent vs BestResponseAgent")
    logger.info(f"Environment: {env.n_firms} firms, {env.max_steps} max steps")
    logger.info(
        f"Parameters: MC={env.marginal_cost}, Demand={env.demand_intercept}+{env.demand_slope}*p, Shock_std={env.shock_std}"
    )
    logger.info(f"Price bounds: [{env.price_min}, {env.price_max}]")
    logger.info("-" * 80)

    # Reset environment
    obs, info = env.reset(seed=42)
    logger.info(f"Initial observation: prices={obs[:2]}, demand_shock={obs[2]:.2f}")

    # Track episode statistics
    episode_prices: List[List[float]] = []
    episode_profits: List[List[float]] = []
    episode_demand_shocks: List[float] = []

    # Run episode
    step = 0
    while step < env.max_steps:
        # Each agent chooses a price
        prices = []
        for i, agent in enumerate(agents):
            price = agent.choose_price(obs, env, info)
            prices.append(price)

        # Log prices for this step
        logger.info(
            f"Step {step + 1:2d}: RandomAgent price={prices[0]:6.2f}, BestResponseAgent price={prices[1]:6.2f}"
        )

        # Take environment step
        action = np.array(prices, dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Log results
        market_price = np.mean(prices)
        logger.info(
            f"         Market price={market_price:6.2f}, Demand shock={info['demand_shock']:6.2f}"
        )
        logger.info(
            f"         Profits: RandomAgent={rewards[0]:8.2f}, BestResponseAgent={rewards[1]:8.2f}"
        )
        logger.info(
            f"         Total demand={info['total_demand']:6.2f}, Individual quantity={info['individual_quantity']:6.2f}"
        )

        # Update agent histories
        for i, agent in enumerate(agents):
            rival_prices = np.array([prices[j] for j in range(len(prices)) if j != i])
            agent.update_history(prices[i], rival_prices)

        # Store episode data
        episode_prices.append(prices.copy())
        episode_profits.append(rewards.copy().tolist())
        episode_demand_shocks.append(info["demand_shock"])

        step += 1

        if terminated or truncated:
            break

    # Episode summary
    logger.info("-" * 80)
    logger.info("EPISODE SUMMARY")
    logger.info("-" * 80)

    # Calculate total profits
    total_profits = np.sum(episode_profits, axis=0)
    logger.info(
        f"Total profits: RandomAgent={total_profits[0]:8.2f}, BestResponseAgent={total_profits[1]:8.2f}"
    )

    # Calculate average prices
    avg_prices = np.mean(episode_prices, axis=0)
    logger.info(
        f"Average prices: RandomAgent={avg_prices[0]:6.2f}, BestResponseAgent={avg_prices[1]:6.2f}"
    )

    # Calculate price statistics
    random_prices = [p[0] for p in episode_prices]
    best_response_prices = [p[1] for p in episode_prices]

    logger.info(
        f"RandomAgent price range: [{min(random_prices):6.2f}, {max(random_prices):6.2f}]"
    )
    logger.info(
        f"BestResponseAgent price range: [{min(best_response_prices):6.2f}, {max(best_response_prices):6.2f}]"
    )

    # Calculate demand shock statistics
    avg_demand_shock = np.mean(episode_demand_shocks)
    logger.info(f"Average demand shock: {avg_demand_shock:6.2f}")

    # Analyze agent behavior
    logger.info("-" * 80)
    logger.info("AGENT BEHAVIOR ANALYSIS")
    logger.info("-" * 80)

    # RandomAgent analysis
    random_std = np.std(random_prices)
    logger.info(f"RandomAgent: Mean={avg_prices[0]:6.2f}, Std={random_std:6.2f}")

    # BestResponseAgent analysis
    best_response_std = np.std(best_response_prices)
    logger.info(
        f"BestResponseAgent: Mean={avg_prices[1]:6.2f}, Std={best_response_std:6.2f}"
    )

    # Check if BestResponseAgent is more consistent (lower std)
    if best_response_std < random_std:
        logger.info(
            "BestResponseAgent shows more consistent pricing (lower standard deviation)"
        )
    else:
        logger.info(
            "RandomAgent shows more consistent pricing (lower standard deviation)"
        )

    # Check profit comparison
    if total_profits[1] > total_profits[0]:
        logger.info("BestResponseAgent achieved higher total profits")
    elif total_profits[0] > total_profits[1]:
        logger.info("RandomAgent achieved higher total profits")
    else:
        logger.info("Both agents achieved equal total profits")

    logger.info("-" * 80)
    logger.info("Demo completed successfully!")


def main() -> None:
    """Main function to run the agent demo."""
    setup_logging()
    run_agent_demo()


if __name__ == "__main__":
    main()
