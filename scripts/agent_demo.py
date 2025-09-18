#!/usr/bin/env python3
"""
Demo script showing RandomAgent vs BestResponseAgent in CartelEnv.

This script runs one episode with RandomAgent competing against BestResponseAgent
and logs the prices, profits, and market outcomes for analysis.
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.firm_agents import BestResponseAgent, RandomAgent  # noqa: E402
from src.cartel.cartel_env import CartelEnv  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("agent_demo.log")],
)
logger = logging.getLogger(__name__)


def run_agent_demo() -> None:
    """Run a demo episode with RandomAgent vs BestResponseAgent."""
    logger.info("Starting agent demo: RandomAgent vs BestResponseAgent")

    # Create environment
    env = CartelEnv(
        n_firms=2,
        max_steps=20,
        marginal_cost=10.0,
        demand_intercept=100.0,
        demand_slope=-1.0,
        shock_std=2.0,  # Small demand shocks for clearer results
        price_min=1.0,
        price_max=100.0,
        seed=42,
    )

    # Create agents
    random_agent = RandomAgent(agent_id=0, seed=42)
    best_response_agent = BestResponseAgent(agent_id=1, seed=43)

    logger.info(f"Environment setup: {env.n_firms} firms, {env.max_steps} steps")
    logger.info(f"Demand: D = {env.demand_intercept} + {env.demand_slope} * p_market")
    logger.info(f"Marginal cost: {env.marginal_cost}")
    logger.info(f"Price bounds: [{env.price_min}, {env.price_max}]")

    # Reset environment
    obs, info = env.reset(seed=42)
    logger.info(f"Initial observation: {obs}")
    logger.info(f"Initial demand shock: {info['demand_shock']:.2f}")

    # Track results
    step_results = []

    # Run episode
    for step in range(env.max_steps):
        logger.info(f"\n--- Step {step + 1} ---")

        # Get prices from agents
        random_price = random_agent.choose_price(obs, env, info)
        best_response_price = best_response_agent.choose_price(obs, env, info)

        prices = np.array([random_price, best_response_price], dtype=np.float32)

        logger.info(f"RandomAgent price: {random_price:.2f}")
        logger.info(f"BestResponseAgent price: {best_response_price:.2f}")
        logger.info(f"Market average price: {np.mean(prices):.2f}")

        # Take environment step
        next_obs, rewards, terminated, truncated, next_info = env.step(prices)

        # Update agent histories
        random_agent.update_history(random_price, np.array([best_response_price]))
        best_response_agent.update_history(
            best_response_price, np.array([random_price])
        )

        # Log results
        logger.info(f"RandomAgent profit: {rewards[0]:.2f}")
        logger.info(f"BestResponseAgent profit: {rewards[1]:.2f}")
        logger.info(f"Total demand: {next_info['total_demand']:.2f}")
        logger.info(f"Individual quantity: {next_info['individual_quantity']:.2f}")
        logger.info(f"Demand shock: {next_info['demand_shock']:.2f}")

        # Store step results
        step_results.append(
            {
                "step": step + 1,
                "random_price": random_price,
                "best_response_price": best_response_price,
                "market_price": np.mean(prices),
                "random_profit": rewards[0],
                "best_response_profit": rewards[1],
                "total_demand": next_info["total_demand"],
                "demand_shock": next_info["demand_shock"],
            }
        )

        # Update observation for next step
        obs = next_obs
        info = next_info

        # Check termination
        if terminated or truncated:
            logger.info(
                f"Episode ended: terminated={terminated}, truncated={truncated}"
            )
            break

    # Summary statistics
    logger.info("\n" + "=" * 50)
    logger.info("EPISODE SUMMARY")
    logger.info("=" * 50)

    random_prices = [r["random_price"] for r in step_results]
    best_response_prices = [r["best_response_price"] for r in step_results]
    random_profits = [r["random_profit"] for r in step_results]
    best_response_profits = [r["best_response_profit"] for r in step_results]

    logger.info(f"Total steps: {len(step_results)}")
    logger.info(
        f"RandomAgent - Avg price: {np.mean(random_prices):.2f}, Total profit: {sum(random_profits):.2f}"
    )
    logger.info(
        f"BestResponseAgent - Avg price: {np.mean(best_response_prices):.2f}, Total profit: {sum(best_response_profits):.2f}"
    )

    # Price statistics
    logger.info(
        f"RandomAgent price range: [{min(random_prices):.2f}, {max(random_prices):.2f}]"
    )
    logger.info(
        f"BestResponseAgent price range: [{min(best_response_prices):.2f}, {max(best_response_prices):.2f}]"
    )

    # Profit comparison
    total_random_profit = sum(random_profits)
    total_best_response_profit = sum(best_response_profits)
    profit_difference = total_best_response_profit - total_random_profit

    logger.info(f"Profit difference (BestResponse - Random): {profit_difference:.2f}")
    logger.info(
        f"BestResponseAgent outperformed by: {profit_difference/total_random_profit*100:.1f}%"
    )

    # Market efficiency
    avg_market_prices = [r["market_price"] for r in step_results]
    logger.info(f"Average market price: {np.mean(avg_market_prices):.2f}")
    logger.info(f"Market price volatility: {np.std(avg_market_prices):.2f}")

    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    run_agent_demo()
