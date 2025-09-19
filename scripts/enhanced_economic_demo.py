#!/usr/bin/env python3
"""
Enhanced Economic Model Demonstration

This script demonstrates the new economic model improvements including:
- Dynamic demand elasticity
- Capacity constraints
- ML-enhanced regulator
- Adaptive learning agents
- Market entry/exit dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cartel.cartel_env import CartelEnv
from agents.ml_regulator import MLRegulator
from agents.adaptive_agent import AdaptiveAgent
from agents.firm_agents import RandomAgent, BestResponseAgent
from episode_logging.episode_logger import EpisodeLogger


def run_enhanced_economic_demo():
    """Run demonstration of enhanced economic model features."""
    print("🚀 Enhanced Economic Model Demonstration")
    print("=" * 50)

    # Create enhanced environment with new features
    env = CartelEnv(
        n_firms=3,
        max_steps=50,
        marginal_cost=10.0,
        demand_intercept=100.0,
        demand_slope=-1.0,
        shock_std=5.0,
        price_min=1.0,
        price_max=100.0,
        # Enhanced economic features
        use_dynamic_elasticity=True,
        elasticity_sensitivity=0.3,
        use_capacity_constraints=True,
        capacity=[80.0, 120.0, 100.0],  # Different capacities per firm
        use_market_entry_exit=True,
        exit_threshold=-50.0,
        max_consecutive_losses=3,
        seed=42,
    )

    # Create ML-enhanced regulator
    regulator = MLRegulator(
        parallel_threshold=2.0,
        parallel_steps=3,
        structural_break_threshold=10.0,
        fine_amount=25.0,
        use_ml_detection=True,
        ml_anomaly_threshold=0.1,
        ml_collusion_threshold=0.7,
        seed=42,
    )

    # Create mixed agent types
    agents = [
        AdaptiveAgent(agent_id=0, learning_rate=0.1, exploration_rate=0.2, seed=42),
        BestResponseAgent(agent_id=1, seed=42),
        RandomAgent(agent_id=2, seed=42),
    ]

    # Set up logging
    log_file = Path("logs/enhanced_economic_demo.jsonl")
    logger = EpisodeLogger(
        log_file=log_file,
        n_firms=env.n_firms,
    )

    print("📊 Environment Configuration:")
    print(f"   - Firms: {env.n_firms}")
    print(f"   - Dynamic Elasticity: {env.use_dynamic_elasticity}")
    print(f"   - Capacity Constraints: {env.use_capacity_constraints}")
    print(f"   - Market Entry/Exit: {env.use_market_entry_exit}")
    print(f"   - ML Regulator: {regulator.use_ml_detection}")
    print("   - Agent Types: Adaptive, Best Response, Random")
    print()

    # Run episode
    obs, info = env.reset(seed=42)
    step = 0

    # Track metrics
    prices_history = []
    profits_history = []
    elasticity_history = []
    capacity_utilization = []
    ml_detections = []
    active_firms_history = []

    print("🔄 Running Enhanced Economic Episode...")
    print()

    while step < env.max_steps:
        # Get actions from agents
        actions = []
        for i, agent in enumerate(agents):
            if info.get("active_firms", [True] * env.n_firms)[i]:
                action = agent.choose_price(obs, env, info)
                actions.append(action)
            else:
                # Inactive firm sets price to 0
                actions.append(0.0)

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(np.array(actions))

        # Monitor with ML regulator
        detection_results = regulator.monitor_step(np.array(actions), step, info)

        # Apply penalties
        modified_rewards = regulator.apply_penalties(rewards, detection_results)

        # Update agents
        for i, agent in enumerate(agents):
            if hasattr(agent, "update_strategy"):
                was_violation = (
                    detection_results.get("parallel_violation", False)
                    or detection_results.get("structural_break_violation", False)
                    or detection_results.get("ml_collusion_detected", False)
                )
                agent.update_strategy(
                    price=actions[i],
                    profit=modified_rewards[i],
                    market_price=info["market_price"],
                    was_violation=was_violation,
                )

        # Log step
        additional_info = {
            # Enhanced features
            "current_elasticity": info.get("current_elasticity", env.price_elasticity),
            "capacity": info.get("capacity", [np.inf] * env.n_firms).tolist(),
            "active_firms": info.get("active_firms", [True] * env.n_firms).tolist(),
            "consecutive_losses": info.get(
                "consecutive_losses", [0] * env.n_firms
            ).tolist(),
            # ML detection results
            "ml_anomaly_detected": detection_results.get("ml_anomaly_detected", False),
            "ml_collusion_detected": detection_results.get(
                "ml_collusion_detected", False
            ),
            "ml_collusion_probability": detection_results.get(
                "ml_collusion_probability", 0.0
            ),
            "parallel_violation": detection_results.get("parallel_violation", False),
            "structural_break_violation": detection_results.get(
                "structural_break_violation", False
            ),
        }

        logger.log_step(
            step=step,
            prices=np.array(actions),
            profits=modified_rewards,
            demand_shock=info["demand_shock"],
            market_price=info["market_price"],
            total_demand=info["total_demand"],
            individual_quantity=info["individual_quantities"],
            total_profits=info["total_profits"],
            regulator_flags=detection_results,
            additional_info=additional_info,
        )

        # Track metrics
        prices_history.append(actions.copy())
        profits_history.append(modified_rewards.tolist())
        elasticity_history.append(info.get("current_elasticity", env.price_elasticity))

        # Calculate capacity utilization
        if info.get("use_capacity_constraints", False):
            capacity = info.get("capacity", [np.inf] * env.n_firms)
            quantities = info["individual_quantities"]
            utilization = [q / c if c > 0 else 0 for q, c in zip(quantities, capacity)]
            capacity_utilization.append(utilization)
        else:
            capacity_utilization.append([0.0] * env.n_firms)

        # Track ML detections
        ml_detections.append(
            {
                "anomaly": detection_results.get("ml_anomaly_detected", False),
                "collusion": detection_results.get("ml_collusion_detected", False),
                "probability": detection_results.get("ml_collusion_probability", 0.0),
            }
        )

        # Track active firms
        active_firms_history.append(
            info.get("active_firms", [True] * env.n_firms).copy()
        )

        step += 1

        # Print progress every 10 steps
        if step % 10 == 0:
            print(
                f"   Step {step:2d}: Market Price = {info['market_price']:.1f}, "
                f"Demand = {info['total_demand']:.1f}, "
                f"Elasticity = {info.get('current_elasticity', env.price_elasticity):.2f}"
            )

    # Log episode summary
    logger.log_episode_summary(
        total_steps=step,
        final_prices=actions,
        final_profits=info["total_profits"].tolist(),
        total_reward=float(np.sum(info["total_profits"])),
    )

    print()
    print("✅ Episode Complete!")
    print()

    # Print results summary
    print("📈 Results Summary:")
    print(f"   - Total Steps: {step}")
    print(f"   - Final Market Price: {info['market_price']:.2f}")
    print(f"   - Final Total Demand: {info['total_demand']:.2f}")
    print(
        f"   - Final Elasticity: {info.get('current_elasticity', env.price_elasticity):.3f}"
    )
    print(
        f"   - Active Firms: {sum(info.get('active_firms', [True] * env.n_firms))}/{env.n_firms}"
    )
    print()

    # ML Regulator Statistics
    ml_stats = regulator.get_ml_statistics()
    print("🤖 ML Regulator Statistics:")
    print(f"   - Training Samples: {ml_stats.get('training_samples', 0)}")
    print(f"   - Collusion Rate: {ml_stats.get('collusion_rate', 0.0):.3f}")
    print(f"   - ML Detections: {sum(1 for d in ml_detections if d['collusion'])}")
    print()

    # Agent Performance
    print("👥 Agent Performance:")
    for i, agent in enumerate(agents):
        if hasattr(agent, "get_strategy_statistics"):
            stats = agent.get_strategy_statistics()
            print(f"   - Agent {i} ({type(agent).__name__}):")
            print(f"     * Total Profits: {stats.get('total_profits', 0):.2f}")
            print(f"     * Avg Profit: {stats.get('avg_profit', 0):.2f}")
            print(f"     * Violation Rate: {stats.get('violation_rate', 0):.3f}")
            if "exploration_rate" in stats:
                print(f"     * Final Exploration Rate: {stats['exploration_rate']:.3f}")
        else:
            print(f"   - Agent {i} ({type(agent).__name__}): Basic agent")
    print()

    # Create visualizations
    create_enhanced_visualizations(
        prices_history,
        profits_history,
        elasticity_history,
        capacity_utilization,
        ml_detections,
        active_firms_history,
    )

    print(f"📁 Episode logged to: {log_file}")
    print("📊 Visualizations saved to: enhanced_economic_plots.png")
    print()
    print("🎉 Enhanced Economic Model Demo Complete!")


def create_enhanced_visualizations(
    prices_history,
    profits_history,
    elasticity_history,
    capacity_utilization,
    ml_detections,
    active_firms_history,
):
    """Create visualizations for the enhanced economic model."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Enhanced Economic Model Results", fontsize=16, fontweight="bold")

    steps = range(len(prices_history))

    # 1. Price Trajectories
    ax1 = axes[0, 0]
    prices_array = np.array(prices_history)
    for i in range(prices_array.shape[1]):
        ax1.plot(steps, prices_array[:, i], label=f"Firm {i}", linewidth=2)
    ax1.set_title("Price Trajectories")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Dynamic Elasticity
    ax2 = axes[0, 1]
    ax2.plot(steps, elasticity_history, "purple", linewidth=2)
    ax2.set_title("Dynamic Price Elasticity")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Elasticity")
    ax2.grid(True, alpha=0.3)

    # 3. Capacity Utilization
    ax3 = axes[1, 0]
    capacity_array = np.array(capacity_utilization)
    for i in range(capacity_array.shape[1]):
        ax3.plot(steps, capacity_array[:, i], label=f"Firm {i}", linewidth=2)
    ax3.set_title("Capacity Utilization")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Utilization Rate")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)

    # 4. ML Detection Results
    ax4 = axes[1, 1]
    ml_probs = [d["probability"] for d in ml_detections]
    ml_collusion = [1 if d["collusion"] else 0 for d in ml_detections]
    ax4.plot(steps, ml_probs, "orange", label="Collusion Probability", linewidth=2)
    ax4.plot(
        steps,
        ml_collusion,
        "red",
        label="ML Collusion Detection",
        linewidth=2,
        alpha=0.7,
    )
    ax4.set_title("ML Detection Results")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Probability / Detection")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)

    # 5. Firm Activity
    ax5 = axes[2, 0]
    active_array = np.array(active_firms_history)
    for i in range(active_array.shape[1]):
        ax5.plot(steps, active_array[:, i], label=f"Firm {i}", linewidth=2)
    ax5.set_title("Firm Activity (1=Active, 0=Exited)")
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Active Status")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.1, 1.1)

    # 6. Cumulative Profits
    ax6 = axes[2, 1]
    profits_array = np.array(profits_history)
    cumulative_profits = np.cumsum(profits_array, axis=0)
    for i in range(cumulative_profits.shape[1]):
        ax6.plot(steps, cumulative_profits[:, i], label=f"Firm {i}", linewidth=2)
    ax6.set_title("Cumulative Profits")
    ax6.set_xlabel("Step")
    ax6.set_ylabel("Cumulative Profit")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("enhanced_economic_plots.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_enhanced_economic_demo()
