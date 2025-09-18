#!/usr/bin/env python3
"""
Demo script for chat-enabled firm agents with LLM-based collusion detection.

This script demonstrates the integration of natural language chat between firms
and an LLM-based regulator detector. It runs episodes with chat-enabled agents
and shows how messages are classified and monitored for collusive behavior.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.chat_firm import (
    ChatFirmAgent,
    CollusiveChatAgent,
    CompetitiveChatAgent,
    ChatMessageManager,
)
from src.detectors.llm_detector import LLMDetector, ChatRegulator
from src.cartel.cartel_env import CartelEnv
from src.agents.regulator import Regulator
from src.episode_logging.episode_logger import EpisodeLogger


def create_chat_agents(
    n_firms: int,
    agent_types: List[str],
    chat_enabled: bool = True,
    seed: Optional[int] = None,
) -> List[ChatFirmAgent]:
    """
    Create a list of chat-enabled agents.

    Args:
        n_firms: Number of firms to create
        agent_types: List of agent types ("collusive", "competitive", "random")
        chat_enabled: Whether agents can generate messages
        seed: Random seed for reproducibility

    Returns:
        List of chat-enabled agents
    """
    agents = []

    for i in range(n_firms):
        agent_type = agent_types[i % len(agent_types)]

        if agent_type == "collusive":
            agent = CollusiveChatAgent(
                agent_id=i,
                chat_enabled=chat_enabled,
                message_frequency=0.6,
                collusion_intensity=0.8,
                seed=seed + i if seed else None,
            )
        elif agent_type == "competitive":
            agent = CompetitiveChatAgent(
                agent_id=i,
                chat_enabled=chat_enabled,
                message_frequency=0.4,
                seed=seed + i if seed else None,
            )
        else:  # random
            agent = ChatFirmAgent(
                agent_id=i,
                chat_enabled=chat_enabled,
                message_frequency=0.3,
                seed=seed + i if seed else None,
            )

        agents.append(agent)

    return agents


def run_chat_episode(
    env: CartelEnv,
    agents: List[ChatFirmAgent],
    message_manager: ChatMessageManager,
    chat_regulator: ChatRegulator,
    price_regulator: Regulator,
    logger: EpisodeLogger,
    n_steps: int = 50,
) -> Dict[str, Any]:
    """
    Run a single episode with chat-enabled agents.

    Args:
        env: The CartelEnv environment
        agents: List of chat-enabled agents
        message_manager: Message manager for coordinating chat
        chat_regulator: Regulator for monitoring chat messages
        price_regulator: Regulator for monitoring price behavior
        logger: Episode logger
        n_steps: Number of steps to run

    Returns:
        Dictionary containing episode results
    """
    # Reset environment and agents
    observation = env.reset()
    for agent in agents:
        agent.reset()
    message_manager.reset()
    chat_regulator.reset()
    price_regulator.reset()

    episode_results = {
        "total_steps": n_steps,
        "total_messages": 0,
        "collusive_messages": 0,
        "total_chat_fines": 0.0,
        "total_price_fines": 0.0,
        "message_history": [],
        "detection_history": [],
        "price_history": [],
        "profit_history": [],
    }

    # Log episode header
    logger.log_episode_header(
        episode_id=0,
        n_firms=env.n_firms,
        n_steps=n_steps,
        agent_types=[type(agent).__name__ for agent in agents],
        environment_params={
            "price_min": env.price_min,
            "price_max": env.price_max,
            "marginal_cost": env.marginal_cost,
            "demand_intercept": env.demand_intercept,
            "demand_slope": env.demand_slope,
        },
    )

    for step in range(n_steps):
        # Collect messages from agents
        messages = message_manager.collect_messages(
            step=step, observation=observation, env=env
        )

        # Distribute messages to all agents
        message_manager.distribute_messages(messages, step)

        # Monitor messages for collusion
        chat_monitoring = chat_regulator.monitor_messages(messages, step)

        # Choose prices
        prices = []
        for agent in agents:
            price = agent.choose_price(observation, env)
            prices.append(price)

        prices_array = np.array(prices)

        # Monitor prices for collusion
        price_monitoring = price_regulator.monitor_step(prices_array, step)

        # Step environment
        observation, rewards, terminated, truncated, info = env.step(prices_array)

        # Apply penalties
        modified_rewards = price_regulator.apply_penalties(rewards, price_monitoring)

        # Update agent histories
        for i, agent in enumerate(agents):
            rival_prices = np.concatenate([prices[:i], prices[i + 1 :]])
            agent.update_history(prices[i], rival_prices)

        # Log step
        logger.log_step(
            step=step,
            prices=prices_array,
            profits=modified_rewards,
            demand_shock=info.get("demand_shock", 0.0),
            market_price=float(np.mean(prices)),
            total_demand=info.get("total_demand", 0.0),
            individual_quantity=info.get("individual_quantity", 0.0),
            total_profits=modified_rewards,
            regulator_flags=price_monitoring,
            additional_info=None,
            rewards=rewards.tolist(),
            messages=messages,
            chat_monitoring=chat_monitoring,
            price_monitoring=price_monitoring,
        )

        # Update episode results
        episode_results["total_messages"] += len(messages)
        episode_results["collusive_messages"] += chat_monitoring["collusive_messages"]
        episode_results["total_chat_fines"] += chat_monitoring["fines_applied"]
        episode_results["total_price_fines"] += float(
            np.sum(price_monitoring["fines_applied"])
        )
        episode_results["message_history"].extend(messages)
        episode_results["detection_history"].extend(
            chat_monitoring.get("classifications", [])
        )
        episode_results["price_history"].append(prices.copy())
        episode_results["profit_history"].append(modified_rewards.tolist())

        if terminated or truncated:
            break

    # Log episode summary
    logger.log_episode_summary(
        total_reward=float(np.sum(episode_results["profit_history"][-1])),
        total_steps=len(episode_results["price_history"]),
        final_prices=prices,
        final_profits=modified_rewards.tolist(),
        chat_summary=chat_regulator.get_message_violation_summary(),
        price_summary=price_regulator.get_violation_summary(),
    )

    return episode_results


def print_episode_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of the episode results.

    Args:
        results: Episode results dictionary
    """
    print("\n" + "=" * 60)
    print("CHAT ENABLED EPISODE SUMMARY")
    print("=" * 60)

    print(f"Total Steps: {results['total_steps']}")
    print(f"Total Messages: {results['total_messages']}")
    print(f"Collusive Messages Detected: {results['collusive_messages']}")
    print(
        f"Collusion Rate: {results['collusive_messages']/max(1, results['total_messages']):.2%}"
    )

    print("\nFines Applied:")
    print(f"  Chat Fines: ${results['total_chat_fines']:.2f}")
    print(f"  Price Fines: ${results['total_price_fines']:.2f}")
    print(
        f"  Total Fines: ${results['total_chat_fines'] + results['total_price_fines']:.2f}"
    )

    # Show sample messages
    if results["message_history"]:
        print("\nSample Messages:")
        for i, msg in enumerate(results["message_history"][:5]):
            print(f"  {i+1}. Agent {msg['sender_id']}: {msg['message']}")
        if len(results["message_history"]) > 5:
            print(f"  ... and {len(results['message_history']) - 5} more messages")

    # Show detection results
    if results["detection_history"]:
        collusive_detections = [
            d for d in results["detection_history"] if d["is_collusive"]
        ]
        if collusive_detections:
            print("\nCollusive Messages Detected:")
            for i, detection in enumerate(collusive_detections[:3]):
                print(
                    f"  {i+1}. Agent {detection['sender_id']}: {detection['message'][:50]}..."
                )
                print(f"     Confidence: {detection['confidence']:.2f}")
            if len(collusive_detections) > 3:
                print(
                    f"  ... and {len(collusive_detections) - 3} more collusive messages"
                )


def main():
    """Main function to run the chat demo."""
    parser = argparse.ArgumentParser(description="Run chat-enabled firm episode demo")
    parser.add_argument(
        "--n-firms", type=int, default=2, help="Number of firms (default: 2)"
    )
    parser.add_argument(
        "--n-steps", type=int, default=50, help="Number of steps (default: 50)"
    )
    parser.add_argument(
        "--agent-types",
        nargs="+",
        default=["collusive", "competitive"],
        choices=["collusive", "competitive", "random"],
        help="Agent types to use (default: collusive competitive)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="Output directory for logs (default: logs)",
    )
    parser.add_argument(
        "--no-chat", action="store_true", help="Disable chat functionality"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create environment
    env = CartelEnv(
        n_firms=args.n_firms,
        price_min=10.0,
        price_max=100.0,
        marginal_cost=20.0,
        demand_intercept=100.0,
        demand_slope=-0.5,
        seed=args.seed,
    )

    # Create agents
    agents = create_chat_agents(
        n_firms=args.n_firms,
        agent_types=args.agent_types,
        chat_enabled=not args.no_chat,
        seed=args.seed,
    )

    # Create message manager
    message_manager = ChatMessageManager(agents)

    # Create regulators
    llm_detector = LLMDetector(
        model_type="stubbed",
        confidence_threshold=0.6,
        seed=args.seed,
    )
    chat_regulator = ChatRegulator(
        llm_detector=llm_detector,
        message_fine_amount=25.0,
        collusion_threshold=0.6,
    )
    price_regulator = Regulator(
        parallel_threshold=5.0,
        parallel_steps=3,
        structural_break_threshold=15.0,
        fine_amount=50.0,
        seed=args.seed,
    )

    # Create logger
    logger = EpisodeLogger(output_dir / "chat_demo_episode.jsonl", n_firms=args.n_firms)

    print("Starting Chat-Enabled Episode Demo")
    print(f"Firms: {args.n_firms}")
    print(f"Steps: {args.n_steps}")
    print(f"Agent Types: {args.agent_types}")
    print(f"Chat Enabled: {not args.no_chat}")
    print(f"Seed: {args.seed}")
    print(f"Output: {logger.log_file}")

    # Run episode
    try:
        results = run_chat_episode(
            env=env,
            agents=agents,
            message_manager=message_manager,
            chat_regulator=chat_regulator,
            price_regulator=price_regulator,
            logger=logger,
            n_steps=args.n_steps,
        )

        # Print summary
        print_episode_summary(results)

        # Export detection results
        detection_file = output_dir / "chat_detection_results.json"
        llm_detector.export_detection_results(str(detection_file))
        print(f"\nDetection results exported to: {detection_file}")

        print("\nEpisode completed successfully!")
        print(f"Log file: {logger.log_file}")

    except Exception as e:
        print(f"Error running episode: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
