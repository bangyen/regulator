#!/usr/bin/env python3
"""
Test script to generate sample episode data with messages for testing the dashboard.

This script creates a sample episode log with chat messages to test the new
message visualization features in the dashboard.
"""

import json
import random
from datetime import datetime
from pathlib import Path


def create_sample_message_episode():
    """Create a sample episode log file with message data for testing."""

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"test_messages_{timestamp}.jsonl"

    # Sample messages for different agent types
    competitive_messages = [
        "We need to stay competitive in this market.",
        "Our prices should reflect market conditions.",
        "Let's focus on our own strategy.",
        "Market competition is healthy for innovation.",
        "We should avoid any coordination on pricing.",
    ]

    collusive_messages = [
        "Let's coordinate our pricing strategy.",
        "We should set similar prices to maximize profits.",
        "How about we all price at $50?",
        "Let's work together on this market.",
        "We can all benefit from higher prices.",
    ]

    neutral_messages = [
        "Market conditions look stable today.",
        "Our prices seem reasonable.",
        "Let's see how the market responds.",
        "We should monitor competitor behavior.",
        "The demand seems to be holding steady.",
    ]

    # Create episode header
    episode_header = {
        "type": "episode_header",
        "episode_id": f"test_messages_{timestamp}",
        "start_time": datetime.now().isoformat(),
        "n_firms": 3,
        "log_file": str(log_file),
        "environment_params": {
            "price_min": 1.0,
            "price_max": 100.0,
            "marginal_cost": 10.0,
            "demand_intercept": 100.0,
            "demand_slope": -1.0,
            "shock_std": 5.0,
            "max_steps": 10,
        },
    }

    # Create step data with messages
    steps = []
    for step_num in range(1, 11):
        # Generate some messages for this step (randomly)
        messages = []
        if random.random() < 0.7:  # 70% chance of having messages
            num_messages = random.randint(1, 3)
            for _ in range(num_messages):
                sender_id = random.randint(0, 2)
                message_type = random.choice(["competitive", "collusive", "neutral"])

                if message_type == "competitive":
                    message_text = random.choice(competitive_messages)
                elif message_type == "collusive":
                    message_text = random.choice(collusive_messages)
                else:
                    message_text = random.choice(neutral_messages)

                messages.append(
                    {
                        "message": message_text,
                        "sender_id": sender_id,
                        "step": step_num,
                        "timestamp": step_num,
                    }
                )

        # Create chat monitoring data
        chat_monitoring = None
        if messages:
            collusive_count = sum(
                1 for msg in messages if msg["message"] in collusive_messages
            )
            chat_monitoring = {
                "messages_analyzed": len(messages),
                "collusive_messages": collusive_count,
                "fines_applied": collusive_count
                * 25.0,  # $25 fine per collusive message
                "violation_details": [
                    f"Collusive message detected: {msg['message']}"
                    for msg in messages
                    if msg["message"] in collusive_messages
                ],
                "classifications": [
                    {
                        "step": step_num,
                        "message": msg["message"],
                        "classification": (
                            "collusive"
                            if msg["message"] in collusive_messages
                            else "competitive"
                        ),
                        "confidence": random.uniform(0.7, 0.95),
                        "reasoning": f"Message contains {'collusive' if msg['message'] in collusive_messages else 'competitive'} language patterns",
                    }
                    for msg in messages
                ],
            }

        # Create step data
        step_data = {
            "type": "step",
            "step": step_num,
            "timestamp": datetime.now().isoformat(),
            "prices": [random.uniform(20, 80) for _ in range(3)],
            "profits": [random.uniform(100, 500) for _ in range(3)],
            "demand_shock": random.uniform(-10, 10),
            "market_price": random.uniform(30, 70),
            "total_demand": random.uniform(40, 80),
            "individual_quantity": [random.uniform(5, 25) for _ in range(3)],
            "total_profits": [random.uniform(200, 800) for _ in range(3)],
            "regulator_flags": {
                "parallel_violation": random.random() < 0.2,
                "structural_break_violation": random.random() < 0.3,
                "fines_applied": [random.uniform(0, 50) for _ in range(3)],
                "violation_details": [],
            },
            "additional_info": {
                "agent_types": ["competitive_chat", "collusive_chat", "neutral_chat"],
                "agent_prices": [random.uniform(20, 80) for _ in range(3)],
            },
        }

        # Add messages and chat monitoring if available
        if messages:
            step_data["messages"] = messages
            step_data["n_messages"] = len(messages)

        if chat_monitoring:
            step_data["chat_monitoring"] = chat_monitoring

        steps.append(step_data)

    # Create episode end
    episode_end = {
        "type": "episode_end",
        "episode_id": f"test_messages_{timestamp}",
        "end_time": datetime.now().isoformat(),
        "duration_seconds": 0.1,
        "total_steps": 10,
        "terminated": False,
        "truncated": True,
        "final_rewards": [random.uniform(1000, 5000) for _ in range(3)],
        "episode_summary": {
            "total_steps": 10,
            "final_market_price": random.uniform(30, 70),
            "total_profits": [random.uniform(2000, 8000) for _ in range(3)],
            "agent_types": ["competitive_chat", "collusive_chat", "neutral_chat"],
            "environment_params": {
                "n_firms": 3,
                "max_steps": 10,
                "marginal_cost": 10.0,
                "demand_intercept": 100.0,
                "demand_slope": -1.0,
                "shock_std": 5.0,
                "price_min": 1.0,
                "price_max": 100.0,
            },
        },
    }

    # Write to file
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(episode_header) + "\n")
        for step in steps:
            f.write(json.dumps(step) + "\n")
        f.write(json.dumps(episode_end) + "\n")

    print(f"Created test episode log: {log_file}")
    print(
        f"Episode contains {sum(len(step.get('messages', [])) for step in steps)} total messages"
    )
    print(
        f"Collusive messages: {sum(step.get('chat_monitoring', {}).get('collusive_messages', 0) for step in steps)}"
    )

    return log_file


if __name__ == "__main__":
    create_sample_message_episode()
