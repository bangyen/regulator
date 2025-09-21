#!/usr/bin/env python3
"""
Generate a comprehensive episode log with both message data and enhanced regulator monitoring.

This script creates a realistic episode log that includes:
- Chat messages from different agent types
- Enhanced regulator monitoring with continuous risk scores
- ML detection results
- Market volatility and penalty tracking
"""

import json
import random
from datetime import datetime
from pathlib import Path


def generate_comprehensive_episode():
    """Generate a comprehensive episode log with both messages and regulator monitoring."""

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"comprehensive_episode_{timestamp}.jsonl"

    # Sample messages for different agent types
    competitive_messages = [
        "We need to stay competitive in this market.",
        "Our prices should reflect market conditions.",
        "Let's focus on our own strategy.",
        "Market competition is healthy for innovation.",
        "We should avoid any coordination on pricing.",
        "Let's compete fairly and let the market decide.",
        "Our pricing strategy should be independent.",
        "Healthy competition benefits everyone.",
    ]

    collusive_messages = [
        "Let's coordinate our pricing strategy.",
        "We should set similar prices to maximize profits.",
        "How about we all price at $50?",
        "Let's work together on this market.",
        "We can all benefit from higher prices.",
        "Let's avoid undercutting each other.",
        "We should maintain price discipline.",
        "Let's keep prices stable and profitable.",
    ]

    neutral_messages = [
        "Market conditions look stable today.",
        "Our prices seem reasonable.",
        "Let's see how the market responds.",
        "We should monitor competitor behavior.",
        "The demand seems to be holding steady.",
        "Market dynamics are interesting today.",
        "Let's observe how others price.",
        "We should be strategic about our approach.",
    ]

    # Create episode header
    episode_header = {
        "type": "episode_header",
        "episode_id": f"comprehensive_episode_{timestamp}",
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
            "max_steps": 20,
        },
    }

    # Create step data with both messages and enhanced regulator monitoring
    steps = []
    base_prices = [45.0, 48.0, 42.0]  # Starting prices for each firm

    for step_num in range(1, 21):
        # Generate some messages for this step (randomly)
        messages = []
        if random.random() < 0.8:  # 80% chance of having messages
            num_messages = random.randint(1, 2)
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

        # Generate realistic price evolution with some coordination
        if step_num > 1:
            # Add some price coordination over time
            if random.random() < 0.3:  # 30% chance of coordination
                target_price = random.uniform(40, 60)
                for i in range(3):
                    base_prices[i] = target_price + random.uniform(-2, 2)
            else:
                # Normal competitive pricing
                for i in range(3):
                    base_prices[i] = max(
                        10, min(90, base_prices[i] + random.uniform(-5, 5))
                    )

        prices = [round(p, 2) for p in base_prices]
        market_price = round(sum(prices) / len(prices), 2)

        # Generate demand shock
        demand_shock = round(random.uniform(-8, 8), 2)
        total_demand = max(10, round(100 - market_price + demand_shock, 2))

        # Calculate individual quantities (simplified)
        individual_quantities = []
        for price in prices:
            if price < market_price:
                quantity = round(total_demand * 0.4 + random.uniform(-5, 5), 2)
            else:
                quantity = round(total_demand * 0.2 + random.uniform(-3, 3), 2)
            individual_quantities.append(max(0, quantity))

        # Calculate profits
        profits = []
        for i, (price, quantity) in enumerate(zip(prices, individual_quantities)):
            profit = round((price - 10) * quantity, 2)  # marginal_cost = 10
            profits.append(profit)

        # Generate enhanced regulator monitoring data
        # Calculate risk scores based on price patterns
        risk_score = 0.0
        market_volatility = 0.0

        if step_num > 1:
            # Calculate price volatility
            prev_prices = [steps[-1]["prices"][i] for i in range(3)]
            price_changes = [abs(prices[i] - prev_prices[i]) for i in range(3)]
            market_volatility = round(sum(price_changes) / len(price_changes), 3)

            # Calculate risk score based on coordination
            price_std = round(
                sum((p - market_price) ** 2 for p in prices) / len(prices), 2
            )
            if price_std < 5:  # Low price variation suggests coordination
                risk_score = round(random.uniform(0.6, 0.9), 3)
            elif price_std < 15:
                risk_score = round(random.uniform(0.3, 0.6), 3)
            else:
                risk_score = round(random.uniform(0.0, 0.3), 3)

        # Determine violations based on risk score
        parallel_violation = risk_score > 0.7 and random.random() < 0.4
        structural_violation = market_volatility > 3 and random.random() < 0.3

        # Calculate fines
        fines_applied = [0.0, 0.0, 0.0]
        if parallel_violation:
            for i in range(3):
                fines_applied[i] = round(random.uniform(20, 50), 2)
        if structural_violation:
            for i in range(3):
                fines_applied[i] += round(random.uniform(15, 35), 2)

        # Generate penalty multipliers
        penalty_multipliers = []
        for i in range(3):
            if fines_applied[i] > 0:
                penalty_multipliers.append(round(random.uniform(1.0, 2.0), 2))
            else:
                penalty_multipliers.append(1.0)

        # Create enhanced regulator flags
        regulator_flags = {
            "parallel_violation": parallel_violation,
            "structural_break_violation": structural_violation,
            "fines_applied": fines_applied,
            "violation_details": [],
            "risk_score": risk_score,
            "market_volatility": market_volatility,
            "penalty_multipliers": penalty_multipliers,
            "violation_severities": [
                round(random.uniform(0.5, 1.0), 2) for _ in range(3)
            ],
        }

        # Add violation details
        if parallel_violation:
            regulator_flags["violation_details"].append(
                f"Parallel pricing detected at step {step_num}"
            )
        if structural_violation:
            regulator_flags["violation_details"].append(
                f"Structural break detected at step {step_num}"
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
                        "confidence": round(random.uniform(0.7, 0.95), 2),
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
            "prices": prices,
            "profits": profits,
            "demand_shock": demand_shock,
            "market_price": market_price,
            "total_demand": total_demand,
            "individual_quantity": individual_quantities,
            "total_profits": profits,
            "regulator_flags": regulator_flags,
            "additional_info": {
                "agent_types": ["competitive_chat", "collusive_chat", "neutral_chat"],
                "agent_prices": prices,
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
    total_fines = sum(sum(step["regulator_flags"]["fines_applied"]) for step in steps)
    total_messages = sum(len(step.get("messages", [])) for step in steps)
    total_collusive_messages = sum(
        step.get("chat_monitoring", {}).get("collusive_messages", 0) for step in steps
    )

    episode_end = {
        "type": "episode_end",
        "episode_id": f"comprehensive_episode_{timestamp}",
        "end_time": datetime.now().isoformat(),
        "duration_seconds": 0.2,
        "total_steps": 20,
        "terminated": False,
        "truncated": True,
        "final_rewards": [
            round(sum(step["profits"][i] for step in steps), 2) for i in range(3)
        ],
        "episode_summary": {
            "total_steps": 20,
            "final_market_price": steps[-1]["market_price"],
            "total_profits": [
                round(sum(step["profits"][i] for step in steps), 2) for i in range(3)
            ],
            "total_fines": total_fines,
            "total_messages": total_messages,
            "collusive_messages": total_collusive_messages,
            "violations": {
                "parallel": sum(
                    1 for step in steps if step["regulator_flags"]["parallel_violation"]
                ),
                "structural_break": sum(
                    1
                    for step in steps
                    if step["regulator_flags"]["structural_break_violation"]
                ),
            },
            "avg_risk_score": round(
                sum(step["regulator_flags"]["risk_score"] for step in steps)
                / len(steps),
                3,
            ),
            "avg_market_volatility": round(
                sum(step["regulator_flags"]["market_volatility"] for step in steps)
                / len(steps),
                3,
            ),
            "agent_types": ["competitive_chat", "collusive_chat", "neutral_chat"],
            "environment_params": {
                "n_firms": 3,
                "max_steps": 20,
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

    print(f"Created comprehensive episode log: {log_file}")
    print(f"Episode contains {total_messages} total messages")
    print(f"Collusive messages: {total_collusive_messages}")
    print(f"Total fines applied: ${total_fines:.2f}")
    print(f"Average risk score: {episode_end['episode_summary']['avg_risk_score']}")
    print(
        f"Average market volatility: {episode_end['episode_summary']['avg_market_volatility']}"
    )
    print(f"Violations: {episode_end['episode_summary']['violations']}")

    return log_file


if __name__ == "__main__":
    generate_comprehensive_episode()
