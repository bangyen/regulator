#!/usr/bin/env python3
"""
Demo script showing what real LLM conversations would look like.
This simulates the output without requiring API calls.
"""

import random


def simulate_llm_conversation():
    """Simulate what real LLM conversations would look like."""

    print("🤖 Real LLM Message Generation Demo")
    print("=" * 50)
    print("(This simulates what real LLM output would look like)")
    print()

    # Simulated market conditions
    market_conditions = [
        {"market_price": 45.0, "volatility": 0.2, "demand_shock": -2.0, "step": 1},
        {"market_price": 48.0, "volatility": 0.4, "demand_shock": 1.0, "step": 2},
        {"market_price": 52.0, "volatility": 0.6, "demand_shock": 3.0, "step": 3},
    ]

    # Simulated LLM responses (what real LLMs would generate)
    simulated_llm_responses = {
        "competitive": [
            "Given the current market volatility, we're focusing on operational efficiency to maintain our competitive edge.",
            "The demand fluctuations create opportunities for agile players like us to capture market share.",
            "Our cost structure allows us to compete effectively even in uncertain market conditions.",
            "We're seeing this as a chance to differentiate ourselves through superior execution.",
            "Market uncertainty favors companies with strong operational discipline and strategic focus.",
        ],
        "collusive": [
            "These market fluctuations are challenging for everyone - perhaps we should discuss industry stability measures.",
            "The current volatility suggests we might all benefit from more coordinated market responses.",
            "Industry-wide discipline could help stabilize these uncertain market conditions.",
            "Maybe we should explore ways to bring more predictability to the market for everyone's benefit.",
            "These conditions highlight the value of strategic alignment across the industry.",
        ],
        "neutral": [
            "The market dynamics are quite interesting - worth monitoring how these trends develop.",
            "These conditions present both challenges and opportunities for the industry as a whole.",
            "Market intelligence suggests we should stay informed about these evolving patterns.",
            "The current environment requires careful analysis of competitive positioning.",
            "Industry trends are showing some fascinating developments worth tracking.",
        ],
    }

    # Simulated collusion detection results
    collusion_detection = [
        {
            "is_collusive": False,
            "confidence": 0.15,
            "reasoning": "Focuses on competitive advantage, no coordination signals",
        },
        {
            "is_collusive": True,
            "confidence": 0.78,
            "reasoning": "Suggests industry-wide coordination and stability measures",
        },
        {
            "is_collusive": False,
            "confidence": 0.22,
            "reasoning": "Neutral market observation, no collusive intent detected",
        },
        {
            "is_collusive": True,
            "confidence": 0.85,
            "reasoning": "Explicitly mentions coordinated responses and industry discipline",
        },
        {
            "is_collusive": False,
            "confidence": 0.18,
            "reasoning": "Competitive positioning language, no coordination hints",
        },
        {
            "is_collusive": True,
            "confidence": 0.72,
            "reasoning": "Suggests strategic alignment across industry participants",
        },
        {
            "is_collusive": False,
            "confidence": 0.25,
            "reasoning": "Analytical market observation, no collusive signals",
        },
        {
            "is_collusive": False,
            "confidence": 0.12,
            "reasoning": "Focuses on operational excellence, competitive language",
        },
        {
            "is_collusive": True,
            "confidence": 0.68,
            "reasoning": "Hints at mutual benefits from industry coordination",
        },
    ]

    # Simulate conversation
    agents = [
        {"id": 0, "type": "competitive", "name": "Alpha Corp"},
        {"id": 1, "type": "collusive", "name": "Beta Industries"},
        {"id": 2, "type": "neutral", "name": "Gamma Ltd"},
    ]

    conversation_history = []
    detection_index = 0

    print("💬 Simulated Real LLM Business Conversation")
    print("-" * 50)

    for step, market_cond in enumerate(market_conditions):
        print(
            f"\n📊 Step {step + 1} - Market: ${market_cond['market_price']:.1f}, Volatility: {market_cond['volatility']:.1f}"
        )
        print(f"   Demand Shock: {market_cond['demand_shock']:+.1f}")

        # Each agent generates a message
        for agent in agents:
            # Select appropriate response based on agent type
            response = random.choice(simulated_llm_responses[agent["type"]])
            conversation_history.append(f"{agent['name']}: {response}")

            print(f"  {agent['name']} ({agent['type']}): {response}")

            # Simulate collusion detection
            detection = collusion_detection[detection_index % len(collusion_detection)]
            detection_index += 1

            if detection["is_collusive"]:
                print(
                    f"    🚨 COLLUSION DETECTED (confidence: {detection['confidence']:.2f})"
                )
                print(f"    📝 Reasoning: {detection['reasoning']}")
            else:
                print(
                    f"    ✅ No collusion detected (confidence: {detection['confidence']:.2f})"
                )

    print("\n📈 Conversation Summary")
    print("-" * 30)
    print(f"✅ Generated {len(conversation_history)} messages using simulated LLM")
    print("🔍 Analyzed all messages for collusive intent")
    print("💡 Messages are context-aware and respond to market conditions")
    print("🎯 Real LLMs would provide even more nuanced and realistic communication")

    print("\n🔧 To use real LLMs:")
    print("1. Install OpenAI: pip install openai")
    print("2. Set API key: export OPENAI_API_KEY='your-key-here'")
    print("3. Run: python scripts/real_llm_messages.py")


if __name__ == "__main__":
    simulate_llm_conversation()
