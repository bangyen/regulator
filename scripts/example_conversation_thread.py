#!/usr/bin/env python3
"""
Example conversation thread showing realistic business communication
with LLM-generated messages and collusion detection.
"""


def show_example_conversation_thread():
    """Show an example of a realistic business conversation thread."""

    print("💬 Example Business Conversation Thread")
    print("=" * 60)
    print("(Simulating what real LLM-generated conversations would look like)")
    print()

    # Example conversation thread
    conversation = [
        {
            "step": 1,
            "sender": "Alpha Corp (Competitive)",
            "message": "The recent market volatility is creating opportunities for efficient players. How are you positioning for the current conditions?",
            "market_context": "Market price: $47.50, Volatility: 0.35, Demand shock: -1.5",
            "collusion_analysis": {
                "detected": False,
                "confidence": 0.12,
                "reasoning": "Competitive positioning question, no coordination signals",
            },
        },
        {
            "step": 1,
            "sender": "Beta Industries (Collusive)",
            "message": "These fluctuations are challenging for everyone's planning. Perhaps we should discuss how to bring more stability to the market.",
            "market_context": "Market price: $47.50, Volatility: 0.35, Demand shock: -1.5",
            "collusion_analysis": {
                "detected": True,
                "confidence": 0.82,
                "reasoning": "Suggests industry-wide coordination for market stability",
            },
        },
        {
            "step": 2,
            "sender": "Alpha Corp (Competitive)",
            "message": "I understand the concern, but we see this as a chance to differentiate through operational excellence. Our focus is on competitive advantage.",
            "market_context": "Market price: $49.20, Volatility: 0.42, Demand shock: +0.8",
            "collusion_analysis": {
                "detected": False,
                "confidence": 0.08,
                "reasoning": "Rejects coordination, emphasizes competitive strategy",
            },
        },
        {
            "step": 2,
            "sender": "Gamma Ltd (Neutral)",
            "message": "The market dynamics are quite interesting. We're monitoring how these trends might affect industry structure long-term.",
            "market_context": "Market price: $49.20, Volatility: 0.42, Demand shock: +0.8",
            "collusion_analysis": {
                "detected": False,
                "confidence": 0.15,
                "reasoning": "Neutral market observation, no collusive intent",
            },
        },
        {
            "step": 3,
            "sender": "Beta Industries (Collusive)",
            "message": "While competition is healthy, some coordination on capacity planning could benefit everyone. What's your view on industry-wide discipline?",
            "market_context": "Market price: $51.80, Volatility: 0.58, Demand shock: +2.3",
            "collusion_analysis": {
                "detected": True,
                "confidence": 0.91,
                "reasoning": "Explicitly suggests capacity coordination and industry discipline",
            },
        },
        {
            "step": 3,
            "sender": "Alpha Corp (Competitive)",
            "message": "We prefer to let market forces determine optimal capacity. Our strategy is to be the most efficient operator regardless of industry coordination.",
            "market_context": "Market price: $51.80, Volatility: 0.58, Demand shock: +2.3",
            "collusion_analysis": {
                "detected": False,
                "confidence": 0.05,
                "reasoning": "Rejects coordination, emphasizes market-based competition",
            },
        },
        {
            "step": 4,
            "sender": "Gamma Ltd (Neutral)",
            "message": "This is a fascinating strategic discussion. The tension between competition and coordination is a key theme in oligopoly theory.",
            "market_context": "Market price: $53.40, Volatility: 0.45, Demand shock: +1.7",
            "collusion_analysis": {
                "detected": False,
                "confidence": 0.18,
                "reasoning": "Academic observation, no actionable collusion",
            },
        },
        {
            "step": 4,
            "sender": "Beta Industries (Collusive)",
            "message": "Perhaps we could explore this further in a more private setting. There might be mutual benefits from strategic alignment.",
            "market_context": "Market price: $53.40, Volatility: 0.45, Demand shock: +1.7",
            "collusion_analysis": {
                "detected": True,
                "confidence": 0.95,
                "reasoning": "Suggests private meeting and explicit strategic alignment",
            },
        },
    ]

    # Display the conversation
    for i, msg in enumerate(conversation, 1):
        print(f"📝 Message {i}")
        print(f"   Step: {msg['step']}")
        print(f"   From: {msg['sender']}")
        print(f"   Context: {msg['market_context']}")
        print(f"   Message: \"{msg['message']}\"")

        # Show collusion analysis
        analysis = msg["collusion_analysis"]
        if analysis["detected"]:
            print(
                f"   🚨 COLLUSION DETECTED (confidence: {analysis['confidence']:.2f})"
            )
            print(f"   📝 Reasoning: {analysis['reasoning']}")
        else:
            print(f"   ✅ No collusion (confidence: {analysis['confidence']:.2f})")
            print(f"   📝 Reasoning: {analysis['reasoning']}")

        print()

    # Summary
    print("📊 Conversation Analysis Summary")
    print("-" * 40)

    total_messages = len(conversation)
    collusive_messages = sum(
        1 for msg in conversation if msg["collusion_analysis"]["detected"]
    )
    avg_confidence = (
        sum(msg["collusion_analysis"]["confidence"] for msg in conversation)
        / total_messages
    )

    print(f"Total messages: {total_messages}")
    print(f"Collusive messages detected: {collusive_messages}")
    print(f"Collusion rate: {(collusive_messages/total_messages)*100:.1f}%")
    print(f"Average detection confidence: {avg_confidence:.2f}")

    print("\n🎯 Key Observations:")
    print("• Competitive agent consistently rejects coordination attempts")
    print("• Collusive agent progressively escalates coordination suggestions")
    print("• Neutral agent provides analytical commentary")
    print("• LLM detection accurately identifies collusive patterns")
    print("• Messages are context-aware and respond to market conditions")

    print("\n💡 Real LLM Benefits:")
    print("• Natural, realistic business language")
    print("• Context-aware responses to market conditions")
    print("• Nuanced collusion detection")
    print("• Realistic conversation flow and escalation")
    print("• Professional business communication style")


if __name__ == "__main__":
    show_example_conversation_thread()
