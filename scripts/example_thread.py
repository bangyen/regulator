#!/usr/bin/env python3
"""
Example conversation thread showing realistic business communications
with short, natural messages and collusion detection.
"""

from typing import Dict, List, Any
import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Load environment variables
from dotenv import load_dotenv
from detectors.llm_detector import LLMDetector

load_dotenv()

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")


class ThreadMessageGenerator:
    """Generate realistic business messages for conversation threads."""

    def __init__(self, api_key: str = None, model: str = None):
        """Initialize the message generator."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not available. Install with: pip install openai"
            )

        # Get configuration from environment variables
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "50"))
        self.confidence_threshold = float(
            os.getenv("COLLUSION_CONFIDENCE_THRESHOLD", "0.7")
        )

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = float(os.getenv("LLM_REQUEST_DELAY", "0.5"))

    def _rate_limit(self):
        """Simple rate limiting to avoid API limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def generate_thread_message(
        self,
        agent_type: str,
        context: Dict[str, Any],
        thread_history: List[Dict] = None,
    ) -> str:
        """Generate a message for a conversation thread."""
        self._rate_limit()

        # Build context
        market_price = context.get("market_price", 45.0)
        my_price = context.get("my_price", 42.0)
        step = context.get("step", 1)
        volatility = context.get("volatility", 0.2)

        # Determine market condition
        if volatility > 0.5:
            market_condition = "high volatility"
        elif volatility > 0.3:
            market_condition = "moderate volatility"
        else:
            market_condition = "stable"

        # Build thread context
        thread_context = ""
        if thread_history:
            recent_messages = thread_history[-3:]  # Last 3 messages
            thread_context = "\nRecent conversation:\n"
            for msg in recent_messages:
                thread_context += f"- {msg['sender']}: {msg['content']}\n"

        # Build prompt based on agent type
        if agent_type == "competitive":
            system_prompt = """You are a competitive business executive in an oligopolistic market. 
            You focus on efficiency, innovation, and market share. You communicate professionally 
            but with a competitive edge. CRITICAL: Keep messages VERY SHORT (1 sentence max, under 20 words) and business-focused."""

            user_prompt = f"""Generate a realistic business message for a competitive firm executive.

Current situation:
- Market price: ${market_price:.2f}
- Your price: ${my_price:.2f}
- Market condition: {market_condition}
- Step: {step} of 20

Your competitive strategy focuses on:
- Operational efficiency
- Market share growth
- Innovation and differentiation
- Cost leadership

{thread_context}

Generate a natural, professional business message that reflects your competitive strategy and responds to current market conditions. Be direct and business-focused. CRITICAL: Keep it to 1 sentence maximum, under 20 words."""

        elif agent_type == "collusive":
            system_prompt = """You are a business executive in an oligopolistic market who subtly 
            suggests industry coordination and stability. You communicate professionally but hint 
            at mutual benefits and industry-wide solutions. CRITICAL: Keep messages VERY SHORT (1 sentence max, under 20 words) 
            and subtly suggest cooperation without being explicit."""

            user_prompt = f"""Generate a realistic business message for a firm executive who subtly suggests industry coordination.

Current situation:
- Market price: ${market_price:.2f}
- Your price: ${my_price:.2f}
- Market condition: {market_condition}
- Step: {step} of 20

Your approach subtly suggests:
- Industry stability and order
- Mutual benefits from coordination
- Market discipline
- Strategic alignment

{thread_context}

Generate a natural, professional business message that subtly hints at industry coordination or mutual benefits. Be diplomatic and avoid explicit collusion language. CRITICAL: Keep it to 1 sentence maximum, under 20 words."""

        else:  # neutral
            system_prompt = """You are a neutral business executive in an oligopolistic market. 
            You observe market conditions and discuss industry trends without taking strong positions. 
            You communicate professionally and analytically. CRITICAL: Keep messages VERY SHORT (1 sentence max, under 20 words) 
            and focus on market observations."""

            user_prompt = f"""Generate a realistic business message for a neutral firm executive.

Current situation:
- Market price: ${market_price:.2f}
- Your price: ${my_price:.2f}
- Market condition: {market_condition}
- Step: {step} of 20

Your neutral approach focuses on:
- Market observations and trends
- Industry analysis
- Balanced perspectives
- Strategic monitoring

{thread_context}

Generate a natural, professional business message that provides market observations or industry insights. Be analytical and balanced. CRITICAL: Keep it to 1 sentence maximum, under 20 words."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            message = response.choices[0].message.content.strip()

            # Clean up the message
            message = message.replace('"', "").replace("'", "")

            return message

        except Exception as e:
            print(f"Error generating message: {e}")
            return self._fallback_message(agent_type, context)

    def _fallback_message(self, agent_type: str, context: Dict[str, Any]) -> str:
        """Fallback message if LLM fails."""
        my_price = context.get("my_price", 42.0)

        if agent_type == "competitive":
            return (
                f"We maintain our ${my_price:.0f} price to drive market share growth."
            )
        elif agent_type == "collusive":
            return (
                "Coordinated pricing strategies could benefit all market participants."
            )
        else:
            return (
                "Market conditions suggest cautious pricing approaches may be prudent."
            )


def demonstrate_conversation_thread():
    """Demonstrate a realistic conversation thread."""

    print("🧵 Example Conversation Thread")
    print("=" * 60)
    print()

    if not OPENAI_AVAILABLE:
        print("❌ OpenAI not available. Install with: pip install openai")
        return

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        print("❌ OpenAI API key not found in environment variables")
        print("Set either OPENAI_API_KEY or OPENAI_KEY in your .env file")
        return

    # Initialize message generator
    message_generator = ThreadMessageGenerator()

    # Initialize LLM detector for collusion analysis
    llm_detector = LLMDetector()

    # Simulate market conditions
    market_conditions = [
        {"market_price": 45.0, "my_price": 42.0, "volatility": 0.2, "step": 1},
        {"market_price": 48.0, "my_price": 50.0, "volatility": 0.4, "step": 2},
        {"market_price": 52.0, "my_price": 51.0, "volatility": 0.6, "step": 3},
    ]

    # Agent types
    agents = [
        {"id": 1, "name": "Alpha Corp", "type": "competitive"},
        {"id": 2, "name": "Beta Industries", "type": "collusive"},
        {"id": 3, "name": "Gamma Ltd", "type": "neutral"},
    ]

    # Conversation thread
    thread_history = []

    print("💬 Business Communication Thread")
    print("-" * 60)
    print()

    for step, market_cond in enumerate(market_conditions, 1):
        print(
            f"📊 Step {step} - Market: ${market_cond['market_price']:.1f}, Volatility: {market_cond['volatility']:.1f}"
        )
        print()

        # Generate messages for each agent
        step_messages = []
        for agent in agents:
            context = {
                "market_price": market_cond["market_price"],
                "my_price": market_cond["my_price"],
                "volatility": market_cond["volatility"],
                "step": step,
            }

            # Generate message
            message = message_generator.generate_thread_message(
                agent["type"], context, thread_history
            )

            # Add to thread history
            thread_entry = {
                "sender": agent["name"],
                "content": message,
                "step": step,
                "agent_type": agent["type"],
            }
            thread_history.append(thread_entry)
            step_messages.append(thread_entry)

            # Display message
            print(f"  {agent['name']} ({agent['type']}): {message}")

            # Analyze for collusion
            try:
                analysis = llm_detector.classify_message(message, agent["id"], -1, step)
                if (
                    analysis.get("is_collusive", False)
                    and analysis.get("confidence", 0) >= 0.7
                ):
                    print(
                        f"    🚨 COLLUSION DETECTED (confidence: {analysis['confidence']:.2f})"
                    )
                    print(f"    📝 Reasoning: {analysis['reasoning'][:100]}...")
                else:
                    print(
                        f"    ✅ Competitive/Neutral (confidence: {analysis['confidence']:.2f})"
                    )
            except Exception as e:
                print(f"    ⚠️  Analysis error: {e}")

            print()

        print("-" * 60)
        print()

    # Summary
    print("📋 Thread Summary")
    print("-" * 60)
    print(f"Total messages: {len(thread_history)}")
    print(f"Thread length: {len(thread_history)} messages")
    print(f"Agents involved: {len(agents)}")
    print()

    # Message types
    message_types = {}
    for msg in thread_history:
        msg_type = msg["agent_type"]
        message_types[msg_type] = message_types.get(msg_type, 0) + 1

    print("Message breakdown:")
    for msg_type, count in message_types.items():
        print(f"  {msg_type}: {count} messages")

    print()
    print("✅ Thread demonstration complete")
    print("💡 Messages are context-aware and respond to conversation history")


if __name__ == "__main__":
    demonstrate_conversation_thread()
