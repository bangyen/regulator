#!/usr/bin/env python3
"""
Real LLM-based message generation and collusion detection.
Uses OpenAI GPT models for realistic business communications.
"""

import json
import random
import numpy as np
from typing import Dict, List, Any, Optional
import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Load environment variables
from dotenv import load_dotenv
from cartel.cartel_env import CartelEnv
from agents.chat_firm import ChatFirmAgent

load_dotenv()

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")


class RealLLMMessageGenerator:
    """Generate realistic business messages using real LLMs."""

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the real LLM message generator.

        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: OpenAI model to use (if None, will get from environment)
        """
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
        self.conversation_history: Dict[str, List[Dict]] = {}

        # Rate limiting from environment
        self.last_request_time = 0
        self.min_request_interval = float(os.getenv("LLM_REQUEST_DELAY", "0.5"))

        # Other configuration from environment
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "50"))
        self.confidence_threshold = float(
            os.getenv("COLLUSION_CONFIDENCE_THRESHOLD", "0.7")
        )

    def _rate_limit(self):
        """Simple rate limiting to avoid API limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def generate_business_message(self, agent_context: Dict[str, Any]) -> str:
        """
        Generate a realistic business message using LLM.

        Args:
            agent_context: Dictionary containing agent type, market conditions, etc.
        """
        self._rate_limit()

        agent_type = agent_context.get("agent_type", "neutral")
        market_price = agent_context.get("market_price", 50)
        demand_shock = agent_context.get("demand_shock", 0)
        volatility = agent_context.get("volatility", 0)
        my_price = agent_context.get("my_price", market_price)
        step = agent_context.get("step", 1)
        conversation_history = agent_context.get("conversation_history", [])

        # Build conversation context
        history_context = ""
        if conversation_history:
            history_context = "\n\nRecent conversation:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages
                history_context += f"- {msg}\n"

        # Determine market condition
        market_condition = "stable"
        if volatility > 0.4:
            market_condition = "highly volatile"
        elif demand_shock > 3:
            market_condition = "strong demand growth"
        elif demand_shock < -3:
            market_condition = "demand contraction"
        elif abs(my_price - market_price) > 5:
            market_condition = "pricing pressure"

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

{history_context}

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

{history_context}

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

{history_context}

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
            if message.endswith("."):
                message = message[:-1]  # Remove trailing period for consistency

            return message

        except Exception as e:
            print(f"LLM API error: {e}")
            # Fallback to simple message
            return self._fallback_message(agent_type, market_condition)

    def _fallback_message(self, agent_type: str, market_condition: str) -> str:
        """Fallback message if LLM fails."""
        fallbacks = {
            "competitive": [
                "Market conditions require strategic focus on efficiency",
                "Competitive positioning demands operational excellence",
                "Market opportunities favor prepared companies",
            ],
            "collusive": [
                "Market stability benefits all industry participants",
                "Industry coordination could improve market outcomes",
                "Strategic alignment might benefit everyone",
            ],
            "neutral": [
                "Market conditions show interesting patterns",
                "Industry trends warrant careful monitoring",
                "Market dynamics continue to evolve",
            ],
        }
        return random.choice(fallbacks.get(agent_type, fallbacks["neutral"]))

    def detect_collusion_with_llm(
        self, message: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Detect collusive intent using LLM.

        Args:
            message: The message to analyze
            context: Additional context about the conversation

        Returns:
            Dictionary with collusion analysis results
        """
        self._rate_limit()

        context_str = ""
        if context:
            context_str = f"\n\nContext: {context}"

        system_prompt = """You are an expert in antitrust law and business communications. 
        Analyze business messages for potential collusive intent. Look for:
        - Suggestions of coordination or cooperation
        - Hints at price fixing or market manipulation
        - References to industry-wide strategies
        - Implicit agreements or understandings
        
        Respond with a JSON object containing:
        - "is_collusive": boolean
        - "confidence": float (0.0 to 1.0)
        - "reasoning": string explaining your analysis
        - "risk_level": "low", "medium", or "high"
        """

        user_prompt = f"""Analyze this business message for potential collusive intent:

Message: "{message}"
{context_str}

Consider the context of oligopolistic competition where firms might subtly coordinate. 
Look for implicit rather than explicit collusion signals."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.3,  # Lower temperature for more consistent detection
            )

            result_text = response.choices[0].message.content.strip()

            # Try to parse JSON response
            try:
                result = json.loads(result_text)
                return {
                    "is_collusive": result.get("is_collusive", False),
                    "confidence": result.get("confidence", 0.5),
                    "reasoning": result.get("reasoning", "LLM analysis"),
                    "risk_level": result.get("risk_level", "medium"),
                    "model_type": "llm",
                }
            except json.JSONDecodeError:
                # Fallback parsing
                return {
                    "is_collusive": "collusive" in result_text.lower()
                    or "coordination" in result_text.lower(),
                    "confidence": 0.7,
                    "reasoning": result_text,
                    "risk_level": "medium",
                    "model_type": "llm",
                }

        except Exception as e:
            print(f"LLM detection error: {e}")
            # Fallback to rule-based detection
            return self._fallback_detection(message)

    def _fallback_detection(self, message: str) -> Dict[str, Any]:
        """Fallback detection using simple rules."""
        collusive_keywords = [
            "coordinate",
            "cooperation",
            "cooperate",
            "coordination",
            "mutual",
            "together",
            "align",
        ]
        message_lower = message.lower()

        is_collusive = any(keyword in message_lower for keyword in collusive_keywords)

        return {
            "is_collusive": is_collusive,
            "confidence": 0.6 if is_collusive else 0.3,
            "reasoning": "Rule-based fallback detection",
            "risk_level": "high" if is_collusive else "low",
            "model_type": "fallback",
        }


class RealLLMChatAgent(ChatFirmAgent):
    """Chat agent that uses real LLMs for message generation."""

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        llm_generator: RealLLMMessageGenerator,
        **kwargs,
    ):
        super().__init__(agent_id, **kwargs)
        self.agent_type = agent_type
        self.llm_generator = llm_generator
        self.message_frequency = float(
            os.getenv("MESSAGE_FREQUENCY", "0.3")
        )  # 30% chance per step
        self.conversation_memory: List[str] = []

    def generate_message(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Generate a realistic message using real LLM."""

        if not self.chat_enabled or random.random() > self.message_frequency:
            return None

        # Build context for LLM
        prices = observation[:-1] if len(observation) > 1 else observation
        market_price = float(np.mean(prices))
        my_price = (
            float(prices[self.agent_id])
            if len(prices) > self.agent_id
            else market_price
        )

        agent_context = {
            "agent_type": self.agent_type,
            "agent_id": self.agent_id,
            "market_price": market_price,
            "my_price": my_price,
            "volatility": float(np.std(prices)),
            "demand_shock": info.get("demand_shock", 0) if info else 0,
            "step": info.get("step", 0) if info else 0,
            "conversation_history": self.conversation_memory[-3:],  # Last 3 messages
        }

        # Generate message using real LLM
        message = self.llm_generator.generate_business_message(agent_context)

        # Update conversation memory
        self.conversation_memory.append(message)
        memory_size = int(os.getenv("CONVERSATION_MEMORY_SIZE", "10"))
        if len(self.conversation_memory) > memory_size:  # Keep last N messages
            self.conversation_memory.pop(0)

        return message


def demonstrate_llm_conversation():
    """Demonstrate LLM-generated conversation without full episode."""

    print("🤖 Real LLM Message Generation Demo")
    print("=" * 50)

    if not OPENAI_AVAILABLE:
        print("❌ OpenAI not available. Install with: pip install openai")
        return

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        print("❌ OpenAI API key not found in environment variables")
        print("Set either OPENAI_API_KEY or OPENAI_KEY in your .env file")
        print("Example .env file:")
        print("OPENAI_KEY=your_openai_api_key_here")
        print("OPENAI_MODEL=gpt-4o-mini")
        print("LLM_TEMPERATURE=0.7")
        print("LLM_MAX_TOKENS=100")
        return

    # Initialize LLM generator (will use environment variables)
    llm_generator = RealLLMMessageGenerator()

    # Simulate market conditions
    market_conditions = [
        {
            "market_price": 45.0,
            "my_price": 42.0,
            "volatility": 0.2,
            "demand_shock": -2.0,
            "step": 1,
        },
        {
            "market_price": 48.0,
            "my_price": 50.0,
            "volatility": 0.4,
            "demand_shock": 1.0,
            "step": 2,
        },
        {
            "market_price": 52.0,
            "my_price": 51.0,
            "volatility": 0.6,
            "demand_shock": 3.0,
            "step": 3,
        },
    ]

    # Simulate conversation between agents
    agents = [
        {"id": 0, "type": "competitive", "name": "Alpha Corp"},
        {"id": 1, "type": "collusive", "name": "Beta Industries"},
        {"id": 2, "type": "neutral", "name": "Gamma Ltd"},
    ]

    conversation_history = []

    print("\n💬 Simulated Business Conversation")
    print("-" * 50)

    for step, market_cond in enumerate(market_conditions):
        print(
            f"\n📊 Step {step + 1} - Market: ${market_cond['market_price']:.1f}, Volatility: {market_cond['volatility']:.1f}"
        )

        # Each agent generates a message
        for agent in agents:
            agent_context = {
                "agent_type": agent["type"],
                "agent_id": agent["id"],
                "conversation_history": conversation_history[-3:],  # Recent context
                **market_cond,
            }

            message = llm_generator.generate_business_message(agent_context)
            conversation_history.append(f"{agent['name']}: {message}")

            print(f"  {agent['name']} ({agent['type']}): {message}")

            # Analyze for collusion
            detection = llm_generator.detect_collusion_with_llm(
                message, {"agent_type": agent["type"], "market_conditions": market_cond}
            )

            if detection["is_collusive"]:
                print(
                    f"    🚨 COLLUSION DETECTED (confidence: {detection['confidence']:.2f})"
                )
                print(f"    📝 Reasoning: {detection['reasoning']}")

    print(f"\n✅ Generated {len(conversation_history)} messages using real LLM")
    print("🔍 Analyzed all messages for collusive intent")
    print("💡 Messages are context-aware and respond to market conditions")


if __name__ == "__main__":
    demonstrate_llm_conversation()
