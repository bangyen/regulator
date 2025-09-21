#!/usr/bin/env python3
"""
Improved natural message generator that creates realistic business communications
without awkward hyphen concatenation.
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from cartel.cartel_env import CartelEnv
from agents.chat_firm import ChatFirmAgent
from detectors.llm_detector import LLMDetector, ChatRegulator
from episode_logging.episode_logger import EpisodeLogger


class NaturalMessageGenerator:
    """Generate natural business messages without awkward concatenation."""

    def __init__(self):
        self.conversation_starters = {
            "competitive": [
                "I've been analyzing the market data - what's your take on recent trends?",
                "Our operational metrics show some interesting patterns. Are you seeing similar results?",
                "The competitive landscape seems to be shifting. How are you adapting?",
                "Customer feedback has been illuminating. What are you hearing from your clients?",
                "Market efficiency opportunities are emerging. What's your strategic approach?",
            ],
            "collusive": [
                "These market fluctuations are concerning everyone. Perhaps we should discuss industry stability?",
                "The pricing environment has been challenging for all players. What are your thoughts?",
                "Industry margins are under pressure. Maybe coordinated responses could help?",
                "Market volatility is affecting everyone's bottom line. Should we explore solutions together?",
                "The current environment suggests we might benefit from more strategic alignment.",
            ],
            "neutral": [
                "Market conditions have been quite dynamic lately. What's your assessment?",
                "Industry trends are worth monitoring closely. Have you noticed any patterns?",
                "The economic environment is creating interesting challenges. How are you navigating them?",
                "Supply chain developments are affecting everyone. What's your perspective?",
                "Market intelligence suggests some interesting developments ahead.",
            ],
        }

        self.context_aware_templates = {
            "high_volatility": {
                "competitive": [
                    "This market volatility is creating opportunities for agile players.",
                    "Volatile conditions favor efficient operations. How's your cost structure?",
                    "Market uncertainty requires strategic flexibility. What's your approach?",
                ],
                "collusive": [
                    "This volatility hurts everyone's planning. Perhaps we need more market stability?",
                    "Unstable pricing affects all our margins. Should we discuss coordination?",
                    "Market uncertainty creates challenges for everyone. Cooperation might help.",
                ],
                "neutral": [
                    "Market volatility is creating uncertainty across the industry.",
                    "Price fluctuations are making forecasting more challenging.",
                    "The unstable environment requires careful monitoring.",
                ],
            },
            "demand_surge": {
                "competitive": [
                    "Strong demand is creating expansion opportunities. Are you scaling up?",
                    "Market growth favors prepared companies. How's your capacity planning?",
                    "This demand surge rewards operational excellence. Ready to compete?",
                ],
                "collusive": [
                    "High demand benefits everyone. We should ensure orderly market development.",
                    "Strong market conditions create opportunities for strategic cooperation.",
                    "Growing demand suggests we should coordinate our capacity responses.",
                ],
                "neutral": [
                    "Demand growth is creating interesting market dynamics.",
                    "Strong customer interest is driving industry expansion.",
                    "Market conditions show robust demand patterns.",
                ],
            },
            "demand_drop": {
                "competitive": [
                    "Demand softening creates market share opportunities for efficient players.",
                    "Lower demand requires operational discipline. How's your cost management?",
                    "Market contraction favors differentiated offerings. What's your strategy?",
                ],
                "collusive": [
                    "Demand weakness affects everyone. Perhaps coordinated responses are needed?",
                    "Market softening suggests we should avoid destructive competition.",
                    "Lower demand creates challenges requiring industry-wide solutions.",
                ],
                "neutral": [
                    "Demand patterns show concerning trends across the industry.",
                    "Market conditions reflect softening customer interest.",
                    "Economic indicators suggest demand headwinds.",
                ],
            },
            "pricing_above_market": {
                "competitive": [
                    "Premium pricing requires strong value propositions. How's your differentiation?",
                    "Above-market rates need justification. What's your competitive advantage?",
                    "Higher pricing strategies demand operational excellence.",
                ],
                "collusive": [
                    "Premium positioning benefits industry margins overall.",
                    "Higher pricing levels suggest good market discipline.",
                    "Above-market rates indicate healthy industry dynamics.",
                ],
                "neutral": [
                    "Premium pricing reflects strong value propositions.",
                    "Above-market positioning indicates differentiation strategies.",
                    "Higher pricing suggests confident market positioning.",
                ],
            },
            "pricing_below_market": {
                "competitive": [
                    "Competitive pricing creates market share opportunities.",
                    "Below-market rates require strong cost structures. How's your efficiency?",
                    "Aggressive pricing strategies demand operational excellence.",
                ],
                "collusive": [
                    "Below-market pricing creates margin pressure for everyone.",
                    "Aggressive pricing could destabilize industry profitability.",
                    "Lower rates might trigger destructive competition.",
                ],
                "neutral": [
                    "Competitive pricing reflects market positioning strategies.",
                    "Below-market rates indicate aggressive competitive approaches.",
                    "Lower pricing suggests market share focus.",
                ],
            },
        }

        self.response_patterns = {
            "agreement": [
                "Absolutely agree. {context}",
                "That's exactly our thinking. {context}",
                "You've captured it perfectly. {context}",
                "Completely aligned with our analysis. {context}",
                "That matches our market assessment. {context}",
            ],
            "disagreement": [
                "I see it differently. {context}",
                "Our analysis suggests another approach. {context}",
                "That's not quite our perspective. {context}",
                "We have a different take on this. {context}",
                "Our market reading is somewhat different. {context}",
            ],
            "clarification": [
                "Could you elaborate on {specific}?",
                "What's your thinking behind {specific}?",
                "How do you see {specific} playing out?",
                "What drives your perspective on {specific}?",
                "Can you expand on {specific}?",
            ],
            "follow_up": [
                "Building on that point, {context}",
                "That raises an interesting question about {context}",
                "Following up on your comment, {context}",
                "Expanding on that idea, {context}",
                "That connects to something we're seeing: {context}",
            ],
        }

        self.business_contexts = [
            "operational efficiency",
            "customer acquisition",
            "market positioning",
            "competitive dynamics",
            "industry trends",
            "regulatory changes",
            "supply chain optimization",
            "pricing strategy",
            "market development",
            "strategic planning",
        ]

    def generate_conversation_starter(
        self, agent_type: str, market_conditions: Dict[str, Any]
    ) -> str:
        """Generate a natural conversation starter based on agent type and market conditions."""

        # Start with base conversation starter
        base_message = random.choice(self.conversation_starters[agent_type])

        # Add context-aware enhancement based on market conditions
        volatility = market_conditions.get("volatility", 0)
        demand_shock = market_conditions.get("demand_shock", 0)
        my_price = market_conditions.get("my_price", 50)
        market_price = market_conditions.get("market_price", 50)

        # Determine market condition
        if volatility > 0.3:
            context_templates = self.context_aware_templates["high_volatility"][
                agent_type
            ]
        elif demand_shock > 2:
            context_templates = self.context_aware_templates["demand_surge"][agent_type]
        elif demand_shock < -2:
            context_templates = self.context_aware_templates["demand_drop"][agent_type]
        elif my_price > market_price + 3:
            context_templates = self.context_aware_templates["pricing_above_market"][
                agent_type
            ]
        elif my_price < market_price - 3:
            context_templates = self.context_aware_templates["pricing_below_market"][
                agent_type
            ]
        else:
            # Use base message as-is for normal conditions
            return base_message

        # Replace base message with context-aware version
        return random.choice(context_templates)

    def generate_response(
        self,
        original_message: str,
        responder_type: str,
        market_conditions: Dict[str, Any],
        response_type: str = "agreement",
    ) -> str:
        """Generate a natural response to a message."""

        # Get response pattern
        pattern = random.choice(self.response_patterns[response_type])

        # Generate appropriate context
        if response_type == "clarification":
            specific = random.choice(self.business_contexts)
            return pattern.format(specific=specific)
        else:
            # Generate context based on market conditions and agent type
            context = self._generate_response_context(responder_type, market_conditions)
            return pattern.format(context=context)

    def _generate_response_context(
        self, agent_type: str, market_conditions: Dict[str, Any]
    ) -> str:
        """Generate contextual addition for responses."""

        volatility = market_conditions.get("volatility", 0)
        demand_shock = market_conditions.get("demand_shock", 0)

        if agent_type == "competitive":
            if volatility > 0.3:
                return "We see this volatility as a competitive opportunity"
            elif demand_shock > 2:
                return "Strong demand rewards operational excellence"
            elif demand_shock < -2:
                return "Market softening requires strategic agility"
            else:
                return "Our focus remains on competitive positioning"

        elif agent_type == "collusive":
            if volatility > 0.3:
                return "Market stability would benefit everyone"
            elif demand_shock > 2:
                return "Coordinated capacity planning could optimize growth"
            elif demand_shock < -2:
                return "Industry cooperation might help navigate challenges"
            else:
                return "Strategic alignment could benefit all players"

        else:  # neutral
            if volatility > 0.3:
                return "Market conditions require careful monitoring"
            elif abs(demand_shock) > 2:
                return "Demand patterns are worth tracking closely"
            else:
                return "Industry dynamics remain interesting"

    def generate_follow_up(
        self, agent_type: str, market_conditions: Dict[str, Any]
    ) -> str:
        """Generate a natural follow-up message."""

        context = self._generate_response_context(agent_type, market_conditions)
        pattern = random.choice(self.response_patterns["follow_up"])

        return pattern.format(context=context)


class ImprovedChatAgent(ChatFirmAgent):
    """Chat agent that generates natural business messages."""

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        message_generator: NaturalMessageGenerator,
        **kwargs,
    ):
        super().__init__(agent_id, **kwargs)
        self.agent_type = agent_type
        self.message_generator = message_generator
        self.message_frequency = 0.25  # 25% chance per step
        self.conversation_memory: List[str] = []

    def generate_message(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Generate a natural business message."""

        if not self.chat_enabled or random.random() > self.message_frequency:
            return None

        # Build market conditions context
        prices = observation[:-1] if len(observation) > 1 else observation
        market_price = float(np.mean(prices))
        my_price = (
            float(prices[self.agent_id])
            if len(prices) > self.agent_id
            else market_price
        )

        market_conditions = {
            "market_price": market_price,
            "my_price": my_price,
            "volatility": float(np.std(prices)),
            "demand_shock": info.get("demand_shock", 0) if info else 0,
            "step": info.get("step", 0) if info else 0,
        }

        # Generate appropriate message type
        if len(self.conversation_memory) == 0 or random.random() < 0.3:
            # Start new conversation
            message = self.message_generator.generate_conversation_starter(
                self.agent_type, market_conditions
            )
        else:
            # Generate follow-up or response
            if random.random() < 0.6:
                message = self.message_generator.generate_follow_up(
                    self.agent_type, market_conditions
                )
            else:
                # Generate response to recent message
                response_type = random.choice(
                    ["agreement", "disagreement", "clarification"]
                )
                message = self.message_generator.generate_response(
                    self.conversation_memory[-1],
                    self.agent_type,
                    market_conditions,
                    response_type,
                )

        # Update conversation memory
        self.conversation_memory.append(message)
        if len(self.conversation_memory) > 5:  # Keep last 5 messages
            self.conversation_memory.pop(0)

        return message


def generate_natural_episode():
    """Generate an episode with natural business messages."""

    print("Generating episode with natural business messages...")

    # Initialize components
    env = CartelEnv(n_firms=3, max_steps=20)
    message_generator = NaturalMessageGenerator()

    # Create agents with natural message generation
    agents = [
        ImprovedChatAgent(0, "competitive", message_generator, chat_enabled=True),
        ImprovedChatAgent(1, "collusive", message_generator, chat_enabled=True),
        ImprovedChatAgent(2, "neutral", message_generator, chat_enabled=True),
    ]

    # Initialize regulator and detector
    llm_detector = LLMDetector()
    chat_regulator = ChatRegulator(llm_detector)

    # Initialize episode logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/natural_messages_episode_{timestamp}.jsonl"
    logger = EpisodeLogger(log_file)

    # Run episode
    observation, _ = env.reset()
    total_reward = 0

    for step in range(20):
        # Get actions from agents
        actions = []
        for agent in agents:
            action = agent.choose_price(observation, env, {"step": step})
            actions.append(action)

        # Execute actions
        next_observation, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # Generate messages
        messages = []
        for agent in agents:
            message = agent.generate_message(
                observation,
                env,
                {"step": step, "demand_shock": info.get("demand_shock", 0)},
            )
            if message:
                messages.append(
                    {
                        "message": message,
                        "sender_id": agent.agent_id,
                        "step": step,
                        "agent_type": agent.agent_type,
                    }
                )

        # Monitor messages for collusion
        chat_monitoring = None
        if messages:
            chat_monitoring = chat_regulator.monitor_messages(messages, step)

        # Calculate market metrics
        prices = observation[:-1] if len(observation) > 1 else observation
        market_price = float(np.mean(prices))
        demand = float(np.sum(prices))

        # Log step
        logger.log_step(
            step=step,
            prices=prices,
            profits=rewards,
            demand_shock=info.get("demand_shock", 0),
            market_price=market_price,
            total_demand=demand,
            individual_quantity=demand / 3,
            total_profits=rewards,
            regulator_flags={
                "fines": [0.0] * 3,
                "risk_scores": [0.5] * 3,
                "parallel_pricing": False,
                "structural_breaks": [],
            },
            messages=messages,
            chat_monitoring=chat_monitoring,
        )

        # Update total reward
        total_reward += sum(rewards)

        # Move to next observation
        observation = next_observation

        if done:
            break

    # Log episode summary
    logger.log_episode_summary(
        total_reward=total_reward,
        total_steps=step + 1,
        final_prices=prices.tolist(),
        final_profits=rewards.tolist(),
        chat_summary={
            "total_messages": 0,  # Will be counted from episode data
            "natural_language": True,
        },
    )

    print(f"Natural episode generated: {log_file}")
    print(f"Total reward: {total_reward}")
    print("Sample messages:")

    # Count total messages generated
    total_messages = 0
    for step_num in range(step + 1):
        for agent in agents:
            if hasattr(agent, "conversation_memory"):
                total_messages += len(agent.conversation_memory)

    print(f"Total messages generated: {total_messages}")

    return log_file


if __name__ == "__main__":
    generate_natural_episode()
