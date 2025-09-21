#!/usr/bin/env python3
"""
Generate a new episode with LLM-based realistic messages.
This script creates an episode with context-aware, realistic business communications.
"""

import random
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from cartel.cartel_env import CartelEnv
from agents.chat_firm import ChatFirmAgent
from detectors.llm_detector import LLMDetector, ChatRegulator
from episode_logging.episode_logger import EpisodeLogger


class LLMMessageGenerator:
    """Generate realistic business messages using LLM-like logic."""

    def __init__(self):
        self.message_templates = {
            "competitive": [
                "Market opportunities require aggressive positioning - efficiency is key",
                "Customer feedback suggests we need to innovate faster",
                "Raw material costs are up 15% - how are you managing margins?",
                "Market volatility creates opportunities for agile players",
                "Industry margins have been compressed - time to optimize operations",
                "Competition is heating up - we need to differentiate better",
                "Supply chain disruptions are affecting everyone - what's your strategy?",
                "Customer demand is shifting - we need to adapt quickly",
            ],
            "collusive": [
                "Market conditions are challenging - maybe we should all be more strategic",
                "Industry margins have been tight - maybe we need more discipline",
                "Price volatility hurts everyone - perhaps we should coordinate better",
                "Market stability benefits all players - cooperation might help",
                "Raw material costs are rising - industry-wide response needed",
                "Customer price sensitivity is increasing - we should all be careful",
                "Market conditions suggest we need industry-wide discipline",
                "Perhaps we should discuss market conditions more regularly",
            ],
            "neutral": [
                "Market conditions are evolving - interesting times ahead",
                "Industry trends suggest we need to stay informed",
                "Market volatility is creating uncertainty for everyone",
                "Customer behavior patterns are changing - worth monitoring",
                "Supply chain issues are affecting the entire industry",
                "Market dynamics are shifting - adaptation is key",
                "Industry consolidation might be on the horizon",
                "Market conditions require careful monitoring",
            ],
        }

    def generate_message(self, agent_context: Dict[str, Any]) -> str:
        """Generate a realistic business message based on current context."""

        agent_type = agent_context.get("agent_type", "neutral")
        market_price = agent_context.get("market_price", 50)
        demand_shock = agent_context.get("demand_shock", 0)
        volatility = agent_context.get("volatility", 0)
        my_price = agent_context.get("my_price", market_price)
        step = agent_context.get("step", 1)

        # Select base template
        templates = self.message_templates.get(
            agent_type, self.message_templates["neutral"]
        )
        base_message = random.choice(templates)

        # Add context-specific modifications
        if abs(demand_shock) > 3:
            if demand_shock > 0:
                base_message += " - demand surge is creating opportunities"
            else:
                base_message += " - demand drop requires careful response"

        if volatility > 0.3:
            base_message += " - market volatility is concerning"

        if abs(my_price - market_price) > 5:
            if my_price > market_price:
                base_message += " - we're pricing above market"
            else:
                base_message += " - we're pricing below market"

        # Add step-specific context
        if step > 15:
            base_message += " - late in the game, strategy matters"
        elif step < 5:
            base_message += " - early stages, positioning is key"

        return base_message


class LLMChatAgent(ChatFirmAgent):
    """Chat agent that generates realistic messages using LLM-like logic."""

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        llm_generator: LLMMessageGenerator,
        **kwargs,
    ):
        super().__init__(agent_id, **kwargs)
        self.agent_type = agent_type
        self.llm_generator = llm_generator
        self.message_frequency = 0.3  # 30% chance of sending message each step

    def generate_message(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Generate a realistic message using LLM-like logic."""

        if not self.chat_enabled or random.random() > self.message_frequency:
            return None

        # Build context for message generation
        # Observation structure: [previous_prices, demand_shock]
        prices = observation[:-1]  # All prices except the last element (demand shock)
        market_price = float(np.mean(prices))
        my_price = (
            float(prices[self.agent_id])
            if len(prices) > self.agent_id
            else market_price
        )

        context = {
            "agent_type": self.agent_type,
            "agent_id": self.agent_id,
            "market_price": market_price,
            "my_price": my_price,
            "demand_shock": info.get("demand_shock", 0) if info else 0,
            "volatility": self._calculate_volatility(prices),
            "step": info.get("step", 0) if info else 0,
        }

        # Generate message using LLM-like logic
        message = self.llm_generator.generate_message(context)
        return message

    def _calculate_volatility(self, observation: np.ndarray) -> float:
        """Calculate simple volatility measure."""
        if len(observation) < 2:
            return 0.0
        return float(np.std(observation))


def generate_llm_episode():
    """Generate an episode with LLM-based realistic messages."""

    print("Generating episode with LLM-based realistic messages...")

    # Initialize components
    env = CartelEnv(n_firms=3, max_steps=20)
    llm_generator = LLMMessageGenerator()

    # Create agents with different strategies
    agents = [
        LLMChatAgent(0, "competitive", llm_generator, chat_enabled=True),
        LLMChatAgent(1, "collusive", llm_generator, chat_enabled=True),
        LLMChatAgent(2, "neutral", llm_generator, chat_enabled=True),
    ]

    # Initialize regulator and detector
    llm_detector = LLMDetector()
    chat_regulator = ChatRegulator(llm_detector)

    # Initialize episode logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/llm_episode_{timestamp}.jsonl"
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
        prices = observation[:-1]  # All prices except the last element (demand shock)
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
            individual_quantity=demand / 3,  # Simplified
            total_profits=rewards,
            regulator_flags={
                "fines": [0.0] * 3,  # Simplified
                "risk_scores": [0.5] * 3,  # Simplified
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
            "total_messages": 0,  # Will be calculated from logged data
            "collusive_messages": 0,
            "competitive_messages": 0,
            "neutral_messages": 0,
        },
    )

    print(f"Episode generated: {log_file}")
    print(f"Total reward: {total_reward}")
    print(f"Final market price: {float(np.mean(prices))}")
    print(f"Final prices: {prices.tolist()}")
    print(f"Final profits: {rewards.tolist()}")

    return log_file


if __name__ == "__main__":
    generate_llm_episode()
