#!/usr/bin/env python3
"""
Enhanced realistic message exchange system with:
- Conversation threads and responses
- Industry-specific terminology
- Realistic timing patterns
- Message threading and context
- Multi-turn conversations
- Industry events and news
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from cartel.cartel_env import CartelEnv
from agents.chat_firm import ChatFirmAgent
from detectors.llm_detector import LLMDetector, ChatRegulator
from episode_logging.episode_logger import EpisodeLogger


class MessageType(Enum):
    """Types of messages in a conversation."""

    INITIATION = "initiation"
    RESPONSE = "response"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    THREAT = "threat"
    WARNING = "warning"


class IndustryEvent(Enum):
    """Industry events that can trigger conversations."""

    REGULATORY_CHANGE = "regulatory_change"
    SUPPLY_SHOCK = "supply_shock"
    DEMAND_SHOCK = "demand_shock"
    COMPETITOR_ACTION = "competitor_action"
    MARKET_VOLATILITY = "market_volatility"
    PRICE_WAR = "price_war"
    MERGER_RUMOR = "merger_rumor"
    TECHNOLOGY_DISRUPTION = "technology_disruption"


@dataclass
class Message:
    """Represents a single message in a conversation."""

    sender_id: int
    receiver_id: int
    content: str
    message_type: MessageType
    timestamp: datetime
    step: int
    context: Dict[str, Any]
    thread_id: Optional[str] = None
    is_response: bool = False
    response_to: Optional[str] = None


@dataclass
class ConversationThread:
    """Represents a conversation thread between agents."""

    thread_id: str
    participants: List[int]
    messages: List[Message]
    topic: str
    started_at: datetime
    last_activity: datetime
    is_active: bool = True


class EnhancedMessageGenerator:
    """Generate highly realistic business messages with conversation threading."""

    def __init__(self):
        self.industry_terminology = {
            "competitive": [
                "market share",
                "competitive advantage",
                "innovation",
                "efficiency",
                "customer acquisition",
                "product differentiation",
                "pricing strategy",
                "operational excellence",
                "market penetration",
                "brand positioning",
            ],
            "collusive": [
                "market stability",
                "industry discipline",
                "coordination",
                "mutual benefit",
                "price discipline",
                "market order",
                "industry cooperation",
                "strategic alignment",
                "market equilibrium",
                "collective action",
            ],
            "neutral": [
                "market conditions",
                "industry trends",
                "economic factors",
                "regulatory environment",
                "supply chain",
                "customer behavior",
                "market dynamics",
                "competitive landscape",
                "industry outlook",
                "market intelligence",
            ],
        }

        self.conversation_starters = {
            "competitive": [
                "Have you noticed the recent market shifts?",
                "What's your take on the current pricing environment?",
                "How are you handling the increased competition?",
                "What's your strategy for the next quarter?",
                "Are you seeing the same cost pressures we are?",
            ],
            "collusive": [
                "The market seems quite volatile lately, doesn't it?",
                "These price fluctuations are hurting everyone's margins",
                "Maybe we should discuss how to bring some stability to the market",
                "The industry could benefit from more coordinated responses",
                "What do you think about the current market conditions?",
            ],
            "neutral": [
                "Interesting market developments this week",
                "The industry landscape seems to be changing",
                "What are your thoughts on the current market situation?",
                "Have you observed any notable trends recently?",
                "The market dynamics are quite fascinating right now",
            ],
        }

        self.response_patterns = {
            "agreement": [
                "I completely agree with that assessment",
                "That's exactly what we've been thinking",
                "You've hit the nail on the head",
                "We're seeing the same patterns",
                "That aligns with our market analysis",
            ],
            "disagreement": [
                "I see it differently - we think...",
                "That's not quite how we're reading the market",
                "Our analysis suggests a different approach",
                "We have a different perspective on this",
                "I'm not sure I agree with that assessment",
            ],
            "clarification": [
                "Can you elaborate on that point?",
                "What specifically do you mean by...",
                "I'd like to understand better...",
                "Could you clarify your position on...",
                "What's your thinking behind that strategy?",
            ],
            "follow_up": [
                "Building on what you said...",
                "That raises an interesting point about...",
                "Following up on our previous discussion...",
                "To add to what we discussed...",
                "Expanding on that idea...",
            ],
        }

        self.industry_events = {
            IndustryEvent.REGULATORY_CHANGE: [
                "New regulations are coming that will affect pricing",
                "The regulatory environment is getting more complex",
                "We need to prepare for upcoming regulatory changes",
            ],
            IndustryEvent.SUPPLY_SHOCK: [
                "Supply chain disruptions are affecting costs",
                "Raw material prices are skyrocketing",
                "We're facing significant supply constraints",
            ],
            IndustryEvent.DEMAND_SHOCK: [
                "Customer demand has shifted dramatically",
                "Market demand is more volatile than expected",
                "We're seeing unexpected demand patterns",
            ],
            IndustryEvent.COMPETITOR_ACTION: [
                "Our main competitor just made a major move",
                "There's been significant competitive activity",
                "The competitive landscape is shifting rapidly",
            ],
            IndustryEvent.MARKET_VOLATILITY: [
                "Market volatility is creating uncertainty",
                "Price fluctuations are making planning difficult",
                "The market is more unpredictable than usual",
            ],
        }

        # Conversation state tracking
        self.active_threads: Dict[str, ConversationThread] = {}
        self.agent_conversation_history: Dict[int, List[str]] = defaultdict(list)
        self.industry_events_queue: List[Tuple[IndustryEvent, int, str]] = []

    def generate_industry_event(
        self, step: int, market_conditions: Dict[str, Any]
    ) -> Optional[IndustryEvent]:
        """Generate industry events based on market conditions."""

        # Determine event probability based on market conditions
        volatility = market_conditions.get("volatility", 0)
        demand_shock = abs(market_conditions.get("demand_shock", 0))

        event_probability = 0.1 + (volatility * 0.3) + (demand_shock * 0.02)

        if random.random() < event_probability:
            # Select event type based on conditions
            if volatility > 0.4:
                return IndustryEvent.MARKET_VOLATILITY
            elif demand_shock > 3:
                return IndustryEvent.DEMAND_SHOCK
            elif random.random() < 0.3:
                return IndustryEvent.SUPPLY_SHOCK
            else:
                return random.choice(list(IndustryEvent))

        return None

    def start_conversation_thread(
        self, sender_id: int, receiver_id: int, topic: str, step: int
    ) -> ConversationThread:
        """Start a new conversation thread between two agents."""

        thread_id = (
            f"thread_{sender_id}_{receiver_id}_{step}_{random.randint(1000, 9999)}"
        )

        thread = ConversationThread(
            thread_id=thread_id,
            participants=[sender_id, receiver_id],
            messages=[],
            topic=topic,
            started_at=datetime.now(),
            last_activity=datetime.now(),
        )

        self.active_threads[thread_id] = thread
        return thread

    def generate_conversation_starter(
        self,
        sender_id: int,
        receiver_id: int,
        agent_type: str,
        market_conditions: Dict[str, Any],
        step: int,
    ) -> Optional[Message]:
        """Generate a conversation starter message."""

        # Check if agents should start a conversation
        conversation_probability = 0.15  # 15% chance per step

        # Increase probability based on market conditions
        volatility = market_conditions.get("volatility", 0)
        if volatility > 0.3:
            conversation_probability += 0.1

        if random.random() > conversation_probability:
            return None

        # Select conversation starter
        starter = random.choice(self.conversation_starters[agent_type])

        # Add industry-specific context
        if market_conditions.get("demand_shock", 0) > 2:
            starter += " - especially with the recent demand changes"
        elif market_conditions.get("volatility", 0) > 0.3:
            starter += " - given the current market volatility"

        # Create message
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=starter,
            message_type=MessageType.INITIATION,
            timestamp=datetime.now(),
            step=step,
            context=market_conditions,
        )

        # Start conversation thread
        thread = self.start_conversation_thread(
            sender_id, receiver_id, "market_discussion", step
        )
        message.thread_id = thread.thread_id
        thread.messages.append(message)

        return message

    def generate_response(
        self,
        original_message: Message,
        responder_type: str,
        market_conditions: Dict[str, Any],
    ) -> Optional[Message]:
        """Generate a response to an existing message."""

        # Determine response probability (80% chance to respond)
        if random.random() > 0.8:
            return None

        # Determine response type based on agent type and message content
        response_type = self._determine_response_type(original_message, responder_type)

        # Generate response content
        if response_type == "agreement":
            response_content = random.choice(self.response_patterns["agreement"])
        elif response_type == "disagreement":
            response_content = random.choice(self.response_patterns["disagreement"])
        elif response_type == "clarification":
            response_content = random.choice(self.response_patterns["clarification"])
        else:
            response_content = random.choice(self.response_patterns["follow_up"])

        # Add industry-specific details
        response_content = self._add_industry_context(
            response_content, responder_type, market_conditions
        )

        # Create response message
        response = Message(
            sender_id=original_message.receiver_id,
            receiver_id=original_message.sender_id,
            content=response_content,
            message_type=MessageType.RESPONSE,
            timestamp=datetime.now(),
            step=original_message.step + 1,
            context=market_conditions,
            thread_id=original_message.thread_id,
            is_response=True,
            response_to=original_message.content,
        )

        # Add to thread
        if original_message.thread_id in self.active_threads:
            self.active_threads[original_message.thread_id].messages.append(response)
            self.active_threads[original_message.thread_id].last_activity = (
                datetime.now()
            )

        return response

    def _determine_response_type(
        self, original_message: Message, responder_type: str
    ) -> str:
        """Determine the type of response based on message content and agent type."""

        content = original_message.content.lower()

        # Competitive agents tend to disagree more
        if responder_type == "competitive":
            if "coordinate" in content or "cooperation" in content:
                return "disagreement"
            elif "strategy" in content or "approach" in content:
                return "clarification"
            else:
                return random.choice(["agreement", "follow_up"])

        # Collusive agents tend to agree more
        elif responder_type == "collusive":
            if "stability" in content or "discipline" in content:
                return "agreement"
            elif "volatility" in content or "uncertainty" in content:
                return "agreement"
            else:
                return random.choice(["agreement", "follow_up"])

        # Neutral agents are balanced
        else:
            return random.choice(
                ["agreement", "disagreement", "clarification", "follow_up"]
            )

    def _add_industry_context(
        self, content: str, agent_type: str, market_conditions: Dict[str, Any]
    ) -> str:
        """Add industry-specific context to message content."""

        # Add specific business details
        if agent_type == "competitive":
            if "efficiency" in content.lower():
                content += " - we're focusing on operational improvements"
            elif "strategy" in content.lower():
                content += " - our market positioning is key"

        elif agent_type == "collusive":
            if "stability" in content.lower():
                content += " - industry-wide coordination could help"
            elif "volatility" in content.lower():
                content += " - perhaps we need more market discipline"

        # Add market-specific context
        volatility = market_conditions.get("volatility", 0)
        if volatility > 0.4:
            content += " - the current volatility is concerning"

        demand_shock = market_conditions.get("demand_shock", 0)
        if abs(demand_shock) > 2:
            if demand_shock > 0:
                content += " - the demand surge is creating opportunities"
            else:
                content += " - the demand drop requires careful response"

        return content

    def generate_follow_up_message(
        self,
        thread: ConversationThread,
        agent_type: str,
        market_conditions: Dict[str, Any],
        step: int,
    ) -> Optional[Message]:
        """Generate a follow-up message in an existing conversation."""

        # Check if conversation should continue (30% chance)
        if random.random() > 0.3:
            return None

        # Generate follow-up based on conversation history
        last_message = thread.messages[-1]

        if agent_type == "competitive":
            follow_up = (
                "Building on that, we think the key is maintaining our competitive edge"
            )
        elif agent_type == "collusive":
            follow_up = "That's a good point - maybe we should explore this further"
        else:
            follow_up = "Interesting perspective - what do you think about the broader implications?"

        # Add market context
        follow_up = self._add_industry_context(follow_up, agent_type, market_conditions)

        # Create follow-up message
        message = Message(
            sender_id=last_message.receiver_id,
            receiver_id=last_message.sender_id,
            content=follow_up,
            message_type=MessageType.FOLLOW_UP,
            timestamp=datetime.now(),
            step=step,
            context=market_conditions,
            thread_id=thread.thread_id,
            is_response=True,
            response_to=last_message.content,
        )

        thread.messages.append(message)
        thread.last_activity = datetime.now()

        return message

    def cleanup_old_threads(self, current_step: int):
        """Clean up old conversation threads."""

        threads_to_remove = []
        for thread_id, thread in self.active_threads.items():
            # Remove threads older than 5 steps
            if current_step - thread.messages[0].step > 5:
                threads_to_remove.append(thread_id)

        for thread_id in threads_to_remove:
            del self.active_threads[thread_id]


class EnhancedChatAgent(ChatFirmAgent):
    """Enhanced chat agent with realistic conversation capabilities."""

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        message_generator: EnhancedMessageGenerator,
        **kwargs,
    ):
        super().__init__(agent_id, **kwargs)
        self.agent_type = agent_type
        self.message_generator = message_generator
        self.conversation_partners: List[int] = []
        self.message_history: List[Message] = []
        self.response_pending: List[Message] = []

    def generate_messages(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> List[Message]:
        """Generate realistic messages with conversation threading."""

        messages = []
        step = info.get("step", 0) if info else 0

        # Build market conditions context
        prices = observation[:-1] if len(observation) > 1 else observation
        market_conditions = {
            "market_price": float(np.mean(prices)),
            "volatility": float(np.std(prices)),
            "demand_shock": info.get("demand_shock", 0) if info else 0,
            "step": step,
        }

        # 1. Respond to pending messages
        for pending_message in self.response_pending:
            response = self.message_generator.generate_response(
                pending_message, self.agent_type, market_conditions
            )
            if response:
                messages.append(response)
                self.message_history.append(response)

        self.response_pending.clear()

        # 2. Generate new conversation starters
        for other_agent_id in range(3):  # Assuming 3 agents
            if other_agent_id != self.agent_id:
                starter = self.message_generator.generate_conversation_starter(
                    self.agent_id,
                    other_agent_id,
                    self.agent_type,
                    market_conditions,
                    step,
                )
                if starter:
                    messages.append(starter)
                    self.message_history.append(starter)

        # 3. Generate follow-up messages in active threads
        for thread in self.message_generator.active_threads.values():
            if self.agent_id in thread.participants and thread.is_active:
                follow_up = self.message_generator.generate_follow_up_message(
                    thread, self.agent_type, market_conditions, step
                )
                if follow_up:
                    messages.append(follow_up)
                    self.message_history.append(follow_up)

        # 4. Clean up old threads
        self.message_generator.cleanup_old_threads(step)

        return messages

    def receive_message(self, message: Message):
        """Receive a message from another agent."""
        self.response_pending.append(message)


def generate_enhanced_realistic_episode():
    """Generate an episode with enhanced realistic message exchanges."""

    print("Generating episode with enhanced realistic message exchanges...")

    # Initialize components
    env = CartelEnv(n_firms=3, max_steps=20)
    message_generator = EnhancedMessageGenerator()

    # Create enhanced agents
    agents = [
        EnhancedChatAgent(0, "competitive", message_generator, chat_enabled=True),
        EnhancedChatAgent(1, "collusive", message_generator, chat_enabled=True),
        EnhancedChatAgent(2, "neutral", message_generator, chat_enabled=True),
    ]

    # Initialize regulator and detector
    llm_detector = LLMDetector()
    chat_regulator = ChatRegulator(llm_detector)

    # Initialize episode logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/enhanced_realistic_episode_{timestamp}.jsonl"
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

        # Generate messages with conversation threading
        all_messages = []
        for agent in agents:
            agent_messages = agent.generate_messages(
                observation,
                env,
                {"step": step, "demand_shock": info.get("demand_shock", 0)},
            )
            all_messages.extend(agent_messages)

        # Simulate message delivery and responses
        for message in all_messages:
            if message.receiver_id != -1:  # Not a broadcast message
                # Find the receiving agent
                for agent in agents:
                    if agent.agent_id == message.receiver_id:
                        agent.receive_message(message)
                        break

        # Convert messages to log format
        log_messages = []
        for msg in all_messages:
            log_messages.append(
                {
                    "message": msg.content,
                    "sender_id": msg.sender_id,
                    "receiver_id": msg.receiver_id,
                    "step": msg.step,
                    "agent_type": agents[msg.sender_id].agent_type,
                    "message_type": msg.message_type.value,
                    "thread_id": msg.thread_id,
                    "is_response": msg.is_response,
                    "response_to": msg.response_to,
                }
            )

        # Monitor messages for collusion
        chat_monitoring = None
        if log_messages:
            chat_monitoring = chat_regulator.monitor_messages(log_messages, step)

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
            messages=log_messages,
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
            "total_messages": len(all_messages),
            "conversation_threads": len(message_generator.active_threads),
            "message_types": {},
        },
    )

    print(f"Enhanced episode generated: {log_file}")
    print(f"Total messages: {len(all_messages)}")
    print(f"Active conversation threads: {len(message_generator.active_threads)}")
    print(f"Total reward: {total_reward}")

    return log_file


if __name__ == "__main__":
    generate_enhanced_realistic_episode()
