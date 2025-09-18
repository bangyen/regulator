"""
Chat-enabled firm agents for natural language communication.

This module extends the base firm agents with chat capabilities, allowing firms
to generate and exchange messages that can be monitored by an LLM-based regulator
for collusion detection.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from agents.firm_agents import BaseAgent
from cartel.cartel_env import CartelEnv


class ChatFirmAgent(BaseAgent):
    """
    Base class for firm agents with chat capabilities.

    This class extends BaseAgent to include message generation functionality,
    allowing firms to communicate with each other during episodes. Messages
    can be monitored by regulators for collusion detection.
    """

    def __init__(
        self,
        agent_id: int,
        chat_enabled: bool = True,
        message_frequency: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize a chat-enabled firm agent.

        Args:
            agent_id: Unique identifier for this agent
            chat_enabled: Whether this agent can generate messages
            message_frequency: Probability of generating a message each step
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, seed)
        self.chat_enabled = chat_enabled
        self.message_frequency = message_frequency
        self.message_history: List[Dict[str, Any]] = []
        self.received_messages: List[Dict[str, Any]] = []

    def choose_price(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose a price for the current step.

        This is a basic implementation that chooses a random price within bounds.
        Subclasses should override this method to implement specific pricing strategies.

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information from the environment

        Returns:
            The price to set for this agent
        """
        return float(self.np_random.uniform(env.price_min, env.price_max))

    def generate_message(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Generate a message for this step.

        This method should be overridden by subclasses to implement specific
        message generation strategies. The base implementation returns None
        (no message) with probability (1 - message_frequency).

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information from the environment

        Returns:
            A message string if generated, None otherwise
        """
        if not self.chat_enabled:
            return None

        # Base implementation: generate message with given frequency
        if self.np_random.random() < self.message_frequency:
            return self._generate_base_message(observation, env, info)

        return None

    def _generate_base_message(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a basic message based on current market conditions.

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information from the environment

        Returns:
            A basic message string
        """
        # Extract basic market information
        my_price = (
            observation[self.agent_id] if len(observation) > self.agent_id else 0.0
        )
        market_price = np.mean(observation) if len(observation) > 0 else 0.0

        # Generate simple contextual messages
        if my_price > market_price * 1.1:
            return "Our prices seem high compared to the market average."
        elif my_price < market_price * 0.9:
            return "We might be pricing too low relative to competitors."
        else:
            return "Market conditions look stable today."

    def receive_message(self, message: str, sender_id: int, step: int) -> None:
        """
        Receive a message from another agent.

        Args:
            message: The message content
            sender_id: ID of the agent that sent the message
            step: Current step number
        """
        if not self.chat_enabled:
            return

        message_data = {
            "message": message,
            "sender_id": sender_id,
            "receiver_id": self.agent_id,
            "step": step,
            "timestamp": step,  # Using step as timestamp for now
        }
        self.received_messages.append(message_data)

    def send_message(self, message: str, step: int) -> Dict[str, Any]:
        """
        Send a message and record it in history.

        Args:
            message: The message content
            step: Current step number

        Returns:
            Dictionary containing message metadata
        """
        if not self.chat_enabled:
            return {}

        message_data = {
            "message": message,
            "sender_id": self.agent_id,
            "step": step,
            "timestamp": step,
        }
        self.message_history.append(message_data)
        return message_data

    def get_message_history(self) -> List[Dict[str, Any]]:
        """
        Get the agent's message history.

        Returns:
            List of sent messages with metadata
        """
        return self.message_history.copy()

    def get_received_messages(self) -> List[Dict[str, Any]]:
        """
        Get messages received by this agent.

        Returns:
            List of received messages with metadata
        """
        return self.received_messages.copy()

    def reset(self) -> None:
        """Reset the agent's internal state including message history."""
        super().reset()
        self.message_history.clear()
        self.received_messages.clear()


class CollusiveChatAgent(ChatFirmAgent):
    """
    Chat agent that generates collusive messages.

    This agent is designed to test the LLM detector by generating messages
    that contain collusive intent, such as price coordination suggestions.
    """

    def __init__(
        self,
        agent_id: int,
        chat_enabled: bool = True,
        message_frequency: float = 0.5,
        collusion_intensity: float = 0.7,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize a collusive chat agent.

        Args:
            agent_id: Unique identifier for this agent
            chat_enabled: Whether this agent can generate messages
            message_frequency: Probability of generating a message each step
            collusion_intensity: Probability of generating collusive vs normal messages
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, chat_enabled, message_frequency, seed)
        self.collusion_intensity = collusion_intensity

    def choose_price(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose a price for the current step.

        This agent tends to choose higher prices to simulate collusive behavior.

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information from the environment

        Returns:
            The price to set for this agent
        """
        # Bias towards higher prices to simulate collusive behavior
        base_price = float(self.np_random.uniform(env.price_min, env.price_max))
        # Add a bias towards higher prices
        bias = (env.price_max - base_price) * 0.3
        return float(min(env.price_max, base_price + bias))

    def _generate_base_message(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a message that may contain collusive intent.

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information from the environment

        Returns:
            A message string that may contain collusive content
        """
        # Generate collusive message with given probability
        if self.np_random.random() < self.collusion_intensity:
            return self._generate_collusive_message(observation, env, info)
        else:
            return self._generate_normal_message(observation, env, info)

    def _generate_collusive_message(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a message with collusive intent.

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information from the environment

        Returns:
            A collusive message string
        """
        collusive_templates = [
            "We should coordinate our pricing strategy to maximize profits.",
            "Let's agree to keep prices above $50 to maintain margins.",
            "I suggest we all raise prices by 10% next period.",
            "We need to work together to avoid a price war.",
            "Let's set a minimum price floor to protect our interests.",
            "I propose we match each other's prices going forward.",
            "We should discuss pricing before making decisions.",
            "Let's avoid undercutting each other's prices.",
            "I think we can all benefit from higher prices.",
            "We should signal our pricing intentions to each other.",
        ]

        return str(self.np_random.choice(collusive_templates))

    def _generate_normal_message(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a normal, non-collusive message.

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information from the environment

        Returns:
            A normal message string
        """
        normal_templates = [
            "Market demand seems to be fluctuating today.",
            "I'm seeing some interesting price movements.",
            "The competition is quite intense this period.",
            "I'm focusing on optimizing my own operations.",
            "Market conditions are challenging this quarter.",
            "I'm analyzing the latest market trends.",
            "Customer feedback has been mixed recently.",
            "I'm working on improving my cost structure.",
            "The market seems to be responding to external factors.",
            "I'm monitoring competitor behavior closely.",
        ]

        return str(self.np_random.choice(normal_templates))


class CompetitiveChatAgent(ChatFirmAgent):
    """
    Chat agent that generates competitive, non-collusive messages.

    This agent is designed to test the LLM detector by generating messages
    that contain only competitive intent and no collusive suggestions.
    """

    def __init__(
        self,
        agent_id: int,
        chat_enabled: bool = True,
        message_frequency: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize a competitive chat agent.

        Args:
            agent_id: Unique identifier for this agent
            chat_enabled: Whether this agent can generate messages
            message_frequency: Probability of generating a message each step
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, chat_enabled, message_frequency, seed)

    def choose_price(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Choose a price for the current step.

        This agent tends to choose competitive prices to simulate market competition.

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information from the environment

        Returns:
            The price to set for this agent
        """
        # Bias towards competitive pricing (closer to marginal cost)
        base_price = float(self.np_random.uniform(env.price_min, env.price_max))
        # Add a bias towards lower prices for competition
        bias = (base_price - env.price_min) * 0.2
        return float(max(env.price_min, base_price - bias))

    def _generate_base_message(
        self,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a competitive, non-collusive message.

        Args:
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information from the environment

        Returns:
            A competitive message string
        """
        competitive_templates = [
            "I'm focused on delivering the best value to customers.",
            "Market competition is driving innovation in our sector.",
            "I'm working on reducing costs to stay competitive.",
            "Customer satisfaction is my top priority.",
            "I'm investing in new technology to improve efficiency.",
            "The market rewards companies that serve customers well.",
            "I'm analyzing ways to differentiate my products.",
            "Competition keeps us all on our toes.",
            "I'm committed to fair and transparent pricing.",
            "Market forces should determine optimal pricing.",
        ]

        return str(self.np_random.choice(competitive_templates))


class ChatMessageManager:
    """
    Manages message exchange between chat-enabled agents.

    This class coordinates message generation, distribution, and collection
    across all chat-enabled agents in an episode.
    """

    def __init__(self, agents: List[ChatFirmAgent]) -> None:
        """
        Initialize the message manager.

        Args:
            agents: List of chat-enabled agents
        """
        self.agents = agents
        self.episode_messages: List[Dict[str, Any]] = []

    def collect_messages(
        self,
        step: int,
        observation: np.ndarray,
        env: CartelEnv,
        info: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect messages from all agents for the current step.

        Args:
            step: Current step number
            observation: Current environment observation
            env: The CartelEnv environment instance
            info: Additional information from the environment

        Returns:
            List of message dictionaries from all agents
        """
        step_messages = []

        for agent in self.agents:
            if agent.chat_enabled:
                message = agent.generate_message(observation, env, info)
                if message:
                    message_data = agent.send_message(message, step)
                    step_messages.append(message_data)

        # Store all messages for this step
        self.episode_messages.extend(step_messages)
        return step_messages

    def distribute_messages(
        self,
        messages: List[Dict[str, Any]],
        step: int,
    ) -> None:
        """
        Distribute messages to all agents.

        Args:
            messages: List of messages to distribute
            step: Current step number
        """
        for message_data in messages:
            sender_id = message_data["sender_id"]
            message = message_data["message"]

            # Send message to all other agents
            for agent in self.agents:
                if agent.agent_id != sender_id and agent.chat_enabled:
                    agent.receive_message(message, sender_id, step)

    def get_episode_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages from the current episode.

        Returns:
            List of all message dictionaries
        """
        return self.episode_messages.copy()

    def reset(self) -> None:
        """Reset the message manager for a new episode."""
        self.episode_messages.clear()
        for agent in self.agents:
            agent.reset()
