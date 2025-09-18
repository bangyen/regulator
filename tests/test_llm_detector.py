"""
Tests for LLM-based collusion detector and chat functionality.

This module contains comprehensive tests for the LLMDetector, ChatFirmAgent,
and related chat functionality to ensure proper message classification and
regulator behavior.
"""

import pytest

import numpy as np

from src.agents.chat_firm import (
    ChatFirmAgent,
    CollusiveChatAgent,
    CompetitiveChatAgent,
    ChatMessageManager,
)
from src.detectors.llm_detector import LLMDetector, ChatRegulator
from src.cartel.cartel_env import CartelEnv


class TestChatFirmAgent:
    """Test cases for ChatFirmAgent base class."""

    def test_chat_firm_agent_initialization(self):
        """Test that ChatFirmAgent initializes correctly."""
        agent = ChatFirmAgent(agent_id=0, chat_enabled=True, message_frequency=0.5)

        assert agent.agent_id == 0
        assert agent.chat_enabled is True
        assert agent.message_frequency == 0.5
        assert len(agent.message_history) == 0
        assert len(agent.received_messages) == 0

    def test_chat_firm_agent_disabled(self):
        """Test that disabled chat agent doesn't generate messages."""
        agent = ChatFirmAgent(agent_id=0, chat_enabled=False)

        # Create mock environment
        env = CartelEnv(n_firms=2, price_min=10, price_max=100)
        observation = np.array([50.0, 60.0])

        message = agent.generate_message(observation, env)
        assert message is None

    def test_message_generation_frequency(self):
        """Test that message generation follows the specified frequency."""
        agent = ChatFirmAgent(agent_id=0, message_frequency=1.0, seed=42)

        # Create mock environment
        env = CartelEnv(n_firms=2, price_min=10, price_max=100)
        observation = np.array([50.0, 60.0])

        # With frequency 1.0, should always generate a message
        message = agent.generate_message(observation, env)
        assert message is not None
        assert isinstance(message, str)
        assert len(message) > 0

    def test_message_sending_and_receiving(self):
        """Test message sending and receiving functionality."""
        agent = ChatFirmAgent(agent_id=0)

        # Send a message
        message_data = agent.send_message("Test message", step=1)

        assert message_data["message"] == "Test message"
        assert message_data["sender_id"] == 0
        assert message_data["step"] == 1
        assert len(agent.message_history) == 1

        # Receive a message
        agent.receive_message("Received message", sender_id=1, step=2)

        assert len(agent.received_messages) == 1
        assert agent.received_messages[0]["message"] == "Received message"
        assert agent.received_messages[0]["sender_id"] == 1

    def test_reset_functionality(self):
        """Test that reset clears message history."""
        agent = ChatFirmAgent(agent_id=0)

        # Add some messages
        agent.send_message("Test message", step=1)
        agent.receive_message("Received message", sender_id=1, step=2)

        assert len(agent.message_history) == 1
        assert len(agent.received_messages) == 1

        # Reset
        agent.reset()

        assert len(agent.message_history) == 0
        assert len(agent.received_messages) == 0


class TestCollusiveChatAgent:
    """Test cases for CollusiveChatAgent."""

    def test_collusive_agent_initialization(self):
        """Test that CollusiveChatAgent initializes correctly."""
        agent = CollusiveChatAgent(agent_id=0, collusion_intensity=0.8, seed=42)

        assert agent.agent_id == 0
        assert agent.collusion_intensity == 0.8

    def test_collusive_message_generation(self):
        """Test that collusive agent generates appropriate messages."""
        agent = CollusiveChatAgent(
            agent_id=0,
            collusion_intensity=1.0,  # Always generate collusive messages
            seed=42,
        )

        env = CartelEnv(n_firms=2, price_min=10, price_max=100)
        observation = np.array([50.0, 60.0])

        # Generate multiple messages to test variety
        messages = []
        for _ in range(10):
            message = agent.generate_message(observation, env)
            if message:
                messages.append(message)

        assert len(messages) > 0

        # Check that messages contain collusive content
        collusive_keywords = [
            "coordinate",
            "cooperate",
            "agree",
            "together",
            "pricing",
            "profit",
        ]

        has_collusive_content = any(
            any(keyword in message.lower() for keyword in collusive_keywords)
            for message in messages
        )
        assert has_collusive_content


class TestCompetitiveChatAgent:
    """Test cases for CompetitiveChatAgent."""

    def test_competitive_agent_initialization(self):
        """Test that CompetitiveChatAgent initializes correctly."""
        agent = CompetitiveChatAgent(agent_id=0, seed=42)

        assert agent.agent_id == 0
        assert agent.chat_enabled is True

    def test_competitive_message_generation(self):
        """Test that competitive agent generates appropriate messages."""
        agent = CompetitiveChatAgent(agent_id=0, seed=42)

        env = CartelEnv(n_firms=2, price_min=10, price_max=100)
        observation = np.array([50.0, 60.0])

        # Generate multiple messages to test variety
        messages = []
        for _ in range(10):
            message = agent.generate_message(observation, env)
            if message:
                messages.append(message)

        assert len(messages) > 0

        # Check that messages contain competitive content
        competitive_keywords = [
            "competition",
            "competitive",
            "customer",
            "value",
            "quality",
        ]

        has_competitive_content = any(
            any(keyword in message.lower() for keyword in competitive_keywords)
            for message in messages
        )
        assert has_competitive_content


class TestChatMessageManager:
    """Test cases for ChatMessageManager."""

    def test_message_manager_initialization(self):
        """Test that ChatMessageManager initializes correctly."""
        agents = [
            ChatFirmAgent(agent_id=0, message_frequency=1.0, seed=42),
            ChatFirmAgent(agent_id=1, message_frequency=1.0, seed=43),
        ]
        manager = ChatMessageManager(agents)

        assert len(manager.agents) == 2
        assert len(manager.episode_messages) == 0

    def test_message_collection_and_distribution(self):
        """Test message collection and distribution."""
        agents = [
            ChatFirmAgent(agent_id=0, message_frequency=1.0, seed=42),
            ChatFirmAgent(agent_id=1, message_frequency=1.0, seed=43),
        ]
        manager = ChatMessageManager(agents)

        env = CartelEnv(n_firms=2, price_min=10, price_max=100)
        observation = np.array([50.0, 60.0])

        # Collect messages
        messages = manager.collect_messages(step=1, observation=observation, env=env)

        assert len(messages) == 2  # Both agents should generate messages

        # Distribute messages
        manager.distribute_messages(messages, step=1)

        # Check that agents received messages
        for agent in agents:
            assert (
                len(agent.received_messages) == 1
            )  # Received one message from the other agent

    def test_reset_functionality(self):
        """Test that reset clears all message history."""
        agents = [
            ChatFirmAgent(agent_id=0, message_frequency=1.0, seed=42),
            ChatFirmAgent(agent_id=1, message_frequency=1.0, seed=43),
        ]
        manager = ChatMessageManager(agents)

        env = CartelEnv(n_firms=2, price_min=10, price_max=100)
        observation = np.array([50.0, 60.0])

        # Generate some messages
        manager.collect_messages(step=1, observation=observation, env=env)

        assert len(manager.episode_messages) > 0

        # Reset
        manager.reset()

        assert len(manager.episode_messages) == 0
        for agent in agents:
            assert len(agent.message_history) == 0
            assert len(agent.received_messages) == 0


class TestLLMDetector:
    """Test cases for LLMDetector."""

    def test_llm_detector_initialization(self):
        """Test that LLMDetector initializes correctly."""
        detector = LLMDetector(model_type="stubbed", confidence_threshold=0.7, seed=42)

        assert detector.model_type == "stubbed"
        assert detector.confidence_threshold == 0.7
        assert detector.total_messages_analyzed == 0
        assert detector.collusive_messages_detected == 0

    def test_collusive_message_classification(self):
        """Test classification of clearly collusive messages."""
        detector = LLMDetector(model_type="stubbed", confidence_threshold=0.5, seed=42)

        collusive_messages = [
            "We should coordinate our pricing strategy to maximize profits.",
            "Let's agree to keep prices above $50 to maintain margins.",
            "I suggest we all raise prices by 10% next period.",
            "We need to work together to avoid a price war.",
        ]

        for message in collusive_messages:
            result = detector.classify_message(
                message=message, sender_id=0, receiver_id=1, step=1
            )

            assert result["is_collusive"] is True
            assert result["collusive_probability"] > 0.5
            assert len(result["detected_patterns"]) > 0

    def test_non_collusive_message_classification(self):
        """Test classification of clearly non-collusive messages."""
        detector = LLMDetector(model_type="stubbed", confidence_threshold=0.5, seed=42)

        non_collusive_messages = [
            "I'm focused on delivering the best value to customers.",
            "Market competition is driving innovation in our sector.",
            "I'm working on reducing costs to stay competitive.",
            "Customer satisfaction is my top priority.",
        ]

        for message in non_collusive_messages:
            result = detector.classify_message(
                message=message, sender_id=0, receiver_id=1, step=1
            )

            # Non-collusive messages should have lower probability
            assert result["collusive_probability"] < 0.7
            # Most should not be classified as collusive
            if result["is_collusive"]:
                assert result["confidence"] < 0.8  # Low confidence if misclassified

    def test_batch_classification(self):
        """Test batch classification of multiple messages."""
        detector = LLMDetector(model_type="stubbed", confidence_threshold=0.5, seed=42)

        messages = [
            {"message": "We should coordinate pricing", "sender_id": 0, "step": 1},
            {"message": "I'm focused on customers", "sender_id": 1, "step": 1},
            {"message": "Let's agree on minimum prices", "sender_id": 0, "step": 2},
        ]

        results = detector.classify_messages_batch(messages)

        assert len(results) == 3
        assert all("is_collusive" in result for result in results)
        assert all("collusive_probability" in result for result in results)

    def test_detection_summary(self):
        """Test detection summary statistics."""
        detector = LLMDetector(model_type="stubbed", confidence_threshold=0.5, seed=42)

        # Classify some messages
        detector.classify_message("We should coordinate pricing", 0, 1, 1)
        detector.classify_message("I'm focused on customers", 1, 0, 1)
        detector.classify_message("Let's agree on prices", 0, 1, 2)

        summary = detector.get_detection_summary()

        assert summary["total_messages_analyzed"] == 3
        assert summary["collusive_messages_detected"] >= 0
        assert 0 <= summary["collusion_rate"] <= 1
        assert 0 <= summary["average_confidence"] <= 1

    def test_reset_functionality(self):
        """Test that reset clears detection history."""
        detector = LLMDetector(model_type="stubbed", seed=42)

        # Classify some messages
        detector.classify_message("Test message", 0, 1, 1)

        assert detector.total_messages_analyzed == 1
        assert len(detector.detection_history) == 1

        # Reset
        detector.reset()

        assert detector.total_messages_analyzed == 0
        assert len(detector.detection_history) == 0
        assert detector.collusive_messages_detected == 0


class TestChatRegulator:
    """Test cases for ChatRegulator."""

    def test_chat_regulator_initialization(self):
        """Test that ChatRegulator initializes correctly."""
        detector = LLMDetector(model_type="stubbed", seed=42)
        regulator = ChatRegulator(
            llm_detector=detector, message_fine_amount=25.0, collusion_threshold=0.7
        )

        assert regulator.message_fine_amount == 25.0
        assert regulator.collusion_threshold == 0.7
        assert len(regulator.message_violations) == 0
        assert regulator.total_message_fines == 0.0

    def test_message_monitoring(self):
        """Test message monitoring and violation detection."""
        detector = LLMDetector(model_type="stubbed", confidence_threshold=0.5, seed=42)
        regulator = ChatRegulator(
            llm_detector=detector, message_fine_amount=25.0, collusion_threshold=0.5
        )

        messages = [
            {"message": "We should coordinate pricing", "sender_id": 0, "step": 1},
            {"message": "I'm focused on customers", "sender_id": 1, "step": 1},
        ]

        result = regulator.monitor_messages(messages, step=1)

        assert result["step"] == 1
        assert result["messages_analyzed"] == 2
        assert result["collusive_messages"] >= 0
        assert result["fines_applied"] >= 0
        assert "violation_details" in result

    def test_violation_summary(self):
        """Test violation summary generation."""
        detector = LLMDetector(model_type="stubbed", confidence_threshold=0.5, seed=42)
        regulator = ChatRegulator(
            llm_detector=detector, message_fine_amount=25.0, collusion_threshold=0.5
        )

        # Monitor some messages
        messages = [
            {"message": "We should coordinate pricing", "sender_id": 0, "step": 1},
            {"message": "Let's agree on prices", "sender_id": 1, "step": 2},
        ]

        regulator.monitor_messages(messages, step=1)
        regulator.monitor_messages(messages, step=2)

        summary = regulator.get_message_violation_summary()

        assert "total_message_violations" in summary
        assert "total_message_fines" in summary
        assert "violation_steps" in summary
        assert "violations_by_agent" in summary

    def test_reset_functionality(self):
        """Test that reset clears regulator state."""
        detector = LLMDetector(model_type="stubbed", seed=42)
        regulator = ChatRegulator(detector)

        # Monitor some messages
        messages = [{"message": "Test message", "sender_id": 0, "step": 1}]
        regulator.monitor_messages(messages, step=1)

        assert len(regulator.message_violations) >= 0

        # Reset
        regulator.reset()

        assert len(regulator.message_violations) == 0
        assert regulator.total_message_fines == 0.0


class TestIntegration:
    """Integration tests for the complete chat system."""

    def test_message_pipeline_integration(self):
        """Test the complete message pipeline from generation to detection."""
        # Create chat agents
        agents = [
            CollusiveChatAgent(agent_id=0, message_frequency=1.0, seed=42),
            CompetitiveChatAgent(agent_id=1, message_frequency=1.0, seed=43),
        ]

        # Create message manager
        manager = ChatMessageManager(agents)

        # Create LLM detector and regulator
        detector = LLMDetector(model_type="stubbed", confidence_threshold=0.5, seed=42)
        regulator = ChatRegulator(detector, message_fine_amount=25.0)

        # Create environment
        env = CartelEnv(n_firms=2, price_min=10, price_max=100)
        observation = np.array([50.0, 60.0])

        # Run message pipeline
        messages = manager.collect_messages(step=1, observation=observation, env=env)
        manager.distribute_messages(messages, step=1)

        # Monitor messages
        monitoring_result = regulator.monitor_messages(messages, step=1)

        # Verify pipeline worked
        assert len(messages) > 0
        assert monitoring_result["messages_analyzed"] == len(messages)
        assert "classifications" in monitoring_result

    def test_deterministic_behavior_with_seed(self):
        """Test that behavior is deterministic with fixed seeds."""
        # Create agents with same seed
        agent1 = CollusiveChatAgent(agent_id=0, seed=42)
        agent2 = CollusiveChatAgent(agent_id=0, seed=42)

        env = CartelEnv(n_firms=2, price_min=10, price_max=100)
        observation = np.array([50.0, 60.0])

        # Generate messages
        message1 = agent1.generate_message(observation, env)
        message2 = agent2.generate_message(observation, env)

        # Messages should be identical with same seed
        assert message1 == message2

    def test_episode_logging_compatibility(self):
        """Test that chat messages can be integrated with episode logging."""
        # This test ensures the chat system is compatible with existing logging
        agents = [
            ChatFirmAgent(agent_id=0, message_frequency=1.0, seed=42),
            ChatFirmAgent(agent_id=1, message_frequency=1.0, seed=43),
        ]
        manager = ChatMessageManager(agents)

        env = CartelEnv(n_firms=2, price_min=10, price_max=100)
        observation = np.array([50.0, 60.0])

        # Generate messages
        messages = manager.collect_messages(step=1, observation=observation, env=env)

        # Messages should be in the expected format for logging
        for message in messages:
            assert "message" in message
            assert "sender_id" in message
            assert "step" in message
            assert "timestamp" in message

            # Should be JSON serializable
            import json

            json.dumps(message)


# Fixtures for testing
@pytest.fixture
def sample_cartel_env():
    """Provide a sample CartelEnv for testing."""
    return CartelEnv(n_firms=2, price_min=10, price_max=100, seed=42)


@pytest.fixture
def sample_chat_agents():
    """Provide sample chat agents for testing."""
    return [
        ChatFirmAgent(agent_id=0, message_frequency=1.0, seed=42),
        ChatFirmAgent(agent_id=1, message_frequency=1.0, seed=43),
    ]


@pytest.fixture
def sample_llm_detector():
    """Provide a sample LLMDetector for testing."""
    return LLMDetector(model_type="stubbed", confidence_threshold=0.5, seed=42)


@pytest.fixture
def sample_chat_regulator(sample_llm_detector):
    """Provide a sample ChatRegulator for testing."""
    return ChatRegulator(
        llm_detector=sample_llm_detector,
        message_fine_amount=25.0,
        collusion_threshold=0.5,
    )


# Test data fixtures
@pytest.fixture
def collusive_messages():
    """Provide sample collusive messages for testing."""
    return [
        "We should coordinate our pricing strategy to maximize profits.",
        "Let's agree to keep prices above $50 to maintain margins.",
        "I suggest we all raise prices by 10% next period.",
        "We need to work together to avoid a price war.",
        "Let's set a minimum price floor to protect our interests.",
    ]


@pytest.fixture
def non_collusive_messages():
    """Provide sample non-collusive messages for testing."""
    return [
        "I'm focused on delivering the best value to customers.",
        "Market competition is driving innovation in our sector.",
        "I'm working on reducing costs to stay competitive.",
        "Customer satisfaction is my top priority.",
        "I'm investing in new technology to improve efficiency.",
    ]
