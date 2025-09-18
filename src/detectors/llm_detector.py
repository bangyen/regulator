"""
LLM-based collusion detector for natural language message analysis.

This module implements an LLMDetector class that classifies messages between firms
into collusive vs non-collusive intent. Initially uses stubbed labels for testing,
with the ability to swap in real LLM models later.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMDetector:
    """
    LLM-based detector for classifying firm messages as collusive or non-collusive.

    This detector analyzes natural language messages between firms to identify
    potential collusive behavior. Initially uses rule-based classification with
    stubbed labels, designed to be easily extended with real LLM models.
    """

    def __init__(
        self,
        model_type: str = "stubbed",
        confidence_threshold: float = 0.7,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the LLM detector.

        Args:
            model_type: Type of model to use ("stubbed" for testing, "llm" for real model)
            confidence_threshold: Minimum confidence for collusion classification
            seed: Random seed for reproducibility
        """
        if model_type not in ["stubbed", "llm"]:
            raise ValueError("model_type must be 'stubbed' or 'llm'")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.np_random = np.random.default_rng(seed)

        # Detection state
        self.detection_history: List[Dict[str, Any]] = []
        self.total_messages_analyzed = 0
        self.collusive_messages_detected = 0

        # Initialize model based on type
        if model_type == "stubbed":
            self._initialize_stubbed_model()
        else:
            self._initialize_llm_model()

    def _initialize_stubbed_model(self) -> None:
        """Initialize the stubbed model with rule-based classification."""
        # Define collusive keywords and phrases
        self.collusive_keywords = {
            "coordinate",
            "coordination",
            "cooperate",
            "cooperation",
            "collude",
            "collusion",
            "agree",
            "agreement",
            "together",
            "jointly",
            "collective",
            "united",
            "price fixing",
            "price coordination",
            "price agreement",
            "price collusion",
            "minimum price",
            "price floor",
            "price ceiling",
            "price target",
            "avoid competition",
            "reduce competition",
            "limit competition",
            "market sharing",
            "market allocation",
            "territory",
            "customer allocation",
            "bid rigging",
            "bid coordination",
            "bid agreement",
            "signal",
            "signaling",
            "communicate",
            "discuss pricing",
            "pricing strategy",
            "maximize profit",
            "joint profit",
            "collective profit",
            "shared profit",
            "price war",
            "avoid price war",
            "prevent price war",
            "stop price war",
            "undercut",
            "undercutting",
            "price cutting",
            "aggressive pricing",
            "maintain price",
            "keep price",
            "hold price",
            "stabilize price",
            "raise price",
            "increase price",
            "higher price",
            "premium price",
        }

        # Define non-collusive keywords (for contrast)
        self.non_collusive_keywords = {
            "competition",
            "competitive",
            "compete",
            "competing",
            "rival",
            "rivalry",
            "customer",
            "customers",
            "value",
            "quality",
            "service",
            "innovation",
            "efficiency",
            "cost reduction",
            "productivity",
            "improvement",
            "market forces",
            "supply and demand",
            "natural market",
            "free market",
            "transparent",
            "transparency",
            "fair",
            "ethical",
            "legal",
            "compliance",
            "differentiate",
            "differentiation",
            "unique",
            "distinctive",
            "specialized",
            "customer satisfaction",
            "customer service",
            "customer focus",
            "independent",
            "autonomous",
            "individual",
            "separate",
            "standalone",
        }

        # Define collusive phrases (more specific patterns)
        self.collusive_phrases = [
            r"we should\s+(coordinate|cooperate|agree|work together)",
            r"let's\s+(coordinate|cooperate|agree|work together)",
            r"(coordinate|cooperate|agree)\s+our\s+(pricing|prices|strategy)",
            r"(set|establish|create)\s+(a\s+)?(minimum|maximum|target)\s+price",
            r"avoid\s+(undercutting|price war|competition)",
            r"(raise|increase|maintain|keep)\s+(our\s+)?prices",
            r"(discuss|talk about|communicate)\s+(pricing|prices)",
            r"(maximize|increase)\s+(joint|collective|shared)\s+profit",
            r"(signal|signaling)\s+(our|pricing)\s+(intentions|plans)",
            r"(work together|collaborate)\s+(to|on)\s+(pricing|prices)",
        ]

    def _initialize_llm_model(self) -> None:
        """
        Initialize the real LLM model with OpenAI integration.

        This method sets up the OpenAI client and validates the API key.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not available. Install with: pip install openai"
            )

        # Get API key from environment
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_KEY environment variable not set. "
                "Please set it in your .env file or environment."
            )

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)

        # Test the connection
        try:
            # Simple test call to verify API key works
            self.client.models.list()
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")

        # Set up the system prompt for collusion detection
        self.system_prompt = """You are an expert economic analyst specializing in detecting collusive behavior in market communications. 

Your task is to analyze messages between firms and determine if they contain evidence of collusive intent. Collusive behavior includes:
- Price fixing agreements
- Market sharing arrangements  
- Coordinated production decisions
- Bid rigging
- Information sharing that facilitates collusion

Respond with a JSON object containing:
- "is_collusive": boolean (true if collusive, false otherwise)
- "confidence": float (0.0 to 1.0, confidence in the assessment)
- "reasoning": string (brief explanation of your assessment)
- "evidence": string (specific phrases or patterns that support your conclusion)

Be conservative in your assessments - only flag clear evidence of collusive intent."""

    def classify_message(
        self,
        message: str,
        sender_id: int,
        receiver_id: int,
        step: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Classify a single message as collusive or non-collusive.

        Args:
            message: The message text to classify
            sender_id: ID of the agent that sent the message
            receiver_id: ID of the agent that received the message
            step: Current step number
            context: Additional context information

        Returns:
            Dictionary containing classification results
        """
        if self.model_type == "stubbed":
            return self._classify_with_stubbed_model(
                message, sender_id, receiver_id, step, context
            )
        else:
            return self._classify_with_llm_model(
                message, sender_id, receiver_id, step, context
            )

    def _classify_with_stubbed_model(
        self,
        message: str,
        sender_id: int,
        receiver_id: int,
        step: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Classify message using the stubbed rule-based model.

        Args:
            message: The message text to classify
            sender_id: ID of the agent that sent the message
            receiver_id: ID of the agent that received the message
            step: Current step number
            context: Additional context information

        Returns:
            Dictionary containing classification results
        """
        # Preprocess message
        message_lower = message.lower().strip()

        # Initialize classification scores
        collusive_score = 0.0
        non_collusive_score = 0.0
        detected_patterns = []

        # Check for collusive keywords
        for keyword in self.collusive_keywords:
            if keyword in message_lower:
                collusive_score += 0.1
                detected_patterns.append(f"collusive_keyword: {keyword}")

        # Check for non-collusive keywords
        for keyword in self.non_collusive_keywords:
            if keyword in message_lower:
                non_collusive_score += 0.05
                detected_patterns.append(f"non_collusive_keyword: {keyword}")

        # Check for collusive phrases (higher weight)
        for pattern in self.collusive_phrases:
            if re.search(pattern, message_lower):
                collusive_score += 0.3
                detected_patterns.append(f"collusive_phrase: {pattern}")

        # Add some randomness to simulate model uncertainty
        noise = self.np_random.normal(0, 0.05)
        collusive_score += noise

        # Normalize scores
        total_score = collusive_score + non_collusive_score
        if total_score > 0:
            collusive_prob = collusive_score / total_score
        else:
            collusive_prob = 0.5  # Default to neutral if no patterns detected

        # Ensure probability is in valid range
        collusive_prob = max(0.0, min(1.0, collusive_prob))

        # Determine classification
        is_collusive = collusive_prob >= self.confidence_threshold
        confidence = abs(collusive_prob - 0.5) * 2  # Convert to confidence measure

        # Create result
        result = {
            "message": message,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "step": step,
            "is_collusive": is_collusive,
            "collusive_probability": collusive_prob,
            "confidence": confidence,
            "detected_patterns": detected_patterns,
            "model_type": self.model_type,
            "context": context or {},
        }

        # Update statistics
        self.total_messages_analyzed += 1
        if is_collusive:
            self.collusive_messages_detected += 1

        # Store in history
        self.detection_history.append(result)

        return result

    def _classify_with_llm_model(
        self,
        message: str,
        sender_id: int,
        receiver_id: int,
        step: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Classify message using the real LLM model.

        Args:
            message: The message text to classify
            sender_id: ID of the agent that sent the message
            receiver_id: ID of the agent that received the message
            step: Current step number
            context: Additional context information

        Returns:
            Dictionary containing classification results
        """
        # Prepare the user message with context
        user_message = f"""Analyze this message between firms for collusive intent:

Message: "{message}"
Sender: Firm {sender_id}
Receiver: Firm {receiver_id}
Step: {step}"""

        if context:
            user_message += f"\nContext: {json.dumps(context, indent=2)}"

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500,
            )

            # Parse the response
            response_content = response.choices[0].message.content
            if response_content is None:
                raise ValueError("Empty response from OpenAI API")
            response_text = response_content.strip()

            # Try to parse as JSON
            try:
                result_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if response isn't valid JSON
                result_data = {
                    "is_collusive": False,
                    "confidence": 0.5,
                    "reasoning": "Could not parse LLM response",
                    "evidence": response_text,
                }

            # Extract results
            is_collusive = result_data.get("is_collusive", False)
            confidence = float(result_data.get("confidence", 0.5))
            reasoning = result_data.get("reasoning", "No reasoning provided")
            evidence = result_data.get("evidence", "No evidence provided")

            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))

        except Exception:
            # Fallback to stubbed model if LLM fails
            return self._classify_with_stubbed_model(
                message, sender_id, receiver_id, step, context
            )

        # Create result
        result = {
            "message": message,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "step": step,
            "is_collusive": is_collusive,
            "collusive_probability": confidence if is_collusive else (1.0 - confidence),
            "confidence": confidence,
            "reasoning": reasoning,
            "evidence": evidence,
            "model_type": self.model_type,
            "context": context or {},
        }

        # Update statistics
        self.total_messages_analyzed += 1
        if is_collusive:
            self.collusive_messages_detected += 1

        # Store in history
        self.detection_history.append(result)

        return result

    def classify_messages_batch(
        self,
        messages: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple messages in batch.

        Args:
            messages: List of message dictionaries
            context: Additional context information

        Returns:
            List of classification results
        """
        results = []
        for message_data in messages:
            result = self.classify_message(
                message=message_data["message"],
                sender_id=message_data["sender_id"],
                receiver_id=message_data.get("receiver_id", -1),
                step=message_data["step"],
                context=context,
            )
            results.append(result)

        return results

    def get_detection_summary(self) -> Dict[str, Any]:
        """
        Get a summary of detection results.

        Returns:
            Dictionary containing detection statistics
        """
        if self.total_messages_analyzed == 0:
            return {
                "total_messages_analyzed": 0,
                "collusive_messages_detected": 0,
                "collusion_rate": 0.0,
                "average_confidence": 0.0,
                "model_type": self.model_type,
            }

        collusion_rate = self.collusive_messages_detected / self.total_messages_analyzed

        # Calculate average confidence
        confidences = [result["confidence"] for result in self.detection_history]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return {
            "total_messages_analyzed": self.total_messages_analyzed,
            "collusive_messages_detected": self.collusive_messages_detected,
            "collusion_rate": collusion_rate,
            "average_confidence": avg_confidence,
            "model_type": self.model_type,
        }

    def get_collusive_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages classified as collusive.

        Returns:
            List of collusive message classifications
        """
        return [result for result in self.detection_history if result["is_collusive"]]

    def get_detection_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete detection history.

        Returns:
            List of all classification results
        """
        return self.detection_history.copy()

    def reset(self) -> None:
        """Reset the detector's state for a new episode."""
        self.detection_history.clear()
        self.total_messages_analyzed = 0
        self.collusive_messages_detected = 0

    def export_detection_results(self, filepath: str) -> None:
        """
        Export detection results to a JSON file.

        Args:
            filepath: Path to save the results
        """
        export_data = {
            "detection_summary": self.get_detection_summary(),
            "detection_history": self.detection_history,
            "collusive_messages": self.get_collusive_messages(),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

    def load_detection_results(self, filepath: str) -> None:
        """
        Load detection results from a JSON file.

        Args:
            filepath: Path to load the results from
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        self.detection_history = data.get("detection_history", [])
        self.total_messages_analyzed = len(self.detection_history)
        self.collusive_messages_detected = len(
            [r for r in self.detection_history if r["is_collusive"]]
        )


class ChatRegulator:
    """
    Regulator that monitors chat messages for collusion detection.

    This class integrates the LLMDetector with the existing regulator framework,
    allowing for message-based collusion detection in addition to price-based detection.
    """

    def __init__(
        self,
        llm_detector: LLMDetector,
        message_fine_amount: float = 25.0,
        collusion_threshold: float = 0.7,
    ) -> None:
        """
        Initialize the chat regulator.

        Args:
            llm_detector: The LLM detector instance
            message_fine_amount: Fine amount for collusive messages
            collusion_threshold: Threshold for collusion classification
        """
        self.llm_detector = llm_detector
        self.message_fine_amount = message_fine_amount
        self.collusion_threshold = collusion_threshold

        # Monitoring state
        self.message_violations: List[Dict[str, Any]] = []
        self.total_message_fines = 0.0

    def monitor_messages(
        self,
        messages: List[Dict[str, Any]],
        step: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Monitor messages for collusive behavior.

        Args:
            messages: List of messages to monitor
            step: Current step number
            context: Additional context information

        Returns:
            Dictionary containing monitoring results
        """
        if not messages:
            return {
                "step": step,
                "messages_analyzed": 0,
                "collusive_messages": 0,
                "fines_applied": 0.0,
                "violation_details": [],
            }

        # Classify all messages
        classifications = self.llm_detector.classify_messages_batch(messages, context)

        # Identify collusive messages
        collusive_messages = [
            result
            for result in classifications
            if result["is_collusive"]
            and result["confidence"] >= self.collusion_threshold
        ]

        # Calculate fines
        total_fines = len(collusive_messages) * self.message_fine_amount

        # Record violations
        for result in collusive_messages:
            violation = {
                "step": step,
                "message": result["message"],
                "sender_id": result["sender_id"],
                "receiver_id": result["receiver_id"],
                "confidence": result["confidence"],
                "fine_amount": self.message_fine_amount,
            }
            self.message_violations.append(violation)

        self.total_message_fines += total_fines

        return {
            "step": step,
            "messages_analyzed": len(messages),
            "collusive_messages": len(collusive_messages),
            "fines_applied": total_fines,
            "violation_details": [
                f"Collusive message from agent {result['sender_id']}: {result['message'][:50]}..."
                for result in collusive_messages
            ],
            "classifications": classifications,
        }

    def get_message_violation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of message violations.

        Returns:
            Dictionary containing violation statistics
        """
        return {
            "total_message_violations": len(self.message_violations),
            "total_message_fines": self.total_message_fines,
            "violation_steps": list(set(v["step"] for v in self.message_violations)),
            "violations_by_agent": self._get_violations_by_agent(),
        }

    def _get_violations_by_agent(self) -> Dict[int, int]:
        """
        Get violation counts by agent.

        Returns:
            Dictionary mapping agent_id to violation count
        """
        violations_by_agent: Dict[int, int] = {}
        for violation in self.message_violations:
            sender_id = violation["sender_id"]
            violations_by_agent[sender_id] = violations_by_agent.get(sender_id, 0) + 1
        return violations_by_agent

    def reset(self) -> None:
        """Reset the chat regulator's state."""
        self.message_violations.clear()
        self.total_message_fines = 0.0
        self.llm_detector.reset()
