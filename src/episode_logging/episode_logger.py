"""
Enhanced episode logger for chat-enabled firm agents.

This module extends the base logger to include chat messages, LLM detection results,
and regulator monitoring data for comprehensive episode analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .logger import Logger


class EpisodeLogger(Logger):
    """
    Enhanced episode logger for chat-enabled firm agents.

    This class extends the base Logger to include chat messages, LLM detection
    results, and regulator monitoring data for comprehensive episode analysis.
    """

    def __init__(
        self,
        log_file: Union[str, Path],
        n_firms: int = 3,
    ) -> None:
        """
        Initialize the enhanced episode logger.

        Args:
            log_file: Path to the log file
            n_firms: Number of firms in the environment
        """
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize base logger
        super().__init__(
            log_dir=log_path.parent,
            episode_id=log_path.stem,
            n_firms=n_firms,
        )

        # Override log file path
        self.log_file = log_path

    def log_episode_header(
        self,
        episode_id: int,
        n_firms: int,
        n_steps: int,
        agent_types: List[str],
        environment_params: Dict[str, Any],
    ) -> None:
        """
        Log enhanced episode header with agent and environment information.

        Args:
            episode_id: Unique episode identifier
            n_firms: Number of firms
            n_steps: Number of steps in the episode
            agent_types: List of agent type names
            environment_params: Environment configuration parameters
        """
        header = {
            "type": "episode_header",
            "episode_id": episode_id,
            "start_time": datetime.now().isoformat(),
            "n_firms": n_firms,
            "n_steps": n_steps,
            "agent_types": agent_types,
            "environment_params": environment_params,
            "log_file": str(self.log_file),
            "episode_summary": {
                "environment_params": environment_params,
                "agent_types": agent_types,
            },
        }

        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")

    def log_step(
        self,
        step: int,
        prices: np.ndarray,
        profits: np.ndarray,
        demand_shock: float,
        market_price: float,
        total_demand: float,
        individual_quantity: float,
        total_profits: np.ndarray,
        regulator_flags: Optional[Dict[str, Any]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        # Additional parameters for chat functionality
        rewards: Optional[List[float]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        chat_monitoring: Optional[Dict[str, Any]] = None,
        price_monitoring: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log enhanced step data including chat messages and monitoring results.

        Args:
            step: Current step number
            prices: List of prices chosen by each firm
            profits: List of profits for each firm (after penalties)
            rewards: List of original rewards for each firm (before penalties)
            demand_shock: Current demand shock value
            market_price: Average market price
            messages: List of chat messages from this step
            chat_monitoring: Chat monitoring results from regulator
            price_monitoring: Price monitoring results from regulator
            additional_info: Optional additional information
        """
        # Validate inputs
        if len(prices) != self.n_firms:
            raise ValueError(f"Prices array must have length {self.n_firms}")
        if len(profits) != self.n_firms:
            raise ValueError(f"Profits array must have length {self.n_firms}")

        # Create step data record
        step_data = {
            "type": "step",
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "prices": [float(p) for p in prices.tolist()],
            "profits": [float(p) for p in profits.tolist()],
            "demand_shock": float(demand_shock),
            "market_price": float(market_price),
        }

        # Add original rewards if provided
        if rewards is not None:
            if len(rewards) != self.n_firms:
                raise ValueError(f"Rewards array must have length {self.n_firms}")
            step_data["rewards"] = [float(r) for r in rewards]

        # Add chat messages if provided
        if messages is not None:
            step_data["messages"] = messages
            step_data["n_messages"] = len(messages)

        # Add chat monitoring results if provided
        if chat_monitoring is not None:
            chat_monitoring_data: Dict[str, Any] = {
                "messages_analyzed": chat_monitoring.get("messages_analyzed", 0),
                "collusive_messages": chat_monitoring.get("collusive_messages", 0),
                "fines_applied": float(chat_monitoring.get("fines_applied", 0.0)),
                "violation_details": chat_monitoring.get("violation_details", []),
            }

            # Include classification results if available
            if "classifications" in chat_monitoring:
                chat_monitoring_data["classifications"] = chat_monitoring[
                    "classifications"
                ]

            step_data["chat_monitoring"] = chat_monitoring_data

        # Add price monitoring results if provided
        if price_monitoring is not None:
            step_data["price_monitoring"] = {
                "parallel_violation": price_monitoring.get("parallel_violation", False),
                "structural_break_violation": price_monitoring.get(
                    "structural_break_violation", False
                ),
                "fines_applied": [
                    float(f) for f in price_monitoring.get("fines_applied", [])
                ],
                "violation_details": price_monitoring.get("violation_details", []),
            }

        # Add additional info if provided
        if additional_info is not None:
            step_data["additional_info"] = additional_info

        # Store in memory and write to file
        self.episode_data.append(step_data)
        self.step_count = step

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_data) + "\n")

    def log_episode_summary(
        self,
        total_reward: float,
        total_steps: int,
        final_prices: List[float],
        final_profits: List[float],
        chat_summary: Optional[Dict[str, Any]] = None,
        price_summary: Optional[Dict[str, Any]] = None,
        additional_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log enhanced episode summary with chat and price monitoring results.

        Args:
            total_reward: Total reward across all firms
            total_steps: Total number of steps completed
            final_prices: Final prices for each firm
            final_profits: Final profits for each firm
            chat_summary: Summary of chat monitoring results
            price_summary: Summary of price monitoring results
            additional_summary: Additional summary statistics
        """
        end_time = datetime.now()
        duration = (end_time - self.episode_start_time).total_seconds()

        summary_data = {
            "type": "episode_summary",
            "episode_id": self.episode_id,
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_steps": total_steps,
            "total_reward": float(total_reward),
            "final_prices": [float(p) for p in final_prices],
            "final_profits": [float(p) for p in final_profits],
        }

        # Add chat summary if provided
        if chat_summary is not None:
            summary_data["chat_summary"] = {
                "total_message_violations": chat_summary.get(
                    "total_message_violations", 0
                ),
                "total_message_fines": float(
                    chat_summary.get("total_message_fines", 0.0)
                ),
                "violation_steps": chat_summary.get("violation_steps", []),
                "violations_by_agent": chat_summary.get("violations_by_agent", {}),
            }

        # Add price summary if provided
        if price_summary is not None:
            summary_data["price_summary"] = {
                "total_parallel_violations": price_summary.get(
                    "total_parallel_violations", 0
                ),
                "total_structural_break_violations": price_summary.get(
                    "total_structural_break_violations", 0
                ),
                "total_fines_applied": float(
                    price_summary.get("total_fines_applied", 0.0)
                ),
                "parallel_violation_steps": price_summary.get(
                    "parallel_violation_steps", []
                ),
                "structural_break_steps": price_summary.get(
                    "structural_break_steps", []
                ),
            }

        # Add additional summary if provided
        if additional_summary is not None:
            summary_data["additional_summary"] = additional_summary

        # Write to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_data) + "\n")

    def get_chat_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about chat messages from the episode.

        Returns:
            Dictionary containing chat message statistics
        """
        total_messages = 0
        total_collusive_messages = 0
        total_chat_fines = 0.0
        messages_by_agent: Dict[int, int] = {}
        collusive_messages_by_agent: Dict[int, int] = {}

        for step_data in self.episode_data:
            if "messages" in step_data:
                messages = step_data["messages"]
                total_messages += len(messages)

                for message in messages:
                    sender_id = message["sender_id"]
                    messages_by_agent[sender_id] = (
                        messages_by_agent.get(sender_id, 0) + 1
                    )

            if "chat_monitoring" in step_data:
                monitoring = step_data["chat_monitoring"]
                total_collusive_messages += monitoring.get("collusive_messages", 0)
                total_chat_fines += monitoring.get("fines_applied", 0.0)

                # Count collusive messages by agent
                if "classifications" in monitoring:
                    for classification in monitoring["classifications"]:
                        if classification.get("is_collusive", False):
                            sender_id = classification["sender_id"]
                            collusive_messages_by_agent[sender_id] = (
                                collusive_messages_by_agent.get(sender_id, 0) + 1
                            )

        return {
            "total_messages": total_messages,
            "total_collusive_messages": total_collusive_messages,
            "collusion_rate": total_collusive_messages / max(1, total_messages),
            "total_chat_fines": total_chat_fines,
            "messages_by_agent": messages_by_agent,
            "collusive_messages_by_agent": collusive_messages_by_agent,
        }

    def get_price_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about price monitoring from the episode.

        Returns:
            Dictionary containing price monitoring statistics
        """
        total_parallel_violations = 0
        total_structural_break_violations = 0
        total_price_fines = 0.0
        parallel_violation_steps = []
        structural_break_steps = []

        for step_data in self.episode_data:
            if "price_monitoring" in step_data:
                monitoring = step_data["price_monitoring"]

                if monitoring.get("parallel_violation", False):
                    total_parallel_violations += 1
                    parallel_violation_steps.append(step_data["step"])

                if monitoring.get("structural_break_violation", False):
                    total_structural_break_violations += 1
                    structural_break_steps.append(step_data["step"])

                fines = monitoring.get("fines_applied", [])
                total_price_fines += sum(fines)

        return {
            "total_parallel_violations": total_parallel_violations,
            "total_structural_break_violations": total_structural_break_violations,
            "total_price_fines": total_price_fines,
            "parallel_violation_steps": parallel_violation_steps,
            "structural_break_steps": structural_break_steps,
        }

    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive episode statistics.

        Returns:
            Dictionary containing all episode statistics
        """
        if not self.episode_data:
            return {}

        # Calculate price statistics
        all_prices = []
        all_profits = []

        for step_data in self.episode_data:
            all_prices.extend(step_data["prices"])
            all_profits.extend(step_data["profits"])

        price_stats = {
            "mean_price": float(np.mean(all_prices)) if all_prices else 0.0,
            "std_price": float(np.std(all_prices)) if all_prices else 0.0,
            "min_price": float(np.min(all_prices)) if all_prices else 0.0,
            "max_price": float(np.max(all_prices)) if all_prices else 0.0,
        }

        profit_stats = {
            "mean_profit": float(np.mean(all_profits)) if all_profits else 0.0,
            "std_profit": float(np.std(all_profits)) if all_profits else 0.0,
            "min_profit": float(np.min(all_profits)) if all_profits else 0.0,
            "max_profit": float(np.max(all_profits)) if all_profits else 0.0,
        }

        return {
            "episode_id": self.episode_id,
            "total_steps": self.step_count,
            "n_firms": self.n_firms,
            "price_statistics": price_stats,
            "profit_statistics": profit_stats,
            "chat_statistics": self.get_chat_statistics(),
            "price_monitoring_statistics": self.get_price_statistics(),
        }

    @classmethod
    def load_chat_episode_data(cls, log_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Load chat-enabled episode data from a JSONL log file.

        Args:
            log_file: Path to the JSONL log file

        Returns:
            Dictionary containing episode header, steps, and summary data
        """
        log_file = Path(log_file)

        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")

        episode_data: Dict[str, Any] = {
            "header": None,
            "steps": [],
            "summary": None,
        }

        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())

                    if data["type"] == "episode_header":
                        episode_data["header"] = data
                    elif data["type"] == "step":
                        episode_data["steps"].append(data)
                    elif data["type"] == "episode_summary":
                        episode_data["summary"] = data

        return episode_data

    @classmethod
    def validate_chat_log_file(cls, log_file: Union[str, Path]) -> bool:
        """
        Validate that a chat log file contains required data structure.

        Args:
            log_file: Path to the JSONL log file

        Returns:
            True if log file is valid, False otherwise
        """
        try:
            episode_data = cls.load_chat_episode_data(log_file)

            # Check required components
            if episode_data["header"] is None:
                return False

            if not episode_data["steps"]:
                return False

            # Check that each step has required keys
            required_keys = {
                "step",
                "prices",
                "profits",
                "demand_shock",
                "market_price",
            }

            for step_data in episode_data["steps"]:
                if not required_keys.issubset(step_data.keys()):
                    return False

                # Check array lengths
                n_firms = episode_data["header"]["n_firms"]
                if len(step_data["prices"]) != n_firms:
                    return False
                if len(step_data["profits"]) != n_firms:
                    return False

            return True

        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
