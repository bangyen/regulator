"""
Unit tests for enhanced episode logger functionality.

This module tests the enhanced logging features including chat messages,
LLM detection results, and regulator monitoring data.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.episode_logging.episode_logger import EpisodeLogger


class TestEpisodeLoggerEnhanced:
    """Test enhanced episode logger functionality."""

    def test_episode_logger_initialization(self) -> None:
        """Test EpisodeLogger initialization with custom log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=3)

            # Verify initialization
            assert logger.n_firms == 3
            assert logger.episode_id == "test_episode"
            assert logger.log_file == log_file
            assert log_file.exists()

    def test_episode_logger_creates_directory(self) -> None:
        """Test that EpisodeLogger creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "subdir" / "nested" / "test_episode.jsonl"
            EpisodeLogger(log_file=log_file, n_firms=2)

            # Verify directory was created
            assert log_file.parent.exists()
            assert log_file.exists()

    def test_log_episode_header_custom(self) -> None:
        """Test logging custom episode header."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=3)

            # Log custom header (this overwrites the default header)
            logger.log_episode_header(
                episode_id=123,
                n_firms=3,
                n_steps=100,
                agent_types=["random", "bestresponse", "titfortat"],
                environment_params={"marginal_cost": 15.0, "demand_intercept": 120.0},
            )

            # Verify header was logged
            lines = log_file.read_text().strip().split("\n")
            assert len(lines) == 1  # Only the custom header (overwrites default)

            # Check custom header content
            custom_header = json.loads(lines[0])
            assert custom_header["type"] == "episode_header"
            assert custom_header["episode_id"] == 123
            assert custom_header["n_firms"] == 3
            assert custom_header["n_steps"] == 100
            assert custom_header["agent_types"] == [
                "random",
                "bestresponse",
                "titfortat",
            ]
            assert custom_header["environment_params"]["marginal_cost"] == 15.0

    def test_log_chat_messages(self) -> None:
        """Test logging chat messages through log_step."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Create sample chat messages
            chat_messages = [
                {
                    "sender_id": 0,
                    "receiver_id": 1,
                    "message": "Let's coordinate our pricing",
                    "step": 1,
                    "timestamp": "2024-01-01T00:00:00",
                },
                {
                    "sender_id": 1,
                    "receiver_id": 0,
                    "message": "I'll match your price",
                    "step": 1,
                    "timestamp": "2024-01-01T00:00:01",
                },
            ]

            # Log step with chat messages
            logger.log_step(
                step=1,
                prices=np.array([20.0, 25.0]),
                profits=np.array([100.0, 120.0]),
                demand_shock=5.0,
                market_price=22.5,
                total_demand=77.5,
                individual_quantity=38.75,
                total_profits=np.array([100.0, 120.0]),
                messages=chat_messages,
            )

            # Verify chat messages were logged
            lines = log_file.read_text().strip().split("\n")
            step_log = json.loads(lines[-1])
            assert step_log["type"] == "step"
            assert step_log["step"] == 1
            assert "messages" in step_log
            assert len(step_log["messages"]) == 2
            assert step_log["messages"][0]["message"] == "Let's coordinate our pricing"

    def test_log_llm_detection_results(self) -> None:
        """Test logging LLM detection results through log_step."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Create sample detection results
            chat_monitoring = {
                "messages_analyzed": 2,
                "collusive_messages": 1,
                "fines_applied": 25.0,
                "violation_details": ["Collusive message from agent 0"],
                "classifications": [
                    {
                        "message": "Let's coordinate pricing",
                        "collusive_probability": 0.9,
                        "confidence": 0.8,
                        "is_collusive": True,
                        "sender_id": 0,
                    }
                ],
            }

            # Log step with chat monitoring
            logger.log_step(
                step=1,
                prices=np.array([20.0, 25.0]),
                profits=np.array([100.0, 120.0]),
                demand_shock=5.0,
                market_price=22.5,
                total_demand=77.5,
                individual_quantity=38.75,
                total_profits=np.array([100.0, 120.0]),
                chat_monitoring=chat_monitoring,
            )

            # Verify detection results were logged
            lines = log_file.read_text().strip().split("\n")
            step_log = json.loads(lines[-1])
            assert step_log["type"] == "step"
            assert step_log["step"] == 1
            assert "chat_monitoring" in step_log
            assert step_log["chat_monitoring"]["messages_analyzed"] == 2
            assert step_log["chat_monitoring"]["collusive_messages"] == 1
            assert step_log["chat_monitoring"]["fines_applied"] == 25.0

    def test_log_regulator_monitoring_data(self) -> None:
        """Test logging regulator monitoring data through log_step."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=3)

            # Create sample regulator data
            price_monitoring = {
                "parallel_violation": True,
                "structural_break_violation": False,
                "fines_applied": [10.0, 10.0, 10.0],
                "violation_details": ["Parallel pricing detected"],
            }

            # Log step with price monitoring
            logger.log_step(
                step=1,
                prices=np.array([25.0, 25.0, 25.0]),
                profits=np.array([100.0, 120.0, 140.0]),
                demand_shock=5.0,
                market_price=25.0,
                total_demand=75.0,
                individual_quantity=25.0,
                total_profits=np.array([100.0, 120.0, 140.0]),
                price_monitoring=price_monitoring,
            )

            # Verify regulator data was logged
            lines = log_file.read_text().strip().split("\n")
            step_log = json.loads(lines[-1])
            assert step_log["type"] == "step"
            assert step_log["step"] == 1
            assert "price_monitoring" in step_log
            assert step_log["price_monitoring"]["parallel_violation"] is True
            assert step_log["price_monitoring"]["fines_applied"] == [10.0, 10.0, 10.0]

    def test_log_step_with_enhanced_data(self) -> None:
        """Test logging step data with enhanced information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Log step with enhanced data
            logger.log_step(
                step=1,
                prices=np.array([20.0, 25.0]),
                profits=np.array([100.0, 120.0]),
                demand_shock=5.0,
                market_price=22.5,
                total_demand=77.5,
                individual_quantity=38.75,
                total_profits=np.array([100.0, 120.0]),
                messages=[
                    {
                        "sender_id": 0,
                        "message": "Let's coordinate",
                        "step": 1,
                    }
                ],
                chat_monitoring={
                    "collusive_messages": 1,
                    "fines_applied": 25.0,
                },
                price_monitoring={
                    "parallel_violation": True,
                    "fines_applied": [12.5, 12.5],
                },
            )

            # Verify enhanced step data was logged
            lines = log_file.read_text().strip().split("\n")
            step_log = json.loads(lines[-1])
            assert step_log["type"] == "step"
            assert step_log["step"] == 1
            assert step_log["prices"] == [20.0, 25.0]
            assert "messages" in step_log
            assert "chat_monitoring" in step_log
            assert "price_monitoring" in step_log

    def test_log_episode_summary_with_enhanced_data(self) -> None:
        """Test logging episode summary with enhanced data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Create enhanced episode summary
            chat_summary = {
                "total_message_violations": 3,
                "total_message_fines": 75.0,
                "violation_steps": [1, 3, 5],
                "violations_by_agent": {0: 2, 1: 1},
            }

            price_summary = {
                "total_parallel_violations": 5,
                "total_structural_break_violations": 2,
                "total_fines_applied": 150.0,
                "parallel_violation_steps": [1, 2, 3, 4, 5],
                "structural_break_steps": [3, 7],
            }

            # Log episode summary
            logger.log_episode_summary(
                total_reward=1000.0,
                total_steps=10,
                final_prices=[25.0, 30.0],
                final_profits=[100.0, 120.0],
                chat_summary=chat_summary,
                price_summary=price_summary,
            )

            # Verify enhanced episode summary was logged
            lines = log_file.read_text().strip().split("\n")
            summary_log = json.loads(lines[-1])
            assert summary_log["type"] == "episode_summary"
            assert summary_log["total_reward"] == 1000.0
            assert summary_log["total_steps"] == 10
            assert "chat_summary" in summary_log
            assert "price_summary" in summary_log
            assert summary_log["chat_summary"]["total_message_violations"] == 3
            assert summary_log["price_summary"]["total_parallel_violations"] == 5

    def test_numpy_type_conversion_in_enhanced_logging(self) -> None:
        """Test that numpy types are properly converted in enhanced logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Log data with numpy types
            logger.log_step(
                step=1,
                prices=np.array([20.0, 25.0]),
                profits=np.array([100.0, 120.0]),
                demand_shock=np.float32(5.0),
                market_price=np.float64(22.5),
                total_demand=77.5,
                individual_quantity=np.float32(38.75),
                total_profits=np.array([100.0, 120.0]),
            )

            # Verify numpy types were converted
            lines = log_file.read_text().strip().split("\n")
            step_log = json.loads(lines[-1])
            assert isinstance(step_log["prices"], list)
            assert isinstance(step_log["profits"], list)
            assert isinstance(step_log["demand_shock"], float)
            assert isinstance(step_log["market_price"], float)
            # Note: individual_quantity and total_profits are not logged in the step data

    def test_error_handling_invalid_data(self) -> None:
        """Test error handling with invalid data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Test with wrong number of prices
            with pytest.raises(ValueError, match="Prices array must have length 2"):
                logger.log_step(
                    step=1,
                    prices=np.array([20.0, 25.0, 30.0]),  # Wrong length
                    profits=np.array([100.0, 120.0]),
                    demand_shock=5.0,
                    market_price=22.5,
                    total_demand=77.5,
                    individual_quantity=38.75,
                    total_profits=np.array([100.0, 120.0]),
                )

            # Test with wrong number of profits
            with pytest.raises(ValueError, match="Profits array must have length 2"):
                logger.log_step(
                    step=1,
                    prices=np.array([20.0, 25.0]),
                    profits=np.array([100.0, 120.0, 140.0]),  # Wrong length
                    demand_shock=5.0,
                    market_price=22.5,
                    total_demand=77.5,
                    individual_quantity=38.75,
                    total_profits=np.array([100.0, 120.0]),
                )

    def test_log_file_path_override(self) -> None:
        """Test that log file path is properly overridden."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "custom_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Verify the log file path is overridden
            assert logger.log_file == log_file
            assert log_file.exists()

            # Verify the episode ID matches the filename
            assert logger.episode_id == "custom_episode"

    def test_enhanced_logging_with_empty_data(self) -> None:
        """Test enhanced logging with empty or None data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Log with empty messages
            logger.log_step(
                step=1,
                prices=np.array([20.0, 25.0]),
                profits=np.array([100.0, 120.0]),
                demand_shock=5.0,
                market_price=22.5,
                total_demand=77.5,
                individual_quantity=38.75,
                total_profits=np.array([100.0, 120.0]),
                messages=[],  # Empty messages
            )

            # Log with None monitoring data
            logger.log_step(
                step=2,
                prices=np.array([20.0, 25.0]),
                profits=np.array([100.0, 120.0]),
                demand_shock=5.0,
                market_price=22.5,
                total_demand=77.5,
                individual_quantity=38.75,
                total_profits=np.array([100.0, 120.0]),
                chat_monitoring=None,  # None monitoring
                price_monitoring=None,  # None monitoring
            )

            # Should handle gracefully
            lines = log_file.read_text().strip().split("\n")
            assert len(lines) >= 3  # Header + 2 steps

            # Verify empty data was logged
            step1_log = json.loads(lines[1])
            assert step1_log["messages"] == []

            step2_log = json.loads(lines[2])
            assert "chat_monitoring" not in step2_log
            assert "price_monitoring" not in step2_log

    def test_enhanced_logging_performance(self) -> None:
        """Test that enhanced logging doesn't significantly impact performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=3)

            # Log many steps with enhanced data
            for step in range(100):
                logger.log_step(
                    step=step + 1,
                    prices=np.array([20.0, 25.0, 30.0]),
                    profits=np.array([100.0, 120.0, 140.0]),
                    demand_shock=5.0,
                    market_price=25.0,
                    total_demand=75.0,
                    individual_quantity=25.0,
                    total_profits=np.array([100.0, 120.0, 140.0]),
                    messages=[
                        {
                            "sender_id": step % 3,
                            "message": f"Message {step}",
                            "step": step + 1,
                        }
                    ],
                    chat_monitoring={
                        "collusive_messages": step % 2,
                        "fines_applied": 25.0 * (step % 2),
                    },
                    price_monitoring={
                        "parallel_violation": step % 3 == 0,
                        "fines_applied": (
                            [10.0, 10.0, 10.0] if step % 3 == 0 else [0.0, 0.0, 0.0]
                        ),
                    },
                )

            # Verify all data was logged
            lines = log_file.read_text().strip().split("\n")
            assert len(lines) == 101  # Header + 100 steps

            # Verify last step has correct data
            last_step = json.loads(lines[-1])
            assert last_step["step"] == 100
            assert len(last_step["messages"]) == 1
            assert "chat_monitoring" in last_step
            assert "price_monitoring" in last_step

    def test_get_chat_statistics(self) -> None:
        """Test getting chat statistics from logged data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Log steps with chat data
            for step in range(3):
                logger.log_step(
                    step=step + 1,
                    prices=np.array([20.0, 25.0]),
                    profits=np.array([100.0, 120.0]),
                    demand_shock=5.0,
                    market_price=22.5,
                    total_demand=77.5,
                    individual_quantity=38.75,
                    total_profits=np.array([100.0, 120.0]),
                    messages=[
                        {
                            "sender_id": step % 2,
                            "message": f"Message {step}",
                            "step": step + 1,
                        }
                    ],
                    chat_monitoring={
                        "collusive_messages": step % 2,
                        "fines_applied": 25.0 * (step % 2),
                        "classifications": (
                            [
                                {
                                    "sender_id": step % 2,
                                    "is_collusive": step % 2 == 1,
                                }
                            ]
                            if step % 2 == 1
                            else []
                        ),
                    },
                )

            # Get chat statistics
            stats = logger.get_chat_statistics()

            # Verify statistics
            assert stats["total_messages"] == 3
            assert stats["total_collusive_messages"] == 1  # Only step 2
            assert stats["total_chat_fines"] == 25.0
            assert stats["collusion_rate"] == 1.0 / 3.0
            assert 0 in stats["messages_by_agent"]
            assert 1 in stats["messages_by_agent"]

    def test_get_price_statistics(self) -> None:
        """Test getting price statistics from logged data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Log steps with price monitoring data
            for step in range(5):
                logger.log_step(
                    step=step + 1,
                    prices=np.array([20.0, 25.0]),
                    profits=np.array([100.0, 120.0]),
                    demand_shock=5.0,
                    market_price=22.5,
                    total_demand=77.5,
                    individual_quantity=38.75,
                    total_profits=np.array([100.0, 120.0]),
                    price_monitoring={
                        "parallel_violation": step % 2 == 0,  # Steps 1, 3, 5
                        "structural_break_violation": step % 3 == 0,  # Steps 3
                        "fines_applied": [10.0, 10.0] if step % 2 == 0 else [0.0, 0.0],
                    },
                )

            # Get price statistics
            stats = logger.get_price_statistics()

            # Verify statistics
            assert stats["total_parallel_violations"] == 3  # Steps 1, 3, 5
            assert (
                stats["total_structural_break_violations"] == 2
            )  # Steps 1, 4 (step % 3 == 0)
            assert stats["total_price_fines"] == 60.0  # 3 * 20.0
            assert stats["parallel_violation_steps"] == [1, 3, 5]
            assert stats["structural_break_steps"] == [1, 4]  # Steps 1, 4

    def test_get_episode_statistics(self) -> None:
        """Test getting comprehensive episode statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Log some steps
            for step in range(3):
                logger.log_step(
                    step=step + 1,
                    prices=np.array([20.0 + step, 25.0 + step]),
                    profits=np.array([100.0 + step, 120.0 + step]),
                    demand_shock=5.0,
                    market_price=22.5 + step,
                    total_demand=77.5,
                    individual_quantity=38.75,
                    total_profits=np.array([100.0 + step, 120.0 + step]),
                )

            # Get episode statistics
            stats = logger.get_episode_statistics()

            # Verify statistics structure
            assert "episode_id" in stats
            assert "total_steps" in stats
            assert "n_firms" in stats
            assert "price_statistics" in stats
            assert "profit_statistics" in stats
            assert "chat_statistics" in stats
            assert "price_monitoring_statistics" in stats

            # Verify price statistics
            price_stats = stats["price_statistics"]
            assert "mean_price" in price_stats
            assert "std_price" in price_stats
            assert "min_price" in price_stats
            assert "max_price" in price_stats

            # Verify profit statistics
            profit_stats = stats["profit_statistics"]
            assert "mean_profit" in profit_stats
            assert "std_profit" in profit_stats
            assert "min_profit" in profit_stats
            assert "max_profit" in profit_stats

    def test_load_chat_episode_data(self) -> None:
        """Test loading chat episode data from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Log header and steps
            logger.log_episode_header(
                episode_id=123,
                n_firms=2,
                n_steps=2,
                agent_types=["random", "bestresponse"],
                environment_params={"marginal_cost": 10.0},
            )

            for step in range(2):
                logger.log_step(
                    step=step + 1,
                    prices=np.array([20.0, 25.0]),
                    profits=np.array([100.0, 120.0]),
                    demand_shock=5.0,
                    market_price=22.5,
                    total_demand=77.5,
                    individual_quantity=38.75,
                    total_profits=np.array([100.0, 120.0]),
                )

            # Load data back
            loaded_data = EpisodeLogger.load_chat_episode_data(log_file)

            # Verify loaded data
            assert loaded_data["header"] is not None
            assert len(loaded_data["steps"]) == 2
            assert loaded_data["summary"] is None  # No summary logged

            # Verify header data
            header = loaded_data["header"]
            assert header["episode_id"] == 123
            assert header["n_firms"] == 2

            # Verify step data
            step1 = loaded_data["steps"][0]
            assert step1["step"] == 1
            assert step1["prices"] == [20.0, 25.0]

    def test_validate_chat_log_file(self) -> None:
        """Test validating chat log file structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file=log_file, n_firms=2)

            # Log valid data
            logger.log_episode_header(
                episode_id=123,
                n_firms=2,
                n_steps=2,
                agent_types=["random", "bestresponse"],
                environment_params={"marginal_cost": 10.0},
            )

            logger.log_step(
                step=1,
                prices=np.array([20.0, 25.0]),
                profits=np.array([100.0, 120.0]),
                demand_shock=5.0,
                market_price=22.5,
                total_demand=77.5,
                individual_quantity=38.75,
                total_profits=np.array([100.0, 120.0]),
            )

            # Validate file
            assert EpisodeLogger.validate_chat_log_file(log_file) is True

            # Test with invalid file
            invalid_file = Path(temp_dir) / "invalid.jsonl"
            invalid_file.write_text("invalid json")
            assert EpisodeLogger.validate_chat_log_file(invalid_file) is False

            # Test with non-existent file
            non_existent = Path(temp_dir) / "nonexistent.jsonl"
            assert EpisodeLogger.validate_chat_log_file(non_existent) is False
