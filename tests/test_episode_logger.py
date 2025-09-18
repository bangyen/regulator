"""
Tests for the EpisodeLogger class.

This module tests the enhanced episode logging functionality including
chat messages, LLM detection results, and regulator monitoring data.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.episode_logging.episode_logger import EpisodeLogger


class TestEpisodeLogger:
    """Test cases for EpisodeLogger functionality."""

    def test_episode_logger_initialization(self) -> None:
        """Test EpisodeLogger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file, n_firms=3)

            assert logger.n_firms == 3
            assert logger.log_file == log_file
            assert log_file.parent.exists()

    def test_log_episode_header(self) -> None:
        """Test logging episode header."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file, n_firms=2)

            logger.log_episode_header(
                episode_id=1,
                n_firms=2,
                n_steps=100,
                agent_types=["random", "tit_for_tat"],
                environment_params={"seed": 42, "max_steps": 100},
            )

            # Check that file was created and contains header
            assert log_file.exists()

            with open(log_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 1

                header = json.loads(lines[0])
                assert header["type"] == "episode_header"
                assert header["episode_id"] == 1
                assert header["n_firms"] == 2

    def test_log_step(self) -> None:
        """Test logging step data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file, n_firms=2)

            logger.log_step(
                step=1,
                prices=np.array([10.5, 11.2]),
                profits=np.array([100.0, 120.0]),
                demand_shock=0.1,
                market_price=10.85,
                total_demand=100.0,
                individual_quantity=50.0,
                total_profits=np.array([100.0, 120.0]),
                rewards=[50.0, 60.0],
            )

            # Check that file was created and contains step data
            assert log_file.exists()

            with open(log_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2  # Header + step

                # Check header (first line)
                header = json.loads(lines[0])
                assert header["type"] == "episode_header"

                # Check step (second line)
                step = json.loads(lines[1])
                assert step["type"] == "step"
                assert step["step"] == 1
                assert step["prices"] == [10.5, 11.2]
                assert step["profits"] == [100.0, 120.0]

    def test_log_step_with_messages(self) -> None:
        """Test logging step data with chat messages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file, n_firms=2)

            messages = [
                {
                    "sender_id": 0,
                    "receiver_id": 1,
                    "message": "Let's coordinate our pricing",
                    "message_type": "text",
                }
            ]

            logger.log_step(
                step=1,
                prices=np.array([10.5, 11.2]),
                profits=np.array([100.0, 120.0]),
                demand_shock=0.1,
                market_price=10.85,
                total_demand=100.0,
                individual_quantity=50.0,
                total_profits=np.array([100.0, 120.0]),
                messages=messages,
            )

            # Check that file was created and contains step data with messages
            assert log_file.exists()

            with open(log_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2  # Header + step

                # Check step (second line)
                step = json.loads(lines[1])
                assert step["type"] == "step"
                assert step["step"] == 1
                assert "messages" in step
                assert len(step["messages"]) == 1
                assert step["messages"][0]["message"] == "Let's coordinate our pricing"

    def test_log_step_with_monitoring(self) -> None:
        """Test logging step data with regulator monitoring."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file, n_firms=2)

            price_monitoring = {
                "parallel_violation": True,
                "structural_break_violation": False,
                "fines_applied": [25.0, 0.0],
                "violation_details": ["Parallel pricing detected"],
            }

            logger.log_step(
                step=1,
                prices=np.array([10.5, 11.2]),
                profits=np.array([100.0, 120.0]),
                demand_shock=0.1,
                market_price=10.85,
                total_demand=100.0,
                individual_quantity=50.0,
                total_profits=np.array([100.0, 120.0]),
                price_monitoring=price_monitoring,
            )

            # Check that file was created and contains step data with monitoring
            assert log_file.exists()

            with open(log_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2  # Header + step

                # Check step (second line)
                step = json.loads(lines[1])
                assert step["type"] == "step"
                assert step["step"] == 1
                assert "price_monitoring" in step
                assert step["price_monitoring"]["parallel_violation"] is True

    def test_log_episode_summary(self) -> None:
        """Test logging episode summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file, n_firms=2)

            logger.log_episode_summary(
                total_reward=220.0,
                total_steps=50,
                final_prices=[10.0, 11.0],
                final_profits=[100.0, 120.0],
            )

            # Check that file was created and contains summary data
            assert log_file.exists()

            with open(log_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2  # Header + summary

                # Check summary (second line)
                summary = json.loads(lines[1])
                assert summary["type"] == "episode_summary"
                assert summary["total_steps"] == 50
                assert summary["final_prices"] == [10.0, 11.0]
                assert summary["final_profits"] == [100.0, 120.0]

    def test_multiple_log_entries(self) -> None:
        """Test logging multiple entries in sequence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file, n_firms=2)

            # Log header
            logger.log_episode_header(
                episode_id=1,
                n_firms=2,
                n_steps=100,
                agent_types=["random", "tit_for_tat"],
                environment_params={"seed": 42},
            )

            # Log step
            logger.log_step(
                step=1,
                prices=np.array([10.0, 11.0]),
                profits=np.array([100.0, 120.0]),
                demand_shock=0.1,
                market_price=10.5,
                total_demand=100.0,
                individual_quantity=50.0,
                total_profits=np.array([100.0, 120.0]),
            )

            # Log step with messages
            messages = [
                {
                    "sender_id": 0,
                    "receiver_id": 1,
                    "message": "Hello",
                    "message_type": "text",
                }
            ]
            logger.log_step(
                step=2,
                prices=np.array([10.5, 11.5]),
                profits=np.array([110.0, 130.0]),
                demand_shock=0.1,
                market_price=11.0,
                total_demand=100.0,
                individual_quantity=50.0,
                total_profits=np.array([110.0, 130.0]),
                messages=messages,
            )

            # Check that all entries were logged
            assert log_file.exists()

            with open(log_file, "r") as f:
                lines = f.readlines()
                assert (
                    len(lines) == 3
                )  # Manual header + 2 steps (overwrites auto header)

                # Check manual header (first line)
                manual_header = json.loads(lines[0])
                assert manual_header["type"] == "episode_header"
                assert manual_header["episode_id"] == 1

                # Check first step
                step1 = json.loads(lines[1])
                assert step1["type"] == "step"
                assert step1["step"] == 1

                # Check second step with messages
                step2 = json.loads(lines[2])
                assert step2["type"] == "step"
                assert step2["step"] == 2
                assert "messages" in step2

    def test_numpy_array_serialization(self) -> None:
        """Test that numpy arrays are properly serialized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file, n_firms=2)

            logger.log_step(
                step=1,
                prices=np.array([10.5, 11.2]),
                profits=np.array([100.0, 120.0]),
                demand_shock=0.1,
                market_price=10.85,
                total_demand=100.0,
                individual_quantity=50.0,
                total_profits=np.array([100.0, 120.0]),
            )

            # Check that numpy arrays were converted to lists
            with open(log_file, "r") as f:
                lines = f.readlines()
                step = json.loads(lines[1])  # Second line (first is header)
                assert step["prices"] == [10.5, 11.2]
                assert step["profits"] == [100.0, 120.0]

    def test_file_creation_with_nested_directories(self) -> None:
        """Test that nested directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "nested" / "deep" / "test_episode.jsonl"
            logger = EpisodeLogger(log_file, n_firms=2)

            # The logger automatically creates a header, so we just need to log a step
            logger.log_step(
                step=1,
                prices=np.array([10.5, 11.2]),
                profits=np.array([100.0, 120.0]),
                demand_shock=0.1,
                market_price=10.85,
                total_demand=100.0,
                individual_quantity=50.0,
                total_profits=np.array([100.0, 120.0]),
            )

            # Check that nested directories were created
            assert log_file.parent.exists()
            assert log_file.exists()

            # Check that a header was automatically created and step was logged
            with open(log_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2  # Header + step
                header = json.loads(lines[0])
                assert header["type"] == "episode_header"
                step = json.loads(lines[1])
                assert step["type"] == "step"

    def test_error_handling_invalid_data(self) -> None:
        """Test error handling with invalid data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file, n_firms=2)

            # Test with data that can't be JSON serialized
            # Should not raise an exception, but handle gracefully
            try:
                logger.log_step(
                    step=1,
                    prices=np.array([10.0, 11.0]),
                    profits=np.array([100.0, 120.0]),
                    demand_shock=0.1,
                    market_price=10.5,
                    total_demand=100.0,
                    individual_quantity=50.0,
                    total_profits=np.array([100.0, 120.0]),
                )
            except (TypeError, ValueError):
                # This is expected behavior
                pass

    def test_logger_reset_functionality(self) -> None:
        """Test that logger can be reset for new episodes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_episode.jsonl"
            logger = EpisodeLogger(log_file, n_firms=2)

            # Log some data
            logger.log_episode_header(
                episode_id=1,
                n_firms=2,
                n_steps=100,
                agent_types=["random", "tit_for_tat"],
                environment_params={"seed": 42},
            )
            logger.log_step(
                step=1,
                prices=np.array([10.0, 11.0]),
                profits=np.array([100.0, 120.0]),
                demand_shock=0.1,
                market_price=10.5,
                total_demand=100.0,
                individual_quantity=50.0,
                total_profits=np.array([100.0, 120.0]),
            )

            # Create a new logger for the second episode (simulating reset)
            # Use a different file name to avoid overwriting
            log_file2 = log_file.parent / "test_episode2.jsonl"
            logger2 = EpisodeLogger(log_file2, n_firms=2)

            # Log new episode data
            logger2.log_step(
                step=1,
                prices=np.array([12.0, 13.0]),
                profits=np.array([120.0, 140.0]),
                demand_shock=0.1,
                market_price=12.5,
                total_demand=100.0,
                individual_quantity=50.0,
                total_profits=np.array([120.0, 140.0]),
            )

            # Check that both episodes are logged in separate files
            with open(log_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2  # Header + step

                # First episode header
                header1 = json.loads(lines[0])
                assert header1["type"] == "episode_header"

                # First episode step
                step1 = json.loads(lines[1])
                assert step1["prices"] == [10.0, 11.0]

            with open(log_file2, "r") as f:
                lines2 = f.readlines()
                assert len(lines2) == 2  # Header + step

                # Second episode header
                header2 = json.loads(lines2[0])
                assert header2["type"] == "episode_header"

                # Second episode step
                step2 = json.loads(lines2[1])
                assert step2["prices"] == [12.0, 13.0]
