"""
Tests for the Logger class.

This module contains comprehensive tests for the structured data logging
functionality, including file creation, data validation, and deterministic behavior.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.logging.logger import Logger


class TestLogger:
    """Test suite for Logger class."""

    def test_logger_initialization(self) -> None:
        """Test that Logger initializes correctly with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, n_firms=3)

            assert logger.n_firms == 3
            assert logger.log_dir == Path(temp_dir)
            assert logger.episode_id is not None
            assert logger.step_count == 0
            assert logger.log_file.exists()

            # Check that header was written
            with open(logger.log_file, "r") as f:
                header_line = f.readline().strip()
                header = json.loads(header_line)

                assert header["type"] == "episode_header"
                assert header["episode_id"] == logger.episode_id
                assert header["n_firms"] == 3

    def test_logger_initialization_with_custom_episode_id(self) -> None:
        """Test Logger initialization with custom episode ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_id = "test_episode_123"
            logger = Logger(log_dir=temp_dir, episode_id=custom_id, n_firms=2)

            assert logger.episode_id == custom_id
            assert logger.log_file.name == f"{custom_id}.jsonl"

    def test_log_step_basic_functionality(self) -> None:
        """Test basic step logging functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, n_firms=2)

            # Create test data
            prices = np.array([20.0, 25.0], dtype=np.float32)
            profits = np.array([100.0, 120.0], dtype=np.float32)
            total_profits = np.array([100.0, 120.0], dtype=np.float32)

            # Log a step
            logger.log_step(
                step=1,
                prices=prices,
                profits=profits,
                demand_shock=2.5,
                market_price=22.5,
                total_demand=50.0,
                individual_quantity=25.0,
                total_profits=total_profits,
            )

            # Verify data was stored
            assert len(logger.episode_data) == 1
            assert logger.step_count == 1

            step_data = logger.episode_data[0]
            assert step_data["type"] == "step"
            assert step_data["step"] == 1
            assert step_data["prices"] == [20.0, 25.0]
            assert step_data["profits"] == [100.0, 120.0]
            assert step_data["demand_shock"] == 2.5
            assert step_data["market_price"] == 22.5
            assert step_data["total_demand"] == 50.0
            assert step_data["individual_quantity"] == 25.0
            assert step_data["total_profits"] == [100.0, 120.0]

    def test_log_step_with_regulator_flags(self) -> None:
        """Test step logging with regulator flags."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, n_firms=3)

            prices = np.array([30.0, 31.0, 29.0], dtype=np.float32)
            profits = np.array([150.0, 155.0, 145.0], dtype=np.float32)
            total_profits = np.array([150.0, 155.0, 145.0], dtype=np.float32)

            regulator_flags = {
                "parallel_pricing_detected": True,
                "suspicious_threshold": 2.0,
                "investigation_triggered": False,
            }

            logger.log_step(
                step=1,
                prices=prices,
                profits=profits,
                demand_shock=1.0,
                market_price=30.0,
                total_demand=60.0,
                individual_quantity=20.0,
                total_profits=total_profits,
                regulator_flags=regulator_flags,
            )

            step_data = logger.episode_data[0]
            assert "regulator_flags" in step_data
            assert step_data["regulator_flags"] == regulator_flags

    def test_log_step_with_additional_info(self) -> None:
        """Test step logging with additional information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, n_firms=2)

            prices = np.array([15.0, 18.0], dtype=np.float32)
            profits = np.array([75.0, 90.0], dtype=np.float32)
            total_profits = np.array([75.0, 90.0], dtype=np.float32)

            additional_info = {
                "agent_types": ["random", "titfortat"],
                "market_share": [0.45, 0.55],
                "price_variance": 1.5,
            }

            logger.log_step(
                step=1,
                prices=prices,
                profits=profits,
                demand_shock=-1.0,
                market_price=16.5,
                total_demand=40.0,
                individual_quantity=20.0,
                total_profits=total_profits,
                additional_info=additional_info,
            )

            step_data = logger.episode_data[0]
            assert "additional_info" in step_data
            assert step_data["additional_info"] == additional_info

    def test_log_step_validation_errors(self) -> None:
        """Test that log_step raises appropriate validation errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, n_firms=3)

            # Test wrong array lengths
            with pytest.raises(ValueError, match="Prices array must have length 3"):
                logger.log_step(
                    step=1,
                    prices=np.array([1.0, 2.0]),  # Wrong length
                    profits=np.array([10.0, 20.0, 30.0]),
                    demand_shock=0.0,
                    market_price=15.0,
                    total_demand=30.0,
                    individual_quantity=10.0,
                    total_profits=np.array([10.0, 20.0, 30.0]),
                )

            with pytest.raises(ValueError, match="Profits array must have length 3"):
                logger.log_step(
                    step=1,
                    prices=np.array([1.0, 2.0, 3.0]),
                    profits=np.array([10.0, 20.0]),  # Wrong length
                    demand_shock=0.0,
                    market_price=15.0,
                    total_demand=30.0,
                    individual_quantity=10.0,
                    total_profits=np.array([10.0, 20.0, 30.0]),
                )

    def test_log_episode_end(self) -> None:
        """Test episode end logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, n_firms=2)

            # Log some steps first
            for i in range(3):
                logger.log_step(
                    step=i + 1,
                    prices=np.array([20.0, 25.0]),
                    profits=np.array([100.0, 120.0]),
                    demand_shock=1.0,
                    market_price=22.5,
                    total_demand=50.0,
                    individual_quantity=25.0,
                    total_profits=np.array([100.0 * (i + 1), 120.0 * (i + 1)]),
                )

            # Log episode end
            final_rewards = np.array([300.0, 360.0])
            episode_summary = {
                "avg_price": 22.5,
                "total_profit": 660.0,
                "price_volatility": 2.5,
            }

            logger.log_episode_end(
                terminated=False,
                truncated=True,
                final_rewards=final_rewards,
                episode_summary=episode_summary,
            )

            # Verify episode end data was written
            with open(logger.log_file, "r") as f:
                lines = f.readlines()
                end_line = lines[-1].strip()
                end_data = json.loads(end_line)

                assert end_data["type"] == "episode_end"
                assert end_data["episode_id"] == logger.episode_id
                assert end_data["total_steps"] == 3
                assert end_data["terminated"] is False
                assert end_data["truncated"] is True
                assert end_data["final_rewards"] == [300.0, 360.0]
                assert end_data["episode_summary"] == episode_summary
                assert "duration_seconds" in end_data

    def test_log_episode_end_validation_errors(self) -> None:
        """Test that log_episode_end raises appropriate validation errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, n_firms=3)

            with pytest.raises(
                ValueError, match="Final rewards array must have length 3"
            ):
                logger.log_episode_end(
                    final_rewards=np.array([100.0, 200.0]),  # Wrong length
                )

    def test_get_episode_data(self) -> None:
        """Test getting episode data from memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, n_firms=2)

            # Log multiple steps
            for i in range(3):
                logger.log_step(
                    step=i + 1,
                    prices=np.array([20.0 + i, 25.0 + i]),
                    profits=np.array([100.0 + i * 10, 120.0 + i * 10]),
                    demand_shock=float(i),
                    market_price=22.5 + i,
                    total_demand=50.0 + i * 5,
                    individual_quantity=25.0 + i * 2.5,
                    total_profits=np.array([100.0 + i * 10, 120.0 + i * 10]),
                )

            episode_data = logger.get_episode_data()

            assert len(episode_data) == 3
            for i, step_data in enumerate(episode_data):
                assert step_data["step"] == i + 1
                assert step_data["prices"] == [20.0 + i, 25.0 + i]
                assert step_data["demand_shock"] == float(i)

    def test_load_episode_data(self) -> None:
        """Test loading episode data from a log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, n_firms=2, episode_id="test_load")

            # Log some data
            logger.log_step(
                step=1,
                prices=np.array([20.0, 25.0]),
                profits=np.array([100.0, 120.0]),
                demand_shock=2.0,
                market_price=22.5,
                total_demand=50.0,
                individual_quantity=25.0,
                total_profits=np.array([100.0, 120.0]),
            )

            logger.log_episode_end(terminated=True)

            # Load the data
            loaded_data = Logger.load_episode_data(logger.log_file)

            assert loaded_data["header"] is not None
            assert loaded_data["header"]["episode_id"] == "test_load"
            assert loaded_data["header"]["n_firms"] == 2

            assert len(loaded_data["steps"]) == 1
            step_data = loaded_data["steps"][0]
            assert step_data["step"] == 1
            assert step_data["prices"] == [20.0, 25.0]

            assert loaded_data["end"] is not None
            assert loaded_data["end"]["terminated"] is True

    def test_load_episode_data_file_not_found(self) -> None:
        """Test loading episode data from non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_file = Path(temp_dir) / "nonexistent.jsonl"

            with pytest.raises(FileNotFoundError):
                Logger.load_episode_data(non_existent_file)

    def test_validate_log_file_valid(self) -> None:
        """Test log file validation with valid file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, n_firms=3)

            # Log valid data
            logger.log_step(
                step=1,
                prices=np.array([20.0, 25.0, 30.0]),
                profits=np.array([100.0, 120.0, 140.0]),
                demand_shock=1.0,
                market_price=25.0,
                total_demand=60.0,
                individual_quantity=20.0,
                total_profits=np.array([100.0, 120.0, 140.0]),
            )

            logger.log_episode_end()

            assert Logger.validate_log_file(logger.log_file) is True

    def test_validate_log_file_invalid(self) -> None:
        """Test log file validation with invalid file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_file = Path(temp_dir) / "invalid.jsonl"

            # Create invalid JSONL file
            with open(invalid_file, "w") as f:
                f.write("invalid json\n")

            assert Logger.validate_log_file(invalid_file) is False

    def test_validate_log_file_missing_required_keys(self) -> None:
        """Test log file validation with missing required keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            incomplete_file = Path(temp_dir) / "incomplete.jsonl"

            # Create incomplete log file
            with open(incomplete_file, "w") as f:
                header = {
                    "type": "episode_header",
                    "episode_id": "test",
                    "n_firms": 2,
                }
                f.write(json.dumps(header) + "\n")

                # Step missing required keys
                incomplete_step = {
                    "type": "step",
                    "step": 1,
                    "prices": [20.0, 25.0],
                    # Missing profits, demand_shock, etc.
                }
                f.write(json.dumps(incomplete_step) + "\n")

            assert Logger.validate_log_file(incomplete_file) is False

    def test_deterministic_logging_with_seed(self) -> None:
        """Test that logging produces identical results with same seed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create two loggers with same parameters
            logger1 = Logger(log_dir=temp_dir, episode_id="test1", n_firms=2)
            logger2 = Logger(log_dir=temp_dir, episode_id="test2", n_firms=2)

            # Log identical data
            prices = np.array([20.0, 25.0])
            profits = np.array([100.0, 120.0])
            total_profits = np.array([100.0, 120.0])

            for logger in [logger1, logger2]:
                logger.log_step(
                    step=1,
                    prices=prices,
                    profits=profits,
                    demand_shock=2.5,
                    market_price=22.5,
                    total_demand=50.0,
                    individual_quantity=25.0,
                    total_profits=total_profits,
                )
                logger.log_episode_end()

            # Load both files
            data1 = Logger.load_episode_data(logger1.log_file)
            data2 = Logger.load_episode_data(logger2.log_file)

            # Compare step data (excluding timestamps and episode_id)
            step1 = data1["steps"][0]
            step2 = data2["steps"][0]

            # Remove timestamp-dependent fields
            for step in [step1, step2]:
                step.pop("timestamp", None)

            assert step1 == step2

    def test_log_file_creation_and_structure(self) -> None:
        """Test that log files are created with correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, episode_id="structure_test", n_firms=3)

            # Verify file exists
            assert logger.log_file.exists()
            assert logger.log_file.suffix == ".jsonl"

            # Log multiple steps
            for i in range(5):
                logger.log_step(
                    step=i + 1,
                    prices=np.array([20.0 + i, 25.0 + i, 30.0 + i]),
                    profits=np.array([100.0 + i * 10, 120.0 + i * 10, 140.0 + i * 10]),
                    demand_shock=float(i),
                    market_price=25.0 + i,
                    total_demand=60.0 + i * 5,
                    individual_quantity=20.0 + i * 1.67,
                    total_profits=np.array(
                        [100.0 + i * 10, 120.0 + i * 10, 140.0 + i * 10]
                    ),
                )

            logger.log_episode_end()

            # Verify file structure
            with open(logger.log_file, "r") as f:
                lines = f.readlines()

                # Should have header + 5 steps + end = 7 lines
                assert len(lines) == 7

                # Check header
                header = json.loads(lines[0].strip())
                assert header["type"] == "episode_header"
                assert header["episode_id"] == "structure_test"

                # Check steps
                for i in range(5):
                    step = json.loads(lines[i + 1].strip())
                    assert step["type"] == "step"
                    assert step["step"] == i + 1
                    assert len(step["prices"]) == 3
                    assert len(step["profits"]) == 3
                    assert len(step["total_profits"]) == 3

                # Check end
                end = json.loads(lines[6].strip())
                assert end["type"] == "episode_end"
                assert end["total_steps"] == 5

    def test_close_method(self) -> None:
        """Test that close method works without errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = Logger(log_dir=temp_dir, n_firms=2)

            # Should not raise any errors
            logger.close()

            # Should still be able to use logger after close
            logger.log_step(
                step=1,
                prices=np.array([20.0, 25.0]),
                profits=np.array([100.0, 120.0]),
                demand_shock=1.0,
                market_price=22.5,
                total_demand=50.0,
                individual_quantity=25.0,
                total_profits=np.array([100.0, 120.0]),
            )
