"""
Structured data logger for CartelEnv episodes.

This module implements a Logger class that records step-by-step data from
CartelEnv episodes and saves it to JSONL files for analysis and reproducibility.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class Logger:
    """
    Structured data logger for CartelEnv episodes.

    This class provides functionality to record and persist step-by-step data
    from CartelEnv episodes, including prices, profits, demand shocks, and
    regulator flags. Data is saved in JSONL format for easy analysis.
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        episode_id: Optional[str] = None,
        n_firms: int = 3,
    ) -> None:
        """
        Initialize the Logger.

        Args:
            log_dir: Directory to save log files
            episode_id: Unique identifier for this episode (auto-generated if None)
            n_firms: Number of firms in the environment
        """
        self.log_dir = Path(log_dir)
        self.n_firms = n_firms

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate episode ID if not provided
        if episode_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_id = f"episode_{timestamp}"

        self.episode_id = episode_id
        self.log_file = self.log_dir / f"{episode_id}.jsonl"

        # Initialize episode metadata
        self.episode_start_time = datetime.now()
        self.step_count = 0
        self.episode_data: List[Dict[str, Any]] = []

        # Write episode header
        self._write_episode_header()

    def _write_episode_header(self) -> None:
        """Write episode metadata header to the log file."""
        header = {
            "type": "episode_header",
            "episode_id": self.episode_id,
            "start_time": self.episode_start_time.isoformat(),
            "n_firms": self.n_firms,
            "log_file": str(self.log_file),
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
    ) -> None:
        """
        Log data for a single step.

        Args:
            step: Current step number
            prices: Array of prices chosen by each firm
            profits: Array of profits for each firm in this step
            demand_shock: Current demand shock value
            market_price: Average market price
            total_demand: Total market demand
            individual_quantity: Quantity allocated to each firm
            total_profits: Cumulative profits for each firm
            regulator_flags: Optional regulator detection flags
            additional_info: Optional additional information
        """
        # Validate inputs
        if len(prices) != self.n_firms:
            raise ValueError(f"Prices array must have length {self.n_firms}")
        if len(profits) != self.n_firms:
            raise ValueError(f"Profits array must have length {self.n_firms}")
        if len(total_profits) != self.n_firms:
            raise ValueError(f"Total profits array must have length {self.n_firms}")

        # Create step data record
        step_data = {
            "type": "step",
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "prices": [float(p) for p in prices.tolist()],
            "profits": [float(p) for p in profits.tolist()],
            "demand_shock": float(demand_shock),
            "market_price": float(market_price),
            "total_demand": float(total_demand),
            "individual_quantity": float(individual_quantity),
            "total_profits": [float(p) for p in total_profits.tolist()],
        }

        # Add regulator flags if provided
        if regulator_flags is not None:
            step_data["regulator_flags"] = regulator_flags

        # Add additional info if provided
        if additional_info is not None:
            step_data["additional_info"] = additional_info

        # Store in memory and write to file
        self.episode_data.append(step_data)
        self.step_count = step

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_data) + "\n")

    def log_episode_end(
        self,
        terminated: bool = False,
        truncated: bool = False,
        final_rewards: Optional[np.ndarray] = None,
        episode_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log episode termination information.

        Args:
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated by max steps
            final_rewards: Final reward values for each firm
            episode_summary: Optional summary statistics
        """
        end_time = datetime.now()
        duration = (end_time - self.episode_start_time).total_seconds()

        end_data = {
            "type": "episode_end",
            "episode_id": self.episode_id,
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_steps": self.step_count,
            "terminated": terminated,
            "truncated": truncated,
        }

        if final_rewards is not None:
            if len(final_rewards) != self.n_firms:
                raise ValueError(f"Final rewards array must have length {self.n_firms}")
            end_data["final_rewards"] = [float(r) for r in final_rewards.tolist()]

        if episode_summary is not None:
            end_data["episode_summary"] = episode_summary

        # Write to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(end_data) + "\n")

    def get_episode_data(self) -> List[Dict[str, Any]]:
        """
        Get all episode data from memory.

        Returns:
            List of all step data records
        """
        return self.episode_data.copy()

    def get_log_file_path(self) -> Path:
        """
        Get the path to the log file.

        Returns:
            Path to the JSONL log file
        """
        return self.log_file

    def close(self) -> None:
        """Close the logger and ensure all data is written."""
        # Data is written immediately, so this is mainly for cleanup
        pass

    @classmethod
    def load_episode_data(cls, log_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Load episode data from a JSONL log file.

        Args:
            log_file: Path to the JSONL log file

        Returns:
            Dictionary containing episode header, steps, and end data
        """
        log_file = Path(log_file)

        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")

        episode_data: Dict[str, Any] = {
            "header": None,
            "steps": [],
            "end": None,
        }

        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())

                    if data["type"] == "episode_header":
                        episode_data["header"] = data
                    elif data["type"] == "step":
                        steps_list = episode_data["steps"]
                        if isinstance(steps_list, list):
                            steps_list.append(data)
                    elif data["type"] == "episode_end":
                        episode_data["end"] = data

        return episode_data

    @classmethod
    def validate_log_file(cls, log_file: Union[str, Path]) -> bool:
        """
        Validate that a log file contains required data structure.

        Args:
            log_file: Path to the JSONL log file

        Returns:
            True if log file is valid, False otherwise
        """
        try:
            episode_data = cls.load_episode_data(log_file)

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
                "total_demand",
                "individual_quantity",
                "total_profits",
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
                if len(step_data["total_profits"]) != n_firms:
                    return False

            return True

        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
