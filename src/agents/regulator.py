"""
Regulator agent for monitoring and detecting cartel behavior.

This module implements a Regulator class that monitors episode logs and applies
rule-based screens to detect potential cartel behavior, including parallel pricing
and sudden structural breaks. When violations are detected, the regulator applies
penalties by subtracting fines from firm rewards.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Regulator:
    """
    Regulator agent that monitors firm behavior and detects cartel violations.

    This regulator implements two main detection mechanisms:
    1. Parallel pricing detection: Flags when firms' prices are within ε for ≥k steps
    2. Structural break detection: Flags when prices jump beyond δ threshold

    When violations are detected, the regulator applies fines by reducing firm rewards.
    """

    def __init__(
        self,
        parallel_threshold: float = 2.0,
        parallel_steps: int = 3,
        structural_break_threshold: float = 10.0,
        fine_amount: float = 50.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the regulator with detection parameters.

        Args:
            parallel_threshold: Maximum price difference (ε) to consider prices parallel
            parallel_steps: Minimum consecutive steps (k) for parallel pricing detection
            structural_break_threshold: Maximum price change (δ) before structural break
            fine_amount: Fine to subtract from firm rewards when violations detected
            seed: Random seed for reproducibility
        """
        if parallel_threshold < 0:
            raise ValueError("Parallel threshold must be non-negative")
        if parallel_steps < 1:
            raise ValueError("Parallel steps must be at least 1")
        if structural_break_threshold < 0:
            raise ValueError("Structural break threshold must be non-negative")
        if fine_amount < 0:
            raise ValueError("Fine amount must be non-negative")

        self.parallel_threshold = parallel_threshold
        self.parallel_steps = parallel_steps
        self.structural_break_threshold = structural_break_threshold
        self.fine_amount = fine_amount
        self.np_random = np.random.default_rng(seed)

        # Episode monitoring state
        self.price_history: List[np.ndarray] = []
        self.parallel_violations: List[Tuple[int, str]] = []  # (step, violation_type)
        self.structural_break_violations: List[Tuple[int, str]] = []
        self.total_fines_applied: float = 0.0

    def monitor_step(
        self,
        prices: np.ndarray,
        step: int,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Monitor a single step and detect potential violations.

        Args:
            prices: Array of prices set by each firm
            step: Current step number
            info: Additional information from environment

        Returns:
            Dictionary containing detection results and any penalties
        """
        # Store price history
        self.price_history.append(prices.copy())

        detection_results: Dict[str, Any] = {
            "step": step,
            "parallel_violation": False,
            "structural_break_violation": False,
            "fines_applied": np.zeros(len(prices)),
            "violation_details": [],
        }

        # Check for parallel pricing violation
        parallel_violation = self._detect_parallel_pricing(step)
        if parallel_violation:
            detection_results["parallel_violation"] = True
            detection_results["violation_details"].append(
                f"Parallel pricing detected at step {step}"
            )
            self.parallel_violations.append((step, "parallel_pricing"))

        # Check for structural break violation
        structural_break_violation = self._detect_structural_break(step)
        if structural_break_violation:
            detection_results["structural_break_violation"] = True
            detection_results["violation_details"].append(
                f"Structural break detected at step {step}"
            )
            self.structural_break_violations.append((step, "structural_break"))

        # Apply fines if violations detected
        if parallel_violation or structural_break_violation:
            detection_results["fines_applied"] = np.full(len(prices), self.fine_amount)
            self.total_fines_applied += len(prices) * self.fine_amount

        return detection_results

    def _detect_parallel_pricing(self, step: int) -> bool:
        """
        Detect if firms are engaging in parallel pricing behavior.

        Parallel pricing is detected when all firms' prices are within the threshold
        for at least the minimum number of consecutive steps.

        Args:
            step: Current step number

        Returns:
            True if parallel pricing violation detected, False otherwise
        """
        # Need at least parallel_steps of history
        if len(self.price_history) < self.parallel_steps:
            return False

        # Check the last parallel_steps steps
        recent_prices = self.price_history[-self.parallel_steps :]

        # For each step, check if all prices are within threshold
        for prices in recent_prices:
            if len(prices) < 2:
                return False  # Need at least 2 firms

            # Check if all prices are within threshold of each other
            max_price = np.max(prices)
            min_price = np.min(prices)
            price_range = max_price - min_price

            if price_range > self.parallel_threshold:
                return False  # Prices too spread out

        return True  # All recent steps show parallel pricing

    def _detect_structural_break(self, step: int) -> bool:
        """
        Detect sudden structural breaks in pricing behavior.

        A structural break is detected when any firm's price changes by more than
        the threshold from the previous step.

        Args:
            step: Current step number

        Returns:
            True if structural break violation detected, False otherwise
        """
        # Need at least 2 steps of history
        if len(self.price_history) < 2:
            return False

        current_prices = self.price_history[-1]
        previous_prices = self.price_history[-2]

        # Check if any firm had a price change exceeding threshold
        price_changes = np.abs(current_prices - previous_prices)
        max_change = np.max(price_changes)

        return bool(max_change > self.structural_break_threshold)

    def apply_penalties(
        self, rewards: np.ndarray, detection_results: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply penalties to firm rewards based on detected violations.

        Args:
            rewards: Original rewards for each firm
            detection_results: Results from monitor_step

        Returns:
            Modified rewards with penalties applied
        """
        penalties: np.ndarray = detection_results["fines_applied"]
        modified_rewards: np.ndarray = rewards - penalties

        # Ensure rewards don't go below zero
        modified_rewards = np.maximum(modified_rewards, 0.0)

        return modified_rewards.astype(np.float32)

    def get_violation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all violations detected during the episode.

        Returns:
            Dictionary containing violation statistics
        """
        return {
            "total_parallel_violations": len(self.parallel_violations),
            "total_structural_break_violations": len(self.structural_break_violations),
            "total_fines_applied": self.total_fines_applied,
            "parallel_violation_steps": [step for step, _ in self.parallel_violations],
            "structural_break_steps": [
                step for step, _ in self.structural_break_violations
            ],
        }

    def reset(self) -> None:
        """Reset the regulator's monitoring state for a new episode."""
        self.price_history.clear()
        self.parallel_violations.clear()
        self.structural_break_violations.clear()
        self.total_fines_applied = 0.0

    def get_price_statistics(self) -> Dict[str, float]:
        """
        Get statistical summary of observed prices.

        Returns:
            Dictionary containing price statistics
        """
        if not self.price_history:
            return {}

        all_prices = np.concatenate(self.price_history)
        return {
            "mean_price": float(np.mean(all_prices)),
            "std_price": float(np.std(all_prices)),
            "min_price": float(np.min(all_prices)),
            "max_price": float(np.max(all_prices)),
            "price_range": float(np.max(all_prices) - np.min(all_prices)),
        }

    def get_parallel_pricing_ratio(self) -> float:
        """
        Calculate the ratio of steps with parallel pricing behavior.

        Returns:
            Ratio of steps with parallel pricing (0.0 to 1.0)
        """
        if len(self.price_history) < self.parallel_steps:
            return 0.0

        parallel_steps_count = 0
        for step in range(self.parallel_steps, len(self.price_history) + 1):
            if self._detect_parallel_pricing(step):
                parallel_steps_count += 1

        total_checkable_steps = len(self.price_history) - self.parallel_steps + 1
        return (
            parallel_steps_count / total_checkable_steps
            if total_checkable_steps > 0
            else 0.0
        )
