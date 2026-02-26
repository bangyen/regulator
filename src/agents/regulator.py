"""
Regulator agent for monitoring and detecting cartel behavior.

This module implements a Regulator class that monitors episode logs and applies
rule-based screens to detect potential cartel behavior, including parallel pricing
and sudden structural breaks. When violations are detected, the regulator applies
penalties by subtracting fines from firm rewards.
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

from src.agents.leniency import LeniencyProgram, LeniencyStatus


class Regulator:
    """
    Regulator agent that monitors firm behavior and detects cartel violations.
    """

    def __init__(
        self,
        parallel_threshold: float = 5.0,
        parallel_steps: int = 4,
        structural_break_threshold: float = 30.0,
        fine_amount: float = 25.0,
        leniency_enabled: bool = True,
        leniency_reduction: float = 0.5,
        seed: Optional[int] = None,
        history_maxlen: int = 20,
    ) -> None:
        """
        Initialize the regulator with detection parameters.
        """
        self.parallel_threshold = parallel_threshold
        self.parallel_steps = parallel_steps
        self.structural_break_threshold = structural_break_threshold
        self.fine_amount = fine_amount
        self.leniency_enabled = leniency_enabled
        self.leniency_reduction = leniency_reduction
        self.np_random = np.random.default_rng(seed)

        # Episode monitoring state
        self.price_history: deque[np.ndarray] = deque(maxlen=history_maxlen)
        self.profit_history: deque[np.ndarray] = deque(maxlen=history_maxlen)
        self.parallel_violations: List[Tuple[int, str]] = []
        self.structural_break_violations: List[Tuple[int, str]] = []
        self.total_fines_applied: float = 0.0
        self.violation_start_step: Optional[int] = None
        self.consecutive_violation_steps: int = 0

        # Leniency program
        self.leniency_program: Optional[LeniencyProgram] = None
        if self.leniency_enabled:
            self.leniency_program = LeniencyProgram(
                leniency_reduction=leniency_reduction, seed=seed
            )

    def monitor_step(
        self,
        prices: np.ndarray,
        step: int,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Monitor a single step and detect potential violations.
        """
        self.price_history.append(prices.copy())

        detection_results: Dict[str, Any] = {
            "step": step,
            "parallel_violation": False,
            "structural_break_violation": False,
            "fines_applied": np.zeros(len(prices)),
            "violation_details": [],
            "prices": prices.copy(),  # Store prices for later fine calculation
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

        # Track violation duration (will calculate fines in apply_penalties)
        if parallel_violation or structural_break_violation:
            if self.violation_start_step is None:
                self.violation_start_step = step
                self.consecutive_violation_steps = 1
            else:
                self.consecutive_violation_steps += 1

            detection_results["violation_detected"] = True
            detection_results["consecutive_violations"] = (
                self.consecutive_violation_steps
            )
        else:
            # No violation this step - reset consecutive counter
            self.violation_start_step = None
            self.consecutive_violation_steps = 0
            detection_results["violation_detected"] = False

        return detection_results

    def _detect_parallel_pricing(self, step: int) -> bool:
        """
        Detect if firms are engaging in parallel pricing behavior.
        """
        if len(self.price_history) < self.parallel_steps:
            return False

        # Get last k steps
        history = list(self.price_history)[-self.parallel_steps :]

        for prices in history:
            if len(prices) < 2:
                return False
            if (np.max(prices) - np.min(prices)) > self.parallel_threshold:
                return False
        return True

    def _detect_structural_break(self, step: int) -> bool:
        """
        Detect sudden structural breaks in pricing behavior.
        """
        if len(self.price_history) < 2:
            return False

        current = self.price_history[-1]
        previous = self.price_history[-2]
        return bool(
            np.max(np.abs(current - previous)) > self.structural_break_threshold
        )

    def apply_penalties(
        self, rewards: np.ndarray, detection_results: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply simplified penalties to firm rewards.
        """
        self.profit_history.append(rewards.copy())
        penalties = np.zeros(len(rewards))

        if detection_results.get("violation_detected", False):
            # Simple constant fine + duration-based escalator
            duration_multiplier = (
                1.0 + (detection_results.get("consecutive_violations", 1) - 1) * 0.1
            )
            base_penalty = self.fine_amount * duration_multiplier
            penalties = np.full(len(rewards), base_penalty)

            # Apply leniency if enabled
            if self.leniency_program is not None:
                for i in range(len(rewards)):
                    reduction = self.leniency_program.get_fine_reduction(i)
                    penalties[i] *= 1.0 - reduction

                # Update evidence
                evidence = 0.8 if detection_results.get("parallel_violation") else 0.5
                for i in range(len(rewards)):
                    self.leniency_program.update_collusion_evidence(
                        i, evidence, detection_results["step"]
                    )

            self.total_fines_applied += float(np.sum(penalties))

        from typing import cast

        detection_results["fines_applied"] = penalties
        return cast(np.ndarray, (rewards - penalties).astype(np.float32))

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

    def reset(self, n_firms: Optional[int] = None) -> None:
        """
        Reset the regulator's monitoring state for a new episode.

        Args:
            n_firms: Number of firms (required if leniency program is enabled)
        """
        self.price_history.clear()
        self.parallel_violations.clear()
        self.structural_break_violations.clear()
        self.total_fines_applied = 0.0
        self.profit_history.clear()
        self.violation_start_step = None
        self.consecutive_violation_steps = 0

        if self.leniency_program is not None:
            if n_firms is None:
                raise ValueError(
                    "n_firms must be provided when leniency program is enabled"
                )
            self.leniency_program.reset(n_firms)

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

    def submit_leniency_report(
        self,
        firm_id: int,
        reported_firms: List[int],
        evidence_strength: float,
        step: int,
    ) -> bool:
        """
        Submit a leniency report from a firm.

        Args:
            firm_id: ID of the reporting firm
            reported_firms: List of firm IDs being reported
            evidence_strength: Strength of evidence provided
            step: Current step number

        Returns:
            True if report was accepted, False otherwise
        """
        if self.leniency_program is None:
            return False

        result: bool = self.leniency_program.submit_report(
            firm_id, reported_firms, evidence_strength, step
        )
        return result

    def get_leniency_status(self, firm_id: int) -> LeniencyStatus:
        """
        Get the leniency status for a firm.

        Args:
            firm_id: ID of the firm

        Returns:
            Current leniency status
        """
        if self.leniency_program is None:
            return LeniencyStatus.NOT_APPLICABLE

        return self.leniency_program.firm_status.get(
            firm_id, LeniencyStatus.NOT_APPLICABLE
        )

    def get_whistleblower_incentive(
        self,
        firm_id: int,
        current_fine: float,
        collusion_probability: float,
    ) -> float:
        """
        Get the whistleblower incentive for a firm.

        Args:
            firm_id: ID of the firm
            current_fine: Current fine amount if caught
            collusion_probability: Probability of being caught in collusion

        Returns:
            Expected benefit from whistleblowing
        """
        if self.leniency_program is None:
            return 0.0

        result: float = self.leniency_program.get_whistleblower_incentive(
            firm_id, current_fine, collusion_probability
        )
        return result

    def get_leniency_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the leniency program's current state.

        Returns:
            Dictionary containing leniency program statistics
        """
        if self.leniency_program is None:
            return {"leniency_enabled": False}

        summary: Dict[str, Any] = self.leniency_program.get_program_summary()
        summary["leniency_enabled"] = True
        return summary

    def get_leniency_reports(self) -> List[Dict[str, Any]]:
        """
        Get a summary of all leniency reports.

        Returns:
            List of dictionaries containing report details
        """
        if self.leniency_program is None:
            return []

        result: List[Dict[str, Any]] = self.leniency_program.get_reports_summary()
        return result
