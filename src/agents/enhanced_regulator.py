"""
Enhanced Regulator with Graduated Penalties and Continuous Monitoring

This module implements an enhanced regulator that provides more sophisticated
monitoring with graduated penalties, continuous risk scores, and market-aware
detection mechanisms.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from enum import Enum

from .regulator import Regulator


class ViolationSeverity(Enum):
    """Violation severity levels for graduated penalties."""

    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class EnhancedRegulator(Regulator):
    """
    Enhanced regulator with graduated penalties and continuous monitoring scores.

    This regulator provides:
    1. Graduated penalty system based on violation severity
    2. Continuous risk scores (0.0-1.0) instead of binary violations
    3. Market condition awareness
    4. Cumulative penalty tracking
    5. Dynamic threshold adjustment
    """

    def __init__(
        self,
        parallel_threshold: float = 5.0,
        parallel_steps: int = 4,
        structural_break_threshold: float = 30.0,
        base_fine_amount: float = 25.0,
        leniency_enabled: bool = True,
        leniency_reduction: float = 0.5,
        # Enhanced parameters
        use_graduated_penalties: bool = True,
        use_market_awareness: bool = True,
        cumulative_penalty_multiplier: float = 1.2,  # Penalty increases with repeat violations
        max_penalty_multiplier: float = 5.0,  # Maximum penalty multiplier
        market_volatility_threshold: float = 0.3,  # High volatility threshold
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the enhanced regulator.

        Args:
            parallel_threshold: Price difference threshold for parallel pricing
            parallel_steps: Steps required for parallel pricing detection
            structural_break_threshold: Price change threshold for structural breaks
            base_fine_amount: Base fine amount for violations
            leniency_enabled: Whether leniency program is enabled
            leniency_reduction: Fine reduction for leniency participants
            use_graduated_penalties: Whether to use graduated penalty system
            use_continuous_scores: Whether to use continuous risk scores
            use_market_awareness: Whether to adjust thresholds based on market conditions
            cumulative_penalty_multiplier: Multiplier for repeat violations
            max_penalty_multiplier: Maximum penalty multiplier
            market_volatility_threshold: Threshold for high market volatility
            seed: Random seed for reproducibility
        """
        super().__init__(
            parallel_threshold=parallel_threshold,
            parallel_steps=parallel_steps,
            structural_break_threshold=structural_break_threshold,
            fine_amount=base_fine_amount,
            leniency_enabled=leniency_enabled,
            leniency_reduction=leniency_reduction,
            seed=seed,
        )

        self.use_graduated_penalties = use_graduated_penalties
        self.use_market_awareness = use_market_awareness
        self.cumulative_penalty_multiplier = cumulative_penalty_multiplier
        self.max_penalty_multiplier = max_penalty_multiplier
        self.market_volatility_threshold = market_volatility_threshold

        # Enhanced monitoring state
        self.violation_counts: Dict[int, int] = {}  # firm_id -> violation_count
        self.market_volatility_history: List[float] = []
        self.penalty_multipliers: Dict[int, float] = {}  # firm_id -> current_multiplier

        # Graduated penalty structure
        self.penalty_structure = {
            ViolationSeverity.MINOR: 0.5,  # 50% of base fine
            ViolationSeverity.MODERATE: 1.0,  # 100% of base fine
            ViolationSeverity.SEVERE: 2.0,  # 200% of base fine
            ViolationSeverity.CRITICAL: 4.0,  # 400% of base fine
        }

    def monitor_step(
        self,
        prices: np.ndarray,
        step: int,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced monitoring with continuous scores and graduated penalties.

        Args:
            prices: Array of prices set by each firm
            step: Current step number
            info: Additional information from environment

        Returns:
            Dictionary containing enhanced detection results
        """
        # Store price history
        self.price_history.append(prices.copy())

        # Calculate market volatility
        market_volatility = self._calculate_market_volatility()
        self.market_volatility_history.append(market_volatility)

        # Enhanced detection results
        detection_results: Dict[str, Any] = {
            "step": step,
            "parallel_violation": False,
            "structural_break_violation": False,
            "fines_applied": np.zeros(len(prices)).tolist(),
            "violation_details": [],
            "market_volatility": float(market_volatility),
            "violation_severities": [],
            "penalty_multipliers": [],
        }

        # Check for violations with enhanced thresholds
        parallel_violation = self._detect_parallel_pricing_enhanced(
            step, market_volatility
        )
        structural_break_violation = self._detect_structural_break_enhanced(
            step, market_volatility
        )

        if parallel_violation or structural_break_violation:
            detection_results["parallel_violation"] = parallel_violation
            detection_results["structural_break_violation"] = structural_break_violation

            # Calculate graduated penalties
            fines, severities, multipliers = self._calculate_graduated_penalties(
                prices, parallel_violation, structural_break_violation
            )

            detection_results["fines_applied"] = fines.tolist()
            detection_results["violation_severities"] = [
                s.value for s in severities
            ]  # Convert enum to string
            detection_results["penalty_multipliers"] = [float(m) for m in multipliers]

            # Update violation tracking
            for i in range(len(prices)):
                if fines[i] > 0:
                    self.violation_counts[i] = self.violation_counts.get(i, 0) + 1
                    self.penalty_multipliers[i] = min(
                        self.max_penalty_multiplier,
                        self.penalty_multipliers.get(i, 1.0)
                        * self.cumulative_penalty_multiplier,
                    )

            # Add violation details
            if parallel_violation:
                detection_results["violation_details"].append(
                    f"Parallel pricing detected at step {step}"
                )
            if structural_break_violation:
                detection_results["violation_details"].append(
                    f"Structural break detected at step {step}"
                )

        return detection_results

    def _calculate_market_volatility(self) -> float:
        """Calculate current market volatility based on price history."""
        if len(self.price_history) < 2:
            return 0.0

        # Use recent price history (last 5 steps or all available)
        recent_prices = (
            self.price_history[-5:]
            if len(self.price_history) >= 5
            else self.price_history
        )

        # Calculate volatility as coefficient of variation
        all_prices = np.concatenate(recent_prices)
        if len(all_prices) == 0:
            return 0.0

        mean_price = np.mean(all_prices)
        std_price = np.std(all_prices)

        return float(std_price / (mean_price + 1e-6))  # Avoid division by zero

    def _detect_parallel_pricing_enhanced(
        self, step: int, market_volatility: float
    ) -> bool:
        """Enhanced parallel pricing detection with market awareness."""
        if len(self.price_history) < self.parallel_steps:
            return False

        # Adjust threshold based on market volatility
        adjusted_threshold = self.parallel_threshold
        if self.use_market_awareness:
            # Higher volatility = more lenient threshold
            volatility_factor = 1.0 + market_volatility
            adjusted_threshold *= volatility_factor

        # Check recent steps for parallel pricing
        recent_prices = self.price_history[-self.parallel_steps :]

        for prices in recent_prices:
            if len(prices) > 1:
                price_std = np.std(prices)
                if price_std > adjusted_threshold:
                    return False  # Not parallel enough

        return True

    def _detect_structural_break_enhanced(
        self, step: int, market_volatility: float
    ) -> bool:
        """Enhanced structural break detection with market awareness."""
        if len(self.price_history) < 2:
            return False

        # Adjust threshold based on market volatility
        adjusted_threshold = self.structural_break_threshold
        if self.use_market_awareness:
            # Higher volatility = more lenient threshold
            volatility_factor = 1.0 + market_volatility
            adjusted_threshold *= volatility_factor

        # Check for significant price changes
        current_prices = self.price_history[-1]
        prev_prices = self.price_history[-2]

        if len(current_prices) == len(prev_prices):
            price_changes = np.abs(current_prices - prev_prices)
            max_change = np.max(price_changes)
            return bool(max_change > adjusted_threshold)

        return False

    def _calculate_graduated_penalties(
        self,
        prices: np.ndarray,
        parallel_violation: bool,
        structural_break_violation: bool,
    ) -> Tuple[np.ndarray, List[ViolationSeverity], List[float]]:
        """
        Calculate graduated penalties based on violation type.

        Returns:
            Tuple of (fines, severities, multipliers)
        """
        n_firms = len(prices)
        fines = np.zeros(n_firms)
        severities: List[ViolationSeverity] = []
        multipliers: List[float] = []

        if not (parallel_violation or structural_break_violation):
            return fines, severities, multipliers

        # Determine violation severity based on violation type
        if parallel_violation and structural_break_violation:
            severity = ViolationSeverity.CRITICAL
        elif parallel_violation:
            severity = ViolationSeverity.SEVERE
        elif structural_break_violation:
            severity = ViolationSeverity.MODERATE
        else:
            severity = ViolationSeverity.MINOR

        # Apply graduated penalties
        base_fine = self.fine_amount
        severity_multiplier = self.penalty_structure[severity]

        for i in range(n_firms):
            # Get cumulative penalty multiplier for this firm
            cumulative_multiplier = self.penalty_multipliers.get(i, 1.0)

            # Calculate final fine
            fine = base_fine * severity_multiplier * cumulative_multiplier

            # Apply leniency reductions if enabled
            if self.leniency_program is not None:
                reduction = self.leniency_program.get_fine_reduction(i)
                fine *= 1.0 - reduction

            fines[i] = fine
            severities.append(severity)
            multipliers.append(cumulative_multiplier)

        return fines, severities, multipliers

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            "total_violations": len(self.parallel_violations)
            + len(self.structural_break_violations),
            "parallel_violations": len(self.parallel_violations),
            "structural_break_violations": len(self.structural_break_violations),
            "total_fines_applied": self.total_fines_applied,
            # Risk scores removed; keep placeholders at 0.0 for backward compatibility
            "average_risk_score": 0.0,
            "current_risk_score": 0.0,
            "market_volatility": (
                self.market_volatility_history[-1]
                if self.market_volatility_history
                else 0.0
            ),
            "violation_counts": self.violation_counts.copy(),
            "penalty_multipliers": self.penalty_multipliers.copy(),
            "market_volatility_history": self.market_volatility_history.copy(),
        }

    def reset(self, n_firms: Optional[int] = None) -> None:
        """Reset regulator state for new episode."""
        super().reset(n_firms)
        self.violation_counts.clear()
        self.market_volatility_history.clear()
        self.penalty_multipliers.clear()
