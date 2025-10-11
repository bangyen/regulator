"""
Regulator agent for monitoring and detecting cartel behavior.

This module implements a Regulator class that monitors episode logs and applies
rule-based screens to detect potential cartel behavior, including parallel pricing
and sudden structural breaks. When violations are detected, the regulator applies
penalties by subtracting fines from firm rewards.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agents.leniency import LeniencyProgram, LeniencyStatus


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
        parallel_threshold: float = 5.0,  # Less sensitive to reduce false positives
        parallel_steps: int = 4,  # More steps required for detection
        structural_break_threshold: float = 30.0,  # Even less sensitive to individual price jumps
        fine_amount: float = 25.0,  # Reasonable fine amount
        leniency_enabled: bool = True,
        leniency_reduction: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the regulator with detection parameters.

        Args:
            parallel_threshold: Maximum price difference (ε) to consider prices parallel
            parallel_steps: Minimum consecutive steps (k) for parallel pricing detection
            structural_break_threshold: Maximum price change (δ) before structural break
            fine_amount: Fine to subtract from firm rewards when violations detected
            leniency_enabled: Whether to enable the leniency program
            leniency_reduction: Fraction of fine reduction for whistleblowers
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
        if not 0.0 <= leniency_reduction <= 1.0:
            raise ValueError("Leniency reduction must be between 0.0 and 1.0")

        self.parallel_threshold = parallel_threshold
        self.parallel_steps = parallel_steps
        self.structural_break_threshold = structural_break_threshold
        self.fine_amount = fine_amount
        self.leniency_enabled = leniency_enabled
        self.leniency_reduction = leniency_reduction
        self.np_random = np.random.default_rng(seed)

        # Episode monitoring state
        self.price_history: List[np.ndarray] = []
        self.parallel_violations: List[Tuple[int, str]] = []  # (step, violation_type)
        self.structural_break_violations: List[Tuple[int, str]] = []
        self.total_fines_applied: float = 0.0
        self.profit_history: List[np.ndarray] = []  # Track firm profits over time
        self.violation_start_step: Optional[int] = None  # When current violation period started
        self.consecutive_violation_steps: int = 0  # Duration of ongoing violation

        # Leniency program
        if self.leniency_enabled:
            self.leniency_program: Optional[LeniencyProgram] = LeniencyProgram(
                leniency_reduction=leniency_reduction, seed=seed
            )
        else:
            self.leniency_program = None

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
            detection_results["consecutive_violations"] = self.consecutive_violation_steps
        else:
            # No violation this step - reset consecutive counter
            self.violation_start_step = None
            self.consecutive_violation_steps = 0
            detection_results["violation_detected"] = False

        return detection_results

    def _detect_parallel_pricing(self, step: int) -> bool:
        """
        Detect if firms are engaging in parallel pricing behavior.

        Parallel pricing is detected when:
        1. All firms have identical prices (immediate detection), OR
        2. All firms' prices are within the threshold for at least the minimum number of consecutive steps, OR
        3. Price variation is very low (coefficient of variation < 0.05) indicating potential collusion

        Args:
            step: Current step number

        Returns:
            True if parallel pricing violation detected, False otherwise
        """
        # Check current step for identical pricing (immediate detection)
        if len(self.price_history) > 0:
            current_prices = self.price_history[-1]
            if len(current_prices) >= 2:
                # Check if all prices are identical (exact match)
                if len(set(current_prices)) == 1:
                    return True  # All firms have identical prices - clear collusion

                # Check for very low price variation (coefficient of variation < 0.05)
                price_mean = np.mean(current_prices)
                if price_mean > 0:  # Avoid division by zero
                    price_std = np.std(current_prices)
                    cv = price_std / price_mean
                    if (
                        cv < 0.05 and price_mean > 20
                    ):  # Very low variation with reasonable prices
                        return True  # Suspiciously similar prices

        # Need at least parallel_steps of history for threshold-based detection
        if len(self.price_history) < self.parallel_steps:
            return False

        # Check the last parallel_steps steps for threshold-based parallel pricing
        recent_prices = self.price_history[-self.parallel_steps :]

        # For each step, check if all prices are within threshold
        for prices in recent_prices:
            if len(prices) < 2:
                return False  # Need at least 2 firms

            # Check if all prices are within threshold of each other
            max_price: float = np.max(prices)
            min_price: float = np.min(prices)
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
        max_change: float = np.max(price_changes)

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
        # Store profit history for future fine calculations
        self.profit_history.append(rewards.copy())
        
        # Calculate fines if violation detected
        if detection_results.get("violation_detected", False):
            prices = detection_results["prices"]
            consecutive_violations = detection_results.get("consecutive_violations", 1)
            parallel_violation = detection_results.get("parallel_violation", False)
            step = detection_results["step"]
            
            # Calculate baseline competitive price (using early market history)
            competitive_price = 10.0  # Approximate competitive equilibrium
            if len(self.price_history) >= 10:
                # Use lower quartile of first 10 steps as competitive baseline
                early_prices = np.array(self.price_history[:10]).flatten()
                competitive_price = np.percentile(early_prices, 25)
            
            # Calculate harm-based fines (more realistic approach)
            penalties = np.zeros(len(prices))
            for i in range(len(prices)):
                # 1. Calculate consumer harm: overcharge × implied market volume
                price_markup = max(0, prices[i] - competitive_price)
                estimated_quantity = max(10.0, 50.0 - prices[i])  # Simple demand curve
                consumer_harm = price_markup * estimated_quantity
                
                # 2. Calculate affected revenue (last 10 periods if available)
                lookback = min(10, len(self.profit_history) - 1)  # -1 because we just appended current
                if lookback > 0:
                    recent_profits = np.array([self.profit_history[-(j+2)][i] 
                                               for j in range(lookback)])
                    avg_recent_profit = np.mean(recent_profits)
                else:
                    avg_recent_profit = rewards[i]
                
                # 3. Fine calculation combining harm and revenue
                # Revenue component: 2% of recent profits (per-step penalty rate)
                revenue_fine = 0.02 * max(0, avg_recent_profit)
                
                # Harm component: 0.15× consumer harm (heavily scaled for simulation)
                harm_fine = 0.15 * consumer_harm
                
                # Base fine from deviation magnitude
                deviation_ratio = min(price_markup / competitive_price, 2.0)  # 0-2x competitive
                base_penalty = self.fine_amount * 0.3 * deviation_ratio  # 0-60% of fine_amount
                
                # 4. Duration multiplier (longer violations = higher fines)
                duration_multiplier = 1.0 + (consecutive_violations - 1) * 0.03
                duration_multiplier = min(duration_multiplier, 1.3)  # Cap at 1.3x
                
                # Combined fine (weighted combination + duration penalty)
                penalties[i] = (0.3 * base_penalty + 0.4 * revenue_fine + 0.3 * harm_fine) * duration_multiplier
                
                # Add some variance based on firm behavior (use price as proxy)
                price_variance = abs(prices[i] - np.mean(prices)) / max(np.mean(prices), 1.0)
                variance_factor = 1.0 + price_variance * 0.2  # Up to 20% variation
                penalties[i] *= variance_factor
                
                # Cap individual fines at 1.8x the fine_amount parameter
                penalties[i] = min(penalties[i], self.fine_amount * 1.8)
                
                # Minimum fine for detected violations (3% of base)
                if penalties[i] < self.fine_amount * 0.03:
                    penalties[i] = self.fine_amount * 0.03

            # Apply leniency reductions if program is enabled
            if self.leniency_program is not None:
                for i in range(len(prices)):
                    reduction = self.leniency_program.get_fine_reduction(i)
                    penalties[i] *= 1.0 - reduction

                # Update collusion evidence for leniency program
                evidence_strength = 0.8 if parallel_violation else 0.6
                for i in range(len(prices)):
                    self.leniency_program.update_collusion_evidence(
                        i, evidence_strength, step
                    )

            # Update detection results with calculated fines
            detection_results["fines_applied"] = penalties
            self.total_fines_applied += np.sum(penalties)
        else:
            penalties = detection_results["fines_applied"]
        
        modified_rewards: np.ndarray = rewards - penalties

        # Allow negative rewards for economic realism (firms can have losses)
        # modified_rewards = np.maximum(modified_rewards, 0.0)

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
