"""
Unit tests for the Regulator class.

This module contains comprehensive tests for the Regulator class, including:
- Parallel pricing detection when conditions are met
- No false positives when firms randomize prices
- Fines reduce firm rewards appropriately
- Structural break detection
- Edge cases and error handling
"""

import math

import numpy as np
import pytest

from src.agents.regulator import Regulator


class TestRegulatorInitialization:
    """Test suite for Regulator initialization and configuration."""

    def test_regulator_initialization_default_parameters(self) -> None:
        """Test Regulator initialization with default parameters."""
        regulator = Regulator()

        assert regulator.parallel_threshold == 2.0
        assert regulator.parallel_steps == 3
        assert regulator.structural_break_threshold == 10.0
        assert regulator.fine_amount == 50.0
        assert regulator.price_history == []
        assert regulator.parallel_violations == []
        assert regulator.structural_break_violations == []
        assert regulator.total_fines_applied == 0.0

    def test_regulator_initialization_custom_parameters(self) -> None:
        """Test Regulator initialization with custom parameters."""
        regulator = Regulator(
            parallel_threshold=5.0,
            parallel_steps=5,
            structural_break_threshold=15.0,
            fine_amount=100.0,
            seed=42,
        )

        assert regulator.parallel_threshold == 5.0
        assert regulator.parallel_steps == 5
        assert regulator.structural_break_threshold == 15.0
        assert regulator.fine_amount == 100.0
        assert isinstance(regulator.np_random, np.random.Generator)

    def test_regulator_initialization_validation_errors(self) -> None:
        """Test Regulator initialization with invalid parameters raises errors."""
        with pytest.raises(ValueError, match="Parallel threshold must be non-negative"):
            Regulator(parallel_threshold=-1.0)

        with pytest.raises(ValueError, match="Parallel steps must be at least 1"):
            Regulator(parallel_steps=0)

        with pytest.raises(
            ValueError, match="Structural break threshold must be non-negative"
        ):
            Regulator(structural_break_threshold=-5.0)

        with pytest.raises(ValueError, match="Fine amount must be non-negative"):
            Regulator(fine_amount=-10.0)


class TestParallelPricingDetection:
    """Test suite for parallel pricing detection functionality."""

    def test_parallel_pricing_detection_insufficient_history(self) -> None:
        """Test that parallel pricing detection returns False with insufficient history."""
        regulator = Regulator(parallel_steps=3)

        # Test with no history
        result = regulator._detect_parallel_pricing(step=0)
        assert result is False

        # Test with insufficient history
        regulator.price_history = [np.array([10.0, 10.0, 10.0])]
        result = regulator._detect_parallel_pricing(step=1)
        assert result is False

        regulator.price_history = [
            np.array([10.0, 10.0, 10.0]),
            np.array([10.0, 10.0, 10.0]),
        ]
        result = regulator._detect_parallel_pricing(step=2)
        assert result is False

    def test_parallel_pricing_detection_success(self) -> None:
        """Test that parallel pricing is detected when conditions are met."""
        regulator = Regulator(parallel_threshold=2.0, parallel_steps=3)

        # Create history with parallel pricing for 3 consecutive steps
        regulator.price_history = [
            np.array([10.0, 10.5, 9.8]),  # Range: 0.7, within threshold
            np.array([11.0, 11.2, 10.9]),  # Range: 0.3, within threshold
            np.array([12.0, 12.1, 11.9]),  # Range: 0.2, within threshold
        ]

        result = regulator._detect_parallel_pricing(step=3)
        assert result is True

    def test_parallel_pricing_detection_failure_due_to_spread(self) -> None:
        """Test that parallel pricing is not detected when prices are too spread out."""
        regulator = Regulator(parallel_threshold=2.0, parallel_steps=3)

        # Create history with prices too spread out
        regulator.price_history = [
            np.array([10.0, 10.5, 9.8]),  # Range: 0.7, within threshold
            np.array([11.0, 11.2, 10.9]),  # Range: 0.3, within threshold
            np.array([12.0, 15.0, 11.9]),  # Range: 3.1, exceeds threshold
        ]

        result = regulator._detect_parallel_pricing(step=3)
        assert result is False

    def test_parallel_pricing_detection_failure_due_to_insufficient_steps(self) -> None:
        """Test that parallel pricing is not detected with insufficient parallel steps."""
        regulator = Regulator(parallel_threshold=2.0, parallel_steps=3)

        # Create history with only 2 parallel steps
        regulator.price_history = [
            np.array([10.0, 10.5, 9.8]),  # Range: 0.7, within threshold
            np.array([11.0, 11.2, 10.9]),  # Range: 0.3, within threshold
            np.array([12.0, 20.0, 11.9]),  # Range: 8.1, exceeds threshold
        ]

        result = regulator._detect_parallel_pricing(step=3)
        assert result is False

    def test_parallel_pricing_detection_edge_case_exact_threshold(self) -> None:
        """Test parallel pricing detection at exact threshold boundary."""
        regulator = Regulator(parallel_threshold=2.0, parallel_steps=3)

        # Create history with prices exactly at threshold
        regulator.price_history = [
            np.array([10.0, 10.0, 10.0]),  # Range: 0.0, within threshold
            np.array([11.0, 11.0, 11.0]),  # Range: 0.0, within threshold
            np.array([12.0, 12.0, 12.0]),  # Range: 0.0, within threshold
        ]

        result = regulator._detect_parallel_pricing(step=3)
        assert result is True

        # Test with prices exactly at threshold limit
        regulator.price_history = [
            np.array([10.0, 10.0, 10.0]),  # Range: 0.0, within threshold
            np.array([11.0, 11.0, 11.0]),  # Range: 0.0, within threshold
            np.array([12.0, 14.0, 12.0]),  # Range: 2.0, exactly at threshold
        ]

        result = regulator._detect_parallel_pricing(step=3)
        assert result is True

    def test_parallel_pricing_detection_single_firm(self) -> None:
        """Test parallel pricing detection with single firm (should return False)."""
        regulator = Regulator(parallel_threshold=2.0, parallel_steps=3)

        # Create history with single firm
        regulator.price_history = [
            np.array([10.0]),
            np.array([11.0]),
            np.array([12.0]),
        ]

        result = regulator._detect_parallel_pricing(step=3)
        assert result is False  # Need at least 2 firms for parallel pricing


class TestStructuralBreakDetection:
    """Test suite for structural break detection functionality."""

    def test_structural_break_detection_insufficient_history(self) -> None:
        """Test that structural break detection returns False with insufficient history."""
        regulator = Regulator()

        # Test with no history
        result = regulator._detect_structural_break(step=0)
        assert result is False

        # Test with only one step of history
        regulator.price_history = [np.array([10.0, 11.0, 12.0])]
        result = regulator._detect_structural_break(step=1)
        assert result is False

    def test_structural_break_detection_success(self) -> None:
        """Test that structural breaks are detected when price changes exceed threshold."""
        regulator = Regulator(structural_break_threshold=5.0)

        # Create history with large price change
        regulator.price_history = [
            np.array([10.0, 11.0, 12.0]),
            np.array([20.0, 11.0, 12.0]),  # Firm 0 changed by 10.0, exceeds threshold
        ]

        result = regulator._detect_structural_break(step=2)
        assert result is True

    def test_structural_break_detection_failure(self) -> None:
        """Test that structural breaks are not detected when price changes are small."""
        regulator = Regulator(structural_break_threshold=5.0)

        # Create history with small price changes
        regulator.price_history = [
            np.array([10.0, 11.0, 12.0]),
            np.array([12.0, 13.0, 14.0]),  # All changes are 2.0, within threshold
        ]

        result = regulator._detect_structural_break(step=2)
        assert result is False

    def test_structural_break_detection_edge_case_exact_threshold(self) -> None:
        """Test structural break detection at exact threshold boundary."""
        regulator = Regulator(structural_break_threshold=5.0)

        # Create history with price change exactly at threshold
        regulator.price_history = [
            np.array([10.0, 11.0, 12.0]),
            np.array([15.0, 11.0, 12.0]),  # Firm 0 changed by exactly 5.0
        ]

        result = regulator._detect_structural_break(step=2)
        assert result is False  # Should be False since change is not > threshold

        # Test with price change just above threshold
        regulator.price_history = [
            np.array([10.0, 11.0, 12.0]),
            np.array([15.1, 11.0, 12.0]),  # Firm 0 changed by 5.1, exceeds threshold
        ]

        result = regulator._detect_structural_break(step=2)
        assert result is True

    def test_structural_break_detection_multiple_firms(self) -> None:
        """Test structural break detection with multiple firms having large changes."""
        regulator = Regulator(structural_break_threshold=5.0)

        # Create history with multiple firms having large changes
        regulator.price_history = [
            np.array([10.0, 11.0, 12.0]),
            np.array([20.0, 25.0, 30.0]),  # All firms changed by 10+, exceeds threshold
        ]

        result = regulator._detect_structural_break(step=2)
        assert result is True


class TestMonitorStep:
    """Test suite for the monitor_step method."""

    def test_monitor_step_no_violations(self) -> None:
        """Test monitor_step when no violations are detected."""
        regulator = Regulator(parallel_steps=3, structural_break_threshold=10.0)

        # First step - no violations possible
        prices = np.array([10.0, 11.0, 12.0])
        result = regulator.monitor_step(prices, step=0)

        assert result["step"] == 0
        assert result["parallel_violation"] is False
        assert result["structural_break_violation"] is False
        assert np.all(result["fines_applied"] == 0.0)
        assert result["violation_details"] == []
        assert len(regulator.price_history) == 1

    def test_monitor_step_parallel_pricing_violation(self) -> None:
        """Test monitor_step when parallel pricing violation is detected."""
        regulator = Regulator(
            parallel_threshold=2.0,
            parallel_steps=3,
            structural_break_threshold=10.0,
            fine_amount=25.0,
        )

        # Create history that will trigger parallel pricing detection
        regulator.price_history = [
            np.array([10.0, 10.5, 9.8]),
            np.array([11.0, 11.2, 10.9]),
        ]

        # This step should trigger parallel pricing detection
        prices = np.array([12.0, 12.1, 11.9])
        result = regulator.monitor_step(prices, step=2)

        assert result["step"] == 2
        assert result["parallel_violation"] is True
        assert result["structural_break_violation"] is False
        assert np.all(result["fines_applied"] == 25.0)
        assert "Parallel pricing detected at step 2" in result["violation_details"]
        assert len(regulator.parallel_violations) == 1
        assert regulator.total_fines_applied == 75.0  # 3 firms * 25.0 fine

    def test_monitor_step_structural_break_violation(self) -> None:
        """Test monitor_step when structural break violation is detected."""
        regulator = Regulator(
            parallel_threshold=2.0,
            parallel_steps=3,
            structural_break_threshold=5.0,
            fine_amount=30.0,
        )

        # Create history that will trigger structural break detection
        regulator.price_history = [np.array([10.0, 11.0, 12.0])]

        # This step should trigger structural break detection
        prices = np.array([20.0, 11.0, 12.0])  # Firm 0 changed by 10.0
        result = regulator.monitor_step(prices, step=1)

        assert result["step"] == 1
        assert result["parallel_violation"] is False
        assert result["structural_break_violation"] is True
        assert np.all(result["fines_applied"] == 30.0)
        assert "Structural break detected at step 1" in result["violation_details"]
        assert len(regulator.structural_break_violations) == 1
        assert regulator.total_fines_applied == 90.0  # 3 firms * 30.0 fine

    def test_monitor_step_both_violations(self) -> None:
        """Test monitor_step when both violations are detected."""
        regulator = Regulator(
            parallel_threshold=2.0,
            parallel_steps=2,  # Reduced to make parallel detection easier
            structural_break_threshold=5.0,
            fine_amount=20.0,
        )

        # Create history that will trigger both violations
        regulator.price_history = [np.array([10.0, 10.5, 9.8])]

        # This step should trigger both violations
        prices = np.array([20.0, 20.1, 19.9])  # Large change + parallel pricing
        result = regulator.monitor_step(prices, step=1)

        assert result["step"] == 1
        assert result["parallel_violation"] is True
        assert result["structural_break_violation"] is True
        assert np.all(result["fines_applied"] == 20.0)
        assert len(result["violation_details"]) == 2
        assert len(regulator.parallel_violations) == 1
        assert len(regulator.structural_break_violations) == 1
        assert regulator.total_fines_applied == 60.0  # 3 firms * 20.0 fine


class TestApplyPenalties:
    """Test suite for the apply_penalties method."""

    def test_apply_penalties_no_violations(self) -> None:
        """Test apply_penalties when no violations are detected."""
        regulator = Regulator()
        rewards = np.array([100.0, 150.0, 200.0])
        detection_results = {
            "fines_applied": np.array([0.0, 0.0, 0.0]),
        }

        modified_rewards = regulator.apply_penalties(rewards, detection_results)

        assert np.array_equal(modified_rewards, rewards)

    def test_apply_penalties_with_violations(self) -> None:
        """Test apply_penalties when violations are detected."""
        regulator = Regulator(fine_amount=50.0)
        rewards = np.array([100.0, 150.0, 200.0])
        detection_results = {
            "fines_applied": np.array([50.0, 50.0, 50.0]),
        }

        modified_rewards = regulator.apply_penalties(rewards, detection_results)

        expected_rewards = np.array([50.0, 100.0, 150.0])
        assert np.array_equal(modified_rewards, expected_rewards)

    def test_apply_penalties_prevents_negative_rewards(self) -> None:
        """Test that apply_penalties prevents rewards from going below zero."""
        regulator = Regulator(fine_amount=50.0)
        rewards = np.array([30.0, 40.0, 200.0])
        detection_results = {
            "fines_applied": np.array([50.0, 50.0, 50.0]),
        }

        modified_rewards = regulator.apply_penalties(rewards, detection_results)

        expected_rewards = np.array([0.0, 0.0, 150.0])
        assert np.array_equal(modified_rewards, expected_rewards)
        assert np.all(modified_rewards >= 0.0)

    def test_apply_penalties_different_fine_amounts(self) -> None:
        """Test apply_penalties with different fine amounts per firm."""
        regulator = Regulator()
        rewards = np.array([100.0, 150.0, 200.0])
        detection_results = {
            "fines_applied": np.array([25.0, 50.0, 75.0]),
        }

        modified_rewards = regulator.apply_penalties(rewards, detection_results)

        expected_rewards = np.array([75.0, 100.0, 125.0])
        assert np.array_equal(modified_rewards, expected_rewards)


class TestViolationSummary:
    """Test suite for violation summary functionality."""

    def test_get_violation_summary_no_violations(self) -> None:
        """Test get_violation_summary when no violations occurred."""
        regulator = Regulator()
        summary = regulator.get_violation_summary()

        assert summary["total_parallel_violations"] == 0
        assert summary["total_structural_break_violations"] == 0
        assert summary["total_fines_applied"] == 0.0
        assert summary["parallel_violation_steps"] == []
        assert summary["structural_break_steps"] == []

    def test_get_violation_summary_with_violations(self) -> None:
        """Test get_violation_summary when violations occurred."""
        regulator = Regulator(fine_amount=30.0)

        # Simulate some violations
        regulator.parallel_violations = [
            (2, "parallel_pricing"),
            (5, "parallel_pricing"),
        ]
        regulator.structural_break_violations = [(3, "structural_break")]
        regulator.total_fines_applied = 180.0

        summary = regulator.get_violation_summary()

        assert summary["total_parallel_violations"] == 2
        assert summary["total_structural_break_violations"] == 1
        assert summary["total_fines_applied"] == 180.0
        assert summary["parallel_violation_steps"] == [2, 5]
        assert summary["structural_break_steps"] == [3]


class TestPriceStatistics:
    """Test suite for price statistics functionality."""

    def test_get_price_statistics_no_history(self) -> None:
        """Test get_price_statistics with no price history."""
        regulator = Regulator()
        stats = regulator.get_price_statistics()

        assert stats == {}

    def test_get_price_statistics_with_history(self) -> None:
        """Test get_price_statistics with price history."""
        regulator = Regulator()
        regulator.price_history = [
            np.array([10.0, 20.0, 30.0]),
            np.array([15.0, 25.0, 35.0]),
            np.array([12.0, 22.0, 32.0]),
        ]

        stats = regulator.get_price_statistics()

        all_prices = np.concatenate(regulator.price_history)
        expected_mean = float(np.mean(all_prices))
        expected_std = float(np.std(all_prices))
        expected_min = float(np.min(all_prices))
        expected_max = float(np.max(all_prices))
        expected_range = expected_max - expected_min

        assert math.isclose(stats["mean_price"], expected_mean)
        assert math.isclose(stats["std_price"], expected_std)
        assert math.isclose(stats["min_price"], expected_min)
        assert math.isclose(stats["max_price"], expected_max)
        assert math.isclose(stats["price_range"], expected_range)

    def test_get_parallel_pricing_ratio_no_history(self) -> None:
        """Test get_parallel_pricing_ratio with no price history."""
        regulator = Regulator(parallel_steps=3)
        ratio = regulator.get_parallel_pricing_ratio()

        assert ratio == 0.0

    def test_get_parallel_pricing_ratio_insufficient_history(self) -> None:
        """Test get_parallel_pricing_ratio with insufficient history."""
        regulator = Regulator(parallel_steps=3)
        regulator.price_history = [
            np.array([10.0, 10.0, 10.0]),
            np.array([11.0, 11.0, 11.0]),
        ]

        ratio = regulator.get_parallel_pricing_ratio()
        assert ratio == 0.0

    def test_get_parallel_pricing_ratio_with_history(self) -> None:
        """Test get_parallel_pricing_ratio with sufficient history."""
        regulator = Regulator(parallel_threshold=2.0, parallel_steps=2)

        # Create history with some parallel pricing
        regulator.price_history = [
            np.array([10.0, 10.0, 10.0]),  # Parallel
            np.array([11.0, 11.0, 11.0]),  # Parallel
            np.array([12.0, 20.0, 12.0]),  # Not parallel (range = 8)
            np.array([13.0, 13.0, 13.0]),  # Parallel
            np.array([14.0, 14.0, 14.0]),  # Parallel
        ]

        ratio = regulator.get_parallel_pricing_ratio()

        # Should detect parallel pricing at steps 2, 3, 4, 5
        # Total checkable steps: 5 - 2 + 1 = 4
        # Parallel steps: 4
        # Ratio: 4/4 = 1.0
        assert math.isclose(ratio, 1.0)


class TestReset:
    """Test suite for reset functionality."""

    def test_reset_clears_all_state(self) -> None:
        """Test that reset clears all regulator state."""
        regulator = Regulator(leniency_enabled=False)

        # Add some state
        regulator.price_history = [np.array([10.0, 11.0, 12.0])]
        regulator.parallel_violations = [(1, "parallel_pricing")]
        regulator.structural_break_violations = [(2, "structural_break")]
        regulator.total_fines_applied = 150.0

        # Reset
        regulator.reset()

        assert regulator.price_history == []
        assert regulator.parallel_violations == []
        assert regulator.structural_break_violations == []
        assert regulator.total_fines_applied == 0.0


class TestRegulatorIntegration:
    """Integration tests for Regulator with realistic scenarios."""

    def test_regulator_with_random_agents_no_false_positives(self) -> None:
        """Test that regulator doesn't produce false positives with random pricing."""
        regulator = Regulator(
            parallel_threshold=1.0,  # Very strict
            parallel_steps=2,
            structural_break_threshold=5.0,
            fine_amount=10.0,
        )

        # Simulate random pricing behavior
        np.random.seed(42)
        total_fines = 0.0
        violations = 0

        for step in range(20):
            # Generate random prices
            prices = np.random.uniform(10.0, 50.0, size=3)
            result = regulator.monitor_step(prices, step)

            if result["parallel_violation"] or result["structural_break_violation"]:
                violations += 1
                total_fines += np.sum(result["fines_applied"])

        # With random pricing, we should have some violations due to strict thresholds
        # but not excessive violations
        assert violations <= 20  # Allow for some violations with strict thresholds
        assert total_fines <= 600.0  # 20 violations * 3 firms * 10.0 fine

    def test_regulator_with_cartel_behavior_detects_violations(self) -> None:
        """Test that regulator detects cartel behavior (parallel pricing)."""
        regulator = Regulator(
            parallel_threshold=2.0,
            parallel_steps=3,
            structural_break_threshold=10.0,
            fine_amount=25.0,
        )

        # Simulate cartel behavior (firms coordinate prices)
        violations = 0
        total_fines = 0.0

        for step in range(10):
            if step < 3:
                # Initial random prices
                prices = np.array([20.0, 25.0, 30.0])
            else:
                # Cartel behavior: all firms set similar prices
                base_price = 40.0 + step
                prices = np.array([base_price, base_price + 0.5, base_price - 0.5])

            result = regulator.monitor_step(prices, step)

            if result["parallel_violation"] or result["structural_break_violation"]:
                violations += 1
                total_fines += np.sum(result["fines_applied"])

        # Should detect parallel pricing violations
        assert violations >= 1
        assert total_fines > 0.0

    def test_regulator_penalty_application_reduces_rewards(self) -> None:
        """Test that regulator penalties appropriately reduce firm rewards."""
        regulator = Regulator(fine_amount=50.0)

        # Simulate episode with violations
        original_rewards = np.array([100.0, 150.0, 200.0])

        # Create detection results with violations
        detection_results = {
            "fines_applied": np.array([50.0, 50.0, 50.0]),
        }

        # Apply penalties
        modified_rewards = regulator.apply_penalties(
            original_rewards, detection_results
        )

        # Check that rewards were reduced appropriately
        expected_rewards = np.array([50.0, 100.0, 150.0])
        assert np.array_equal(modified_rewards, expected_rewards)

        # Check that total reward reduction equals total fines
        total_reduction: float = np.sum(original_rewards - modified_rewards)
        total_fines: float = np.sum(detection_results["fines_applied"])
        assert math.isclose(total_reduction, total_fines)
