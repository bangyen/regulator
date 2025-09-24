"""
Tests for the EnhancedRegulator class.

This module tests the enhanced regulator functionality including graduated penalties,
continuous monitoring scores, market awareness, and cumulative penalty tracking.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from src.agents.enhanced_regulator import EnhancedRegulator, ViolationSeverity


class TestViolationSeverity:
    """Test the ViolationSeverity enum."""

    def test_violation_severity_values(self):
        """Test that violation severity enum has correct values."""
        assert ViolationSeverity.MINOR.value == "minor"
        assert ViolationSeverity.MODERATE.value == "moderate"
        assert ViolationSeverity.SEVERE.value == "severe"
        assert ViolationSeverity.CRITICAL.value == "critical"


class TestEnhancedRegulator:
    """Test the EnhancedRegulator class."""

    @pytest.fixture
    def regulator(self):
        """Create a basic enhanced regulator for testing."""
        return EnhancedRegulator(
            parallel_threshold=5.0,
            parallel_steps=4,
            structural_break_threshold=30.0,
            base_fine_amount=25.0,
            leniency_enabled=True,
            leniency_reduction=0.5,
            use_graduated_penalties=True,
            use_market_awareness=True,
            cumulative_penalty_multiplier=1.2,
            max_penalty_multiplier=5.0,
            market_volatility_threshold=0.3,
            seed=42,
        )

    def test_initialization(self, regulator):
        """Test enhanced regulator initialization."""
        assert regulator.use_graduated_penalties is True
        assert regulator.use_market_awareness is True
        assert regulator.cumulative_penalty_multiplier == 1.2
        assert regulator.max_penalty_multiplier == 5.0
        assert regulator.market_volatility_threshold == 0.3
        assert regulator.violation_counts == {}
        assert regulator.market_volatility_history == []
        assert regulator.penalty_multipliers == {}

    def test_penalty_structure(self, regulator):
        """Test that penalty structure is correctly initialized."""
        expected_structure = {
            ViolationSeverity.MINOR: 0.5,
            ViolationSeverity.MODERATE: 1.0,
            ViolationSeverity.SEVERE: 2.0,
            ViolationSeverity.CRITICAL: 4.0,
        }
        assert regulator.penalty_structure == expected_structure

    def test_monitor_step_no_violations(self, regulator):
        """Test monitoring step with no violations."""
        prices = np.array([30.0, 60.0, 90.0])  # Very different prices

        # Add some price history to avoid edge cases
        for _ in range(5):
            regulator.price_history.append(np.array([30.0, 60.0, 90.0]))

        result = regulator.monitor_step(prices, step=10)

        assert result["step"] == 10
        assert result["parallel_violation"] is False
        assert result["structural_break_violation"] is False
        assert all(fine == 0 for fine in result["fines_applied"])
        assert result["violation_details"] == []
        assert "market_volatility" in result
        assert result["violation_severities"] == []
        assert result["penalty_multipliers"] == []

    def test_monitor_step_parallel_violation(self, regulator):
        """Test monitoring step with parallel pricing violation."""
        # Create parallel pricing scenario
        parallel_prices = np.array([50.0, 50.1, 50.2])  # Very similar prices

        # Add price history with parallel pricing
        for _ in range(4):
            regulator.price_history.append(parallel_prices)

        result = regulator.monitor_step(parallel_prices, step=10)

        assert result["parallel_violation"] is True
        assert result["structural_break_violation"] is False
        assert any(fine > 0 for fine in result["fines_applied"])
        assert len(result["violation_severities"]) == 3
        assert len(result["penalty_multipliers"]) == 3

    def test_monitor_step_structural_break_violation(self, regulator):
        """Test monitoring step with structural break violation."""
        # Add price history
        for _ in range(2):
            regulator.price_history.append(np.array([50.0, 52.0, 48.0]))

        # Create structural break (large price change)
        new_prices = np.array([90.0, 92.0, 88.0])  # Large price increase (>40)

        result = regulator.monitor_step(new_prices, step=10)

        assert result["parallel_violation"] is False
        assert result["structural_break_violation"] is True
        assert any(fine > 0 for fine in result["fines_applied"])

    def test_monitor_step_both_violations(self, regulator):
        """Test monitoring step with both parallel and structural break violations."""
        # Add price history
        for _ in range(4):
            regulator.price_history.append(np.array([50.0, 50.1, 50.2]))

        # Create both violations
        new_prices = np.array([90.0, 90.1, 90.2])  # Parallel + structural break

        result = regulator.monitor_step(new_prices, step=10)

        assert result["parallel_violation"] is True
        assert result["structural_break_violation"] is True
        assert any(fine > 0 for fine in result["fines_applied"])
        # Should be critical severity
        assert all(
            severity == "critical" for severity in result["violation_severities"]
        )

    def test_calculate_market_volatility_empty_history(self, regulator):
        """Test market volatility calculation with empty history."""
        volatility = regulator._calculate_market_volatility()
        assert volatility == 0.0

    def test_calculate_market_volatility_single_step(self, regulator):
        """Test market volatility calculation with single step."""
        regulator.price_history.append(np.array([50.0, 52.0, 48.0]))
        volatility = regulator._calculate_market_volatility()
        assert volatility == 0.0  # Need at least 2 steps for volatility

    def test_calculate_market_volatility_multiple_steps(self, regulator):
        """Test market volatility calculation with multiple steps."""
        # Add price history with variation
        regulator.price_history.append(np.array([50.0, 52.0, 48.0]))
        regulator.price_history.append(np.array([55.0, 57.0, 53.0]))
        regulator.price_history.append(np.array([45.0, 47.0, 43.0]))

        volatility = regulator._calculate_market_volatility()
        assert volatility > 0.0  # Should have some volatility
        assert isinstance(volatility, float)

    def test_detect_parallel_pricing_enhanced_insufficient_history(self, regulator):
        """Test parallel pricing detection with insufficient history."""
        regulator.price_history.append(np.array([50.0, 52.0, 48.0]))

        violation = regulator._detect_parallel_pricing_enhanced(
            step=10, market_volatility=0.1
        )
        assert violation is False

    def test_detect_parallel_pricing_enhanced_parallel_prices(self, regulator):
        """Test parallel pricing detection with parallel prices."""
        # Add parallel pricing history
        parallel_prices = np.array([50.0, 50.1, 50.2])
        for _ in range(4):
            regulator.price_history.append(parallel_prices)

        violation = regulator._detect_parallel_pricing_enhanced(
            step=10, market_volatility=0.1
        )
        assert violation is True

    def test_detect_parallel_pricing_enhanced_non_parallel_prices(self, regulator):
        """Test parallel pricing detection with non-parallel prices."""
        # Add non-parallel pricing history
        for i in range(4):
            prices = np.array([50.0 + i, 60.0 + i, 70.0 + i])  # Increasing spread
            regulator.price_history.append(prices)

        violation = regulator._detect_parallel_pricing_enhanced(
            step=10, market_volatility=0.1
        )
        assert violation is False

    def test_detect_parallel_pricing_enhanced_market_awareness(self, regulator):
        """Test parallel pricing detection with market awareness."""
        # Add parallel pricing history
        parallel_prices = np.array([50.0, 50.1, 50.2])
        for _ in range(4):
            regulator.price_history.append(parallel_prices)

        # Test with high volatility (should be more lenient)
        violation = regulator._detect_parallel_pricing_enhanced(
            step=10, market_volatility=0.5
        )
        assert violation is True

    def test_detect_structural_break_enhanced_insufficient_history(self, regulator):
        """Test structural break detection with insufficient history."""
        regulator.price_history.append(np.array([50.0, 52.0, 48.0]))

        violation = regulator._detect_structural_break_enhanced(
            step=10, market_volatility=0.1
        )
        assert violation is False

    def test_detect_structural_break_enhanced_structural_break(self, regulator):
        """Test structural break detection with actual structural break."""
        # Add price history
        regulator.price_history.append(np.array([50.0, 52.0, 48.0]))
        regulator.price_history.append(
            np.array([85.0, 87.0, 83.0])
        )  # Large change (>30)

        violation = regulator._detect_structural_break_enhanced(
            step=10, market_volatility=0.1
        )
        assert violation is True

    def test_detect_structural_break_enhanced_no_break(self, regulator):
        """Test structural break detection with no structural break."""
        # Add price history with small changes
        regulator.price_history.append(np.array([50.0, 52.0, 48.0]))
        regulator.price_history.append(np.array([51.0, 53.0, 49.0]))  # Small change

        violation = regulator._detect_structural_break_enhanced(
            step=10, market_volatility=0.1
        )
        assert violation is False

    def test_detect_structural_break_enhanced_market_awareness(self, regulator):
        """Test structural break detection with market awareness."""
        # Add price history
        regulator.price_history.append(np.array([50.0, 52.0, 48.0]))
        regulator.price_history.append(
            np.array([100.0, 102.0, 98.0])
        )  # Very large change (>45)

        # Test with high volatility (should be more lenient)
        violation = regulator._detect_structural_break_enhanced(
            step=10, market_volatility=0.5
        )
        assert violation is True

    def test_calculate_graduated_penalties_no_violations(self, regulator):
        """Test graduated penalty calculation with no violations."""
        prices = np.array([50.0, 52.0, 48.0])

        fines, severities, multipliers = regulator._calculate_graduated_penalties(
            prices, parallel_violation=False, structural_break_violation=False
        )

        assert np.all(fines == 0)
        assert severities == []
        assert multipliers == []

    def test_calculate_graduated_penalties_parallel_violation(self, regulator):
        """Test graduated penalty calculation with parallel violation."""
        prices = np.array([50.0, 52.0, 48.0])

        fines, severities, multipliers = regulator._calculate_graduated_penalties(
            prices, parallel_violation=True, structural_break_violation=False
        )

        assert np.all(fines > 0)
        assert all(s == ViolationSeverity.SEVERE for s in severities)
        assert len(multipliers) == 3
        assert all(m >= 1.0 for m in multipliers)

    def test_calculate_graduated_penalties_structural_break_violation(self, regulator):
        """Test graduated penalty calculation with structural break violation."""
        prices = np.array([50.0, 52.0, 48.0])

        fines, severities, multipliers = regulator._calculate_graduated_penalties(
            prices, parallel_violation=False, structural_break_violation=True
        )

        assert np.all(fines > 0)
        assert all(s == ViolationSeverity.MODERATE for s in severities)
        assert len(multipliers) == 3

    def test_calculate_graduated_penalties_both_violations(self, regulator):
        """Test graduated penalty calculation with both violations."""
        prices = np.array([50.0, 52.0, 48.0])

        fines, severities, multipliers = regulator._calculate_graduated_penalties(
            prices, parallel_violation=True, structural_break_violation=True
        )

        assert np.all(fines > 0)
        assert all(s == ViolationSeverity.CRITICAL for s in severities)
        assert len(multipliers) == 3

    def test_calculate_graduated_penalties_cumulative_multiplier(self, regulator):
        """Test graduated penalty calculation with cumulative multiplier."""
        prices = np.array([50.0, 52.0, 48.0])

        # Set up cumulative multipliers
        regulator.penalty_multipliers[0] = 2.0
        regulator.penalty_multipliers[1] = 1.5
        regulator.penalty_multipliers[2] = 1.0

        fines, severities, multipliers = regulator._calculate_graduated_penalties(
            prices, parallel_violation=True, structural_break_violation=False
        )

        assert np.all(fines > 0)
        assert multipliers[0] == 2.0
        assert multipliers[1] == 1.5
        assert multipliers[2] == 1.0

    def test_calculate_graduated_penalties_leniency_reduction(self, regulator):
        """Test graduated penalty calculation with leniency reduction."""
        prices = np.array([50.0, 52.0, 48.0])

        # Mock leniency program
        mock_leniency = Mock()
        mock_leniency.get_fine_reduction.side_effect = lambda i: 0.5 if i == 0 else 0.0
        regulator.leniency_program = mock_leniency

        fines, severities, multipliers = regulator._calculate_graduated_penalties(
            prices, parallel_violation=True, structural_break_violation=False
        )

        assert np.all(fines > 0)
        # Firm 0 should have reduced fine due to leniency
        assert fines[0] < fines[1]  # Assuming same base fine

    def test_get_monitoring_summary(self, regulator):
        """Test monitoring summary generation."""
        # Add some violations and data
        regulator.parallel_violations.append({"step": 1, "fines": [10.0, 10.0, 10.0]})
        regulator.structural_break_violations.append(
            {"step": 2, "fines": [20.0, 20.0, 20.0]}
        )
        regulator.total_fines_applied = 90.0
        regulator.market_volatility_history = [0.1, 0.2, 0.3]
        regulator.violation_counts = {0: 2, 1: 1}
        regulator.penalty_multipliers = {0: 1.5, 1: 1.2}

        summary = regulator.get_monitoring_summary()

        assert summary["total_violations"] == 2
        assert summary["parallel_violations"] == 1
        assert summary["structural_break_violations"] == 1
        assert summary["total_fines_applied"] == 90.0
        assert summary["average_risk_score"] == 0.0  # Placeholder
        assert summary["current_risk_score"] == 0.0  # Placeholder
        assert summary["market_volatility"] == 0.3
        assert summary["violation_counts"] == {0: 2, 1: 1}
        assert summary["penalty_multipliers"] == {0: 1.5, 1: 1.2}
        assert summary["market_volatility_history"] == [0.1, 0.2, 0.3]

    def test_reset(self, regulator):
        """Test regulator reset functionality."""
        # Add some data
        regulator.violation_counts = {0: 2, 1: 1}
        regulator.market_volatility_history = [0.1, 0.2, 0.3]
        regulator.penalty_multipliers = {0: 1.5, 1: 1.2}

        regulator.reset(n_firms=3)

        assert regulator.violation_counts == {}
        assert regulator.market_volatility_history == []
        assert regulator.penalty_multipliers == {}

    def test_violation_tracking(self, regulator):
        """Test violation tracking and penalty multiplier updates."""
        prices = np.array([50.0, 52.0, 48.0])

        # Add price history for parallel violation
        for _ in range(4):
            regulator.price_history.append(prices)

        # First violation
        result1 = regulator.monitor_step(prices, step=10)
        assert result1["parallel_violation"] is True

        # Check violation counts and multipliers
        assert regulator.violation_counts[0] == 1
        assert regulator.violation_counts[1] == 1
        assert regulator.violation_counts[2] == 1
        assert regulator.penalty_multipliers[0] == 1.2
        assert regulator.penalty_multipliers[1] == 1.2
        assert regulator.penalty_multipliers[2] == 1.2

        # Second violation
        result2 = regulator.monitor_step(prices, step=11)
        assert result2["parallel_violation"] is True

        # Check updated multipliers
        assert regulator.penalty_multipliers[0] == 1.44  # 1.2 * 1.2
        assert regulator.penalty_multipliers[1] == 1.44
        assert regulator.penalty_multipliers[2] == 1.44

    def test_max_penalty_multiplier_limit(self, regulator):
        """Test that penalty multipliers don't exceed maximum."""
        prices = np.array([50.0, 52.0, 48.0])

        # Add price history for parallel violation
        for _ in range(4):
            regulator.price_history.append(prices)

        # Set high initial multiplier
        regulator.penalty_multipliers[0] = 4.0

        result = regulator.monitor_step(prices, step=10)
        assert result["parallel_violation"] is True

        # Should be capped at max_penalty_multiplier
        assert regulator.penalty_multipliers[0] == 4.8  # 4.0 * 1.2

    def test_market_awareness_disabled(self):
        """Test behavior when market awareness is disabled."""
        regulator = EnhancedRegulator(
            use_market_awareness=False,
            parallel_threshold=5.0,
            parallel_steps=4,
            structural_break_threshold=30.0,
        )

        # Add price history
        for _ in range(4):
            regulator.price_history.append(np.array([50.0, 50.1, 50.2]))

        # Should still detect parallel pricing
        violation = regulator._detect_parallel_pricing_enhanced(
            step=10, market_volatility=0.5
        )
        assert violation is True

    def test_graduated_penalties_disabled(self):
        """Test behavior when graduated penalties are disabled."""
        regulator = EnhancedRegulator(
            use_graduated_penalties=False,
            parallel_threshold=5.0,
            parallel_steps=4,
            structural_break_threshold=30.0,
        )

        # Add price history for violation
        for _ in range(4):
            regulator.price_history.append(np.array([50.0, 50.1, 50.2]))

        result = regulator.monitor_step(np.array([50.0, 50.1, 50.2]), step=10)

        # Should still detect violations but use base penalty structure
        assert result["parallel_violation"] is True
        assert any(fine > 0 for fine in result["fines_applied"])

    def test_edge_case_empty_prices(self, regulator):
        """Test edge case with empty price arrays."""
        # Add some price history
        for _ in range(4):
            regulator.price_history.append(np.array([50.0, 52.0, 48.0]))

        # Test with empty prices
        empty_prices = np.array([])

        # This should handle gracefully
        result = regulator.monitor_step(empty_prices, step=10)
        assert result["step"] == 10
        assert result["fines_applied"] == []

    def test_edge_case_single_firm(self, regulator):
        """Test edge case with single firm."""
        single_firm_regulator = EnhancedRegulator()
        single_firm_regulator.reset(n_firms=1)

        # Add price history
        for _ in range(4):
            single_firm_regulator.price_history.append(np.array([50.0]))

        result = single_firm_regulator.monitor_step(np.array([50.0]), step=10)

        assert result["step"] == 10
        assert len(result["fines_applied"]) == 1
