"""
Tests for leniency program and whistleblower dynamics.

This module tests the leniency program implementation, including whistleblower
incentives, fine reductions, and the integration with the regulator system.
"""

import pytest
import numpy as np

from src.agents.leniency import (
    LeniencyProgram,
    LeniencyStatus,
    WhistleblowerAgent,
)
from src.agents.regulator import Regulator
from src.agents.firm_agents import (
    WhistleblowerTitForTatAgent,
    StrategicWhistleblowerAgent,
)
from src.cartel.cartel_env import CartelEnv


class TestLeniencyProgram:
    """Test cases for the LeniencyProgram class."""

    def test_init_valid_parameters(self) -> None:
        """Test initialization with valid parameters."""
        program = LeniencyProgram(
            leniency_reduction=0.5,
            max_reports_per_episode=3,
            evidence_threshold=0.7,
            audit_probability=0.1,
            seed=42,
        )

        assert program.leniency_reduction == 0.5
        assert program.max_reports_per_episode == 3
        assert program.evidence_threshold == 0.7
        assert program.audit_probability == 0.1

    def test_init_invalid_parameters(self) -> None:
        """Test initialization with invalid parameters."""
        with pytest.raises(
            ValueError, match="Leniency reduction must be between 0.0 and 1.0"
        ):
            LeniencyProgram(leniency_reduction=1.5)

        with pytest.raises(
            ValueError, match="Max reports per episode must be at least 1"
        ):
            LeniencyProgram(max_reports_per_episode=0)

        with pytest.raises(
            ValueError, match="Evidence threshold must be between 0.0 and 1.0"
        ):
            LeniencyProgram(evidence_threshold=-0.1)

        with pytest.raises(
            ValueError, match="Audit probability must be between 0.0 and 1.0"
        ):
            LeniencyProgram(audit_probability=1.5)

    def test_reset(self) -> None:
        """Test resetting the leniency program."""
        program = LeniencyProgram(seed=42)
        program.reset(3)

        assert len(program.reports) == 0
        assert len(program.firm_status) == 3
        assert all(
            status == LeniencyStatus.AVAILABLE
            for status in program.firm_status.values()
        )
        assert len(program.audit_threats) == 3
        assert len(program.collusion_evidence) == 3

    def test_update_collusion_evidence(self) -> None:
        """Test updating collusion evidence for firms."""
        program = LeniencyProgram(seed=42)
        program.reset(3)

        program.update_collusion_evidence(0, 0.8, 10)
        assert program.collusion_evidence[0] == 0.8
        assert program.audit_threats[0] > 0

        # Test persistence - higher evidence should update
        program.update_collusion_evidence(0, 0.9, 11)
        assert program.collusion_evidence[0] == 0.9

        # Test persistence - lower evidence should not update
        program.update_collusion_evidence(0, 0.7, 12)
        assert program.collusion_evidence[0] == 0.9

    def test_can_report(self) -> None:
        """Test checking if a firm can report."""
        program = LeniencyProgram(max_reports_per_episode=2, seed=42)
        program.reset(3)

        # Initially all firms can report
        assert program.can_report(0)
        assert program.can_report(1)
        assert program.can_report(2)

        # After submitting a report, firm can't report again
        program.submit_report(0, [1, 2], 0.8, 10)
        assert not program.can_report(0)
        assert program.can_report(1)
        assert program.can_report(2)

        # After reaching max reports, no one can report
        program.submit_report(1, [0, 2], 0.8, 11)
        assert not program.can_report(0)
        assert not program.can_report(1)
        assert not program.can_report(2)

    def test_submit_report_success(self) -> None:
        """Test successful report submission."""
        program = LeniencyProgram(evidence_threshold=0.7, seed=42)
        program.reset(3)

        success = program.submit_report(0, [1, 2], 0.8, 10)
        assert success
        assert len(program.reports) == 1
        assert program.firm_status[0] == LeniencyStatus.APPLIED

        report = program.reports[0]
        assert report.firm_id == 0
        assert report.reported_firms == [1, 2]
        assert report.evidence_strength == 0.8
        assert report.fine_reduction == 0.5  # default leniency reduction

    def test_submit_report_failure(self) -> None:
        """Test failed report submission."""
        program = LeniencyProgram(evidence_threshold=0.7, seed=42)
        program.reset(3)

        # Evidence too weak
        success = program.submit_report(0, [1, 2], 0.6, 10)
        assert not success
        assert len(program.reports) == 0

        # Firm reporting itself
        success = program.submit_report(0, [0, 1], 0.8, 10)
        assert not success

        # Firm already used leniency
        program.submit_report(0, [1, 2], 0.8, 10)
        success = program.submit_report(0, [1, 2], 0.8, 11)
        assert not success

    def test_get_fine_reduction(self) -> None:
        """Test getting fine reduction for firms."""
        program = LeniencyProgram(leniency_reduction=0.6, seed=42)
        program.reset(3)

        # No leniency applied
        assert program.get_fine_reduction(0) == 0.0

        # Apply leniency
        program.submit_report(0, [1, 2], 0.8, 10)
        assert program.get_fine_reduction(0) == 0.6
        assert program.get_fine_reduction(1) == 0.0

    def test_get_whistleblower_incentive(self) -> None:
        """Test calculating whistleblower incentives."""
        program = LeniencyProgram(leniency_reduction=0.5, seed=42)
        program.reset(3)

        # Update evidence to create audit threat
        program.update_collusion_evidence(0, 0.8, 10)

        current_fine = 100.0
        collusion_probability = 0.3

        incentive = program.get_whistleblower_incentive(
            0, current_fine, collusion_probability
        )

        # Expected calculation:
        # Expected fine without report: 100 * 0.3 = 30
        # Expected fine with report: 100 * 0.5 * 0.3 = 15
        # Audit benefit: audit_threat * 100 * 0.5
        expected_base_incentive = 30 - 15  # 15
        assert incentive >= expected_base_incentive

    def test_should_whistleblow(self) -> None:
        """Test whistleblow decision logic."""
        program = LeniencyProgram(seed=42)
        program.reset(3)

        # High incentive scenario
        program.update_collusion_evidence(0, 0.9, 10)
        should_whistleblow = program.should_whistleblow(0, 100.0, 0.5, threshold=5.0)
        assert should_whistleblow

        # Low incentive scenario
        should_whistleblow = program.should_whistleblow(0, 10.0, 0.1, threshold=50.0)
        assert not should_whistleblow

    def test_get_program_summary(self) -> None:
        """Test getting program summary."""
        program = LeniencyProgram(seed=42)
        program.reset(3)

        summary = program.get_program_summary()
        assert summary["total_reports"] == 0
        assert summary["firms_with_leniency"] == 0
        assert not summary["max_reports_reached"]
        assert summary["leniency_reduction"] == 0.5

        # Add some reports
        program.submit_report(0, [1, 2], 0.8, 10)
        program.update_collusion_evidence(1, 0.7, 11)

        summary = program.get_program_summary()
        assert summary["total_reports"] == 1
        assert summary["firms_with_leniency"] == 1
        assert summary["average_evidence_strength"] > 0


class TestWhistleblowerAgent:
    """Test cases for the WhistleblowerAgent class."""

    def test_init(self) -> None:
        """Test initialization."""
        program = LeniencyProgram(seed=42)
        agent = WhistleblowerAgent(
            0, program, whistleblow_threshold=10.0, risk_aversion=1.2
        )

        assert agent.agent_id == 0
        assert agent.whistleblow_threshold == 10.0
        assert agent.risk_aversion == 1.2

    def test_evaluate_whistleblow_decision(self) -> None:
        """Test whistleblow decision evaluation."""
        program = LeniencyProgram(seed=42)
        program.reset(3)
        agent = WhistleblowerAgent(0, program, whistleblow_threshold=10.0)

        # High incentive scenario
        program.update_collusion_evidence(0, 0.8, 10)
        should_whistleblow, incentive = agent.evaluate_whistleblow_decision(
            100.0, 0.4, 10
        )

        assert incentive > 0
        assert len(agent.whistleblow_history) == 1
        assert agent.whistleblow_history[0] == (10, should_whistleblow, incentive)

    def test_get_whistleblow_statistics(self) -> None:
        """Test getting whistleblow statistics."""
        program = LeniencyProgram(seed=42)
        program.reset(3)
        agent = WhistleblowerAgent(0, program)

        # No decisions yet
        stats = agent.get_whistleblow_statistics()
        assert stats["total_decisions"] == 0
        assert stats["whistleblow_rate"] == 0.0

        # Add some decisions
        agent.evaluate_whistleblow_decision(100.0, 0.3, 10)
        agent.evaluate_whistleblow_decision(50.0, 0.2, 11)

        stats = agent.get_whistleblow_statistics()
        assert stats["total_decisions"] == 2
        assert stats["average_incentive"] > 0

    def test_reset(self) -> None:
        """Test resetting the agent."""
        program = LeniencyProgram(seed=42)
        program.reset(3)
        agent = WhistleblowerAgent(0, program)

        agent.evaluate_whistleblow_decision(100.0, 0.3, 10)
        assert len(agent.whistleblow_history) == 1

        agent.reset()
        assert len(agent.whistleblow_history) == 0


class TestWhistleblowerFirmAgents:
    """Test cases for whistleblower firm agents."""

    def test_whistleblower_tit_for_tat_agent(self) -> None:
        """Test WhistleblowerTitForTatAgent."""
        program = LeniencyProgram(seed=42)
        program.reset(3)
        agent = WhistleblowerTitForTatAgent(0, program)

        # Test pricing behavior (inherited from TitForTatAgent)
        env = CartelEnv(n_firms=3, seed=42)
        observation = np.array([10.0, 12.0, 11.0, 0.0])  # prices + demand shock

        # First call with no history should use default
        price = agent.choose_price(observation, env)
        assert env.price_min <= price <= env.price_max

        # Test whistleblow evaluation
        whistled, incentive = agent.evaluate_whistleblow_opportunity(
            100.0, 0.3, 10, [1, 2]
        )
        assert isinstance(whistled, bool)
        assert incentive >= 0

    def test_strategic_whistleblower_agent(self) -> None:
        """Test StrategicWhistleblowerAgent."""
        program = LeniencyProgram(seed=42)
        program.reset(3)
        agent = StrategicWhistleblowerAgent(0, program)

        # Test pricing behavior (inherited from BestResponseAgent)
        env = CartelEnv(n_firms=3, seed=42)
        observation = np.array([10.0, 12.0, 11.0, 0.0])

        price = agent.choose_price(observation, env)
        assert env.price_min <= price <= env.price_max

        # Test whistleblow evaluation
        whistled, incentive = agent.evaluate_whistleblow_opportunity(
            100.0, 0.3, 10, [1, 2]
        )
        assert isinstance(whistled, bool)
        assert incentive >= 0

    def test_combined_statistics(self) -> None:
        """Test getting combined statistics from whistleblower agents."""
        program = LeniencyProgram(seed=42)
        program.reset(3)
        agent = WhistleblowerTitForTatAgent(0, program)

        # Add some history
        agent.update_history(10.0, np.array([12.0, 11.0]))
        agent.evaluate_whistleblow_decision(100.0, 0.3, 10)

        stats = agent.get_combined_statistics()
        assert "price_history_length" in stats
        assert "whistleblow_count" in stats
        assert stats["price_history_length"] == 1


class TestRegulatorLeniencyIntegration:
    """Test cases for regulator integration with leniency program."""

    def test_regulator_with_leniency_enabled(self) -> None:
        """Test regulator with leniency program enabled."""
        regulator = Regulator(
            leniency_enabled=True, leniency_reduction=0.6, fine_amount=50.0, seed=42
        )

        assert regulator.leniency_enabled
        assert regulator.leniency_program is not None
        assert regulator.leniency_reduction == 0.6

    def test_regulator_with_leniency_disabled(self) -> None:
        """Test regulator with leniency program disabled."""
        regulator = Regulator(leniency_enabled=False, seed=42)

        assert not regulator.leniency_enabled
        assert regulator.leniency_program is None

    def test_regulator_reset_with_leniency(self) -> None:
        """Test regulator reset with leniency program."""
        regulator = Regulator(leniency_enabled=True, seed=42)

        # Should work with n_firms
        regulator.reset(n_firms=3)
        assert regulator.leniency_program is not None
        assert len(regulator.leniency_program.firm_status) == 3

        # Should fail without n_firms
        with pytest.raises(ValueError, match="n_firms must be provided"):
            regulator.reset()

    def test_submit_leniency_report(self) -> None:
        """Test submitting leniency reports through regulator."""
        regulator = Regulator(leniency_enabled=True, seed=42)
        regulator.reset(n_firms=3)

        # Successful report
        success = regulator.submit_leniency_report(0, [1, 2], 0.8, 10)
        assert success

        # Failed report (leniency disabled)
        regulator_no_leniency = Regulator(leniency_enabled=False)
        success = regulator_no_leniency.submit_leniency_report(0, [1, 2], 0.8, 10)
        assert not success

    def test_get_leniency_status(self) -> None:
        """Test getting leniency status through regulator."""
        regulator = Regulator(leniency_enabled=True, seed=42)
        regulator.reset(n_firms=3)

        # Initially available
        status = regulator.get_leniency_status(0)
        assert status == LeniencyStatus.AVAILABLE

        # After report, applied
        regulator.submit_leniency_report(0, [1, 2], 0.8, 10)
        status = regulator.get_leniency_status(0)
        assert status == LeniencyStatus.APPLIED

    def test_get_whistleblower_incentive(self) -> None:
        """Test getting whistleblower incentives through regulator."""
        regulator = Regulator(leniency_enabled=True, seed=42)
        regulator.reset(n_firms=3)

        # Update evidence
        assert regulator.leniency_program is not None
        regulator.leniency_program.update_collusion_evidence(0, 0.8, 10)

        incentive = regulator.get_whistleblower_incentive(0, 100.0, 0.3)
        assert incentive >= 0

        # No incentive when leniency disabled
        regulator_no_leniency = Regulator(leniency_enabled=False)
        incentive = regulator_no_leniency.get_whistleblower_incentive(0, 100.0, 0.3)
        assert incentive == 0.0

    def test_get_leniency_summary(self) -> None:
        """Test getting leniency summary through regulator."""
        regulator = Regulator(leniency_enabled=True, seed=42)
        regulator.reset(n_firms=3)

        summary = regulator.get_leniency_summary()
        assert summary["leniency_enabled"]
        assert "total_reports" in summary

        # No leniency
        regulator_no_leniency = Regulator(leniency_enabled=False)
        summary = regulator_no_leniency.get_leniency_summary()
        assert not summary["leniency_enabled"]

    def test_fine_reduction_in_monitoring(self) -> None:
        """Test that fines are reduced for whistleblowers in monitoring."""
        regulator = Regulator(
            leniency_enabled=True, leniency_reduction=0.5, fine_amount=100.0, seed=42
        )
        regulator.reset(n_firms=3)

        # Submit leniency report for firm 0
        regulator.submit_leniency_report(0, [1, 2], 0.8, 10)

        # Create parallel pricing violation
        prices = np.array([50.0, 50.1, 50.2])  # Very close prices
        for step in range(5):  # Need enough steps for parallel detection
            detection_results = regulator.monitor_step(prices, step)

        # Check that firm 0 has reduced fine
        fines = detection_results["fines_applied"]
        assert fines[0] < fines[1]  # Firm 0 should have reduced fine
        assert fines[0] == 50.0  # 50% reduction
        assert fines[1] == 100.0  # No reduction


class TestLeniencyWelfareEffects:
    """Test cases for welfare effects of leniency program."""

    def test_whistleblowing_reduces_fine_for_reporting_firm(self) -> None:
        """Test that whistleblowing reduces fine for the reporting firm."""
        program = LeniencyProgram(leniency_reduction=0.5, seed=42)
        program.reset(3)

        # Firm 0 submits leniency report
        success = program.submit_report(0, [1, 2], 0.8, 10)
        assert success

        # Check fine reduction
        reduction = program.get_fine_reduction(0)
        assert reduction == 0.5

        # Other firms get no reduction
        assert program.get_fine_reduction(1) == 0.0
        assert program.get_fine_reduction(2) == 0.0

    def test_collusion_probability_drops_with_leniency(self) -> None:
        """Test that collusion probability drops with leniency program."""
        # This is a conceptual test - in practice, this would be measured
        # through repeated simulations comparing with/without leniency

        # With leniency, firms have incentive to break collusion
        program_with_leniency = LeniencyProgram(seed=42)
        program_with_leniency.reset(3)

        # Simulate high collusion evidence
        for firm_id in range(3):
            program_with_leniency.update_collusion_evidence(firm_id, 0.9, 10)

        # Check that firms have high incentive to whistleblow
        incentives = []
        for firm_id in range(3):
            incentive = program_with_leniency.get_whistleblower_incentive(
                firm_id, 100.0, 0.8
            )
            incentives.append(incentive)

        # At least one firm should have high incentive
        assert max(incentives) > 10.0

    def test_welfare_improves_with_leniency(self) -> None:
        """Test that welfare improves with leniency program enabled."""
        # This is a conceptual test - in practice, this would be measured
        # through consumer surplus calculations in simulations

        # Leniency program should reduce collusion, leading to:
        # 1. Lower prices (better for consumers)
        # 2. More competitive behavior
        # 3. Higher total welfare

        program = LeniencyProgram(seed=42)
        program.reset(3)

        # High evidence scenario should create strong whistleblower incentives
        for firm_id in range(3):
            program.update_collusion_evidence(firm_id, 0.8, 10)

        # Check that the program creates incentives to break collusion
        total_incentive = sum(
            program.get_whistleblower_incentive(firm_id, 100.0, 0.7)
            for firm_id in range(3)
        )

        # Should create meaningful incentives
        assert total_incentive > 0


class TestLeniencyEdgeCases:
    """Test edge cases and error conditions."""

    def test_evidence_strength_boundaries(self) -> None:
        """Test evidence strength at boundaries."""
        program = LeniencyProgram(evidence_threshold=0.5, seed=42)
        program.reset(3)

        # Exactly at threshold should work
        success = program.submit_report(0, [1, 2], 0.5, 10)
        assert success

        # Just below threshold should fail
        program.reset(3)
        success = program.submit_report(0, [1, 2], 0.499, 10)
        assert not success

    def test_max_reports_limit(self) -> None:
        """Test maximum reports limit."""
        program = LeniencyProgram(max_reports_per_episode=2, seed=42)
        program.reset(3)

        # First two reports should succeed
        assert program.submit_report(0, [1, 2], 0.8, 10)
        assert program.submit_report(1, [0, 2], 0.8, 11)

        # Third report should fail
        assert not program.submit_report(2, [0, 1], 0.8, 12)

    def test_empty_reported_firms_list(self) -> None:
        """Test reporting with empty firms list."""
        program = LeniencyProgram(seed=42)
        program.reset(3)

        # Should fail with empty list
        success = program.submit_report(0, [], 0.8, 10)
        assert not success

    def test_invalid_firm_ids(self) -> None:
        """Test with invalid firm IDs."""
        program = LeniencyProgram(seed=42)
        program.reset(3)

        # Invalid firm ID in reported firms
        success = program.submit_report(0, [1, 2, 5], 0.8, 10)
        # This should still work as we don't validate firm IDs in reported list
        assert success

        # Invalid reporting firm ID should return 0.0 (no reduction)
        reduction = program.get_fine_reduction(5)
        assert reduction == 0.0

    def test_zero_fine_scenarios(self) -> None:
        """Test scenarios with zero fines."""
        program = LeniencyProgram(seed=42)
        program.reset(3)

        # Zero fine should result in zero incentive
        incentive = program.get_whistleblower_incentive(0, 0.0, 0.5)
        assert incentive == 0.0

        # Zero collusion probability should result in zero incentive
        incentive = program.get_whistleblower_incentive(0, 100.0, 0.0)
        assert incentive == 0.0
