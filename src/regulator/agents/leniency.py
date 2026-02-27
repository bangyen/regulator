"""
Leniency program and whistleblower dynamics for cartel detection.

This module implements a leniency program that allows firms to report collusion
in exchange for reduced fines. It includes whistleblower dynamics where firms
can strategically decide to report collusion when facing audit threats or
detection risks.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np


class LeniencyStatus(Enum):
    """Status of leniency program for a firm."""

    NOT_APPLICABLE = "not_applicable"
    AVAILABLE = "available"
    APPLIED = "applied"
    EXPIRED = "expired"


@dataclass
class LeniencyReport:
    """Represents a leniency report from a firm."""

    firm_id: int
    step: int
    reported_firms: List[int]
    evidence_strength: float
    fine_reduction: float


class LeniencyProgram:
    """
    Implements a leniency program for cartel detection.

    This class manages the leniency program where firms can report collusion
    in exchange for reduced fines. It tracks whistleblower reports, manages
    fine reductions, and provides incentives for firms to cooperate with
    regulatory investigations.
    """

    def __init__(
        self,
        leniency_reduction: float = 0.5,
        max_reports_per_episode: int = 3,
        evidence_threshold: float = 0.7,
        audit_probability: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the leniency program.

        Args:
            leniency_reduction: Fraction of fine reduction for whistleblowers (0.0-1.0)
            max_reports_per_episode: Maximum number of reports allowed per episode
            evidence_threshold: Minimum evidence strength required for valid report
            audit_probability: Probability of audit when collusion is suspected
            seed: Random seed for reproducibility
        """
        if not 0.0 <= leniency_reduction <= 1.0:
            raise ValueError("Leniency reduction must be between 0.0 and 1.0")
        if max_reports_per_episode < 1:
            raise ValueError("Max reports per episode must be at least 1")
        if not 0.0 <= evidence_threshold <= 1.0:
            raise ValueError("Evidence threshold must be between 0.0 and 1.0")
        if not 0.0 <= audit_probability <= 1.0:
            raise ValueError("Audit probability must be between 0.0 and 1.0")

        self.leniency_reduction = leniency_reduction
        self.max_reports_per_episode = max_reports_per_episode
        self.evidence_threshold = evidence_threshold
        self.audit_probability = audit_probability
        self.np_random = np.random.default_rng(seed)

        # Episode state
        self.reports: List[LeniencyReport] = []
        self.firm_status: Dict[int, LeniencyStatus] = {}
        self.audit_threats: Dict[int, float] = {}  # firm_id -> threat_level
        self.collusion_evidence: Dict[int, float] = {}  # firm_id -> evidence_strength

    def reset(self, n_firms: int) -> None:
        """
        Reset the leniency program for a new episode.

        Args:
            n_firms: Number of firms in the market
        """
        self.reports.clear()
        self.firm_status = {i: LeniencyStatus.AVAILABLE for i in range(n_firms)}
        self.audit_threats = {i: 0.0 for i in range(n_firms)}
        self.collusion_evidence = {i: 0.0 for i in range(n_firms)}

    def update_collusion_evidence(
        self, firm_id: int, evidence_strength: float, step: int
    ) -> None:
        """
        Update the evidence strength for a firm's collusion behavior.

        Args:
            firm_id: ID of the firm
            evidence_strength: Strength of evidence (0.0-1.0)
            step: Current step number
        """
        if not 0.0 <= evidence_strength <= 1.0:
            raise ValueError("Evidence strength must be between 0.0 and 1.0")

        # Update evidence with some persistence
        current_evidence = self.collusion_evidence.get(firm_id, 0.0)
        self.collusion_evidence[firm_id] = max(current_evidence, evidence_strength)

        # Update audit threat based on evidence
        self._update_audit_threat(firm_id, evidence_strength)

    def _update_audit_threat(self, firm_id: int, evidence_strength: float) -> None:
        """
        Update the audit threat level for a firm based on evidence.

        Args:
            firm_id: ID of the firm
            evidence_strength: Strength of collusion evidence
        """
        # Audit threat increases with evidence strength
        base_threat = evidence_strength * self.audit_probability
        current_threat = self.audit_threats.get(firm_id, 0.0)

        # Threat accumulates over time but with some decay
        self.audit_threats[firm_id] = min(1.0, current_threat * 0.9 + base_threat)

    def can_report(self, firm_id: int) -> bool:
        """
        Check if a firm can submit a leniency report.

        Args:
            firm_id: ID of the firm

        Returns:
            True if the firm can report, False otherwise
        """
        if firm_id not in self.firm_status:
            return False

        # Check if firm has already used leniency
        if self.firm_status[firm_id] == LeniencyStatus.APPLIED:
            return False

        # Check if maximum reports reached
        if len(self.reports) >= self.max_reports_per_episode:
            return False

        return True

    def submit_report(
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
        if not self.can_report(firm_id):
            return False

        if evidence_strength < self.evidence_threshold:
            return False

        if firm_id in reported_firms:
            return False  # Can't report yourself

        if not reported_firms:
            return False  # Can't report empty list

        # Create and store the report
        fine_reduction = self.leniency_reduction
        report = LeniencyReport(
            firm_id=firm_id,
            step=step,
            reported_firms=reported_firms.copy(),
            evidence_strength=evidence_strength,
            fine_reduction=fine_reduction,
        )

        self.reports.append(report)
        self.firm_status[firm_id] = LeniencyStatus.APPLIED

        return True

    def get_fine_reduction(self, firm_id: int) -> float:
        """
        Get the fine reduction for a firm based on leniency status.

        Args:
            firm_id: ID of the firm

        Returns:
            Fine reduction factor (0.0 = no reduction, 1.0 = full reduction)
        """
        if firm_id not in self.firm_status:
            return 0.0

        if self.firm_status[firm_id] == LeniencyStatus.APPLIED:
            # Find the report for this firm
            for report in self.reports:
                if report.firm_id == firm_id:
                    return report.fine_reduction

        return 0.0

    def get_whistleblower_incentive(
        self, firm_id: int, current_fine: float, collusion_probability: float
    ) -> float:
        """
        Calculate the incentive for a firm to whistleblow.

        Args:
            firm_id: ID of the firm
            current_fine: Current fine amount if caught
            collusion_probability: Probability of being caught in collusion

        Returns:
            Expected benefit from whistleblowing
        """
        if not self.can_report(firm_id):
            return 0.0

        # Expected fine without whistleblowing
        expected_fine_no_report = current_fine * collusion_probability

        # Expected fine with whistleblowing (reduced)
        expected_fine_with_report = (
            current_fine * (1 - self.leniency_reduction) * collusion_probability
        )

        # Additional benefit from audit threat reduction
        audit_threat = self.audit_threats.get(firm_id, 0.0)
        audit_benefit = (
            audit_threat * current_fine * 0.5
        )  # Assume 50% audit fine reduction

        incentive = expected_fine_no_report - expected_fine_with_report + audit_benefit
        return max(0.0, incentive)

    def should_whistleblow(
        self,
        firm_id: int,
        current_fine: float,
        collusion_probability: float,
        threshold: float = 10.0,
    ) -> bool:
        """
        Determine if a firm should whistleblow based on incentives.

        Args:
            firm_id: ID of the firm
            current_fine: Current fine amount if caught
            collusion_probability: Probability of being caught in collusion
            threshold: Minimum incentive threshold to whistleblow

        Returns:
            True if firm should whistleblow, False otherwise
        """
        if not self.can_report(firm_id):
            return False

        incentive = self.get_whistleblower_incentive(
            firm_id, current_fine, collusion_probability
        )
        return incentive >= threshold

    def get_audit_threat(self, firm_id: int) -> float:
        """
        Get the current audit threat level for a firm.

        Args:
            firm_id: ID of the firm

        Returns:
            Audit threat level (0.0-1.0)
        """
        return self.audit_threats.get(firm_id, 0.0)

    def get_collusion_evidence(self, firm_id: int) -> float:
        """
        Get the current collusion evidence strength for a firm.

        Args:
            firm_id: ID of the firm

        Returns:
            Evidence strength (0.0-1.0)
        """
        return self.collusion_evidence.get(firm_id, 0.0)

    def get_program_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the leniency program's current state.

        Returns:
            Dictionary containing program statistics
        """
        total_reports = len(self.reports)
        firms_with_leniency = sum(
            1
            for status in self.firm_status.values()
            if status == LeniencyStatus.APPLIED
        )

        avg_evidence = (
            np.mean(list(self.collusion_evidence.values()))
            if self.collusion_evidence
            else 0.0
        )
        avg_audit_threat = (
            np.mean(list(self.audit_threats.values())) if self.audit_threats else 0.0
        )

        return {
            "total_reports": total_reports,
            "firms_with_leniency": firms_with_leniency,
            "max_reports_reached": total_reports >= self.max_reports_per_episode,
            "average_evidence_strength": float(avg_evidence),
            "average_audit_threat": float(avg_audit_threat),
            "leniency_reduction": self.leniency_reduction,
            "evidence_threshold": self.evidence_threshold,
        }

    def get_reports_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of all leniency reports.

        Returns:
            List of dictionaries containing report details
        """
        return [
            {
                "firm_id": report.firm_id,
                "step": report.step,
                "reported_firms": report.reported_firms,
                "evidence_strength": report.evidence_strength,
                "fine_reduction": report.fine_reduction,
            }
            for report in self.reports
        ]


class WhistleblowerAgent:
    """
    Agent that can strategically whistleblow based on leniency incentives.

    This agent extends the base firm agent functionality to include
    whistleblowing decisions based on leniency program incentives and
    audit threat levels.
    """

    def __init__(
        self,
        agent_id: int,
        leniency_program: LeniencyProgram,
        whistleblow_threshold: float = 10.0,
        risk_aversion: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the whistleblower agent.

        Args:
            agent_id: Unique identifier for this agent
            leniency_program: The leniency program instance
            whistleblow_threshold: Minimum incentive to whistleblow
            risk_aversion: Risk aversion parameter (higher = more risk averse)
            seed: Random seed for reproducibility
        """
        self.agent_id = agent_id
        self.leniency_program = leniency_program
        self.whistleblow_threshold = whistleblow_threshold
        self.risk_aversion = risk_aversion
        self.np_random = np.random.default_rng(seed)

        # Track whistleblowing history
        self.whistleblow_history: List[Tuple[int, bool, float]] = (
            []
        )  # (step, whistled, incentive)

    def evaluate_whistleblow_decision(
        self,
        current_fine: float,
        collusion_probability: float,
        step: int,
    ) -> Tuple[bool, float]:
        """
        Evaluate whether to whistleblow based on current incentives.

        Args:
            current_fine: Current fine amount if caught
            collusion_probability: Probability of being caught in collusion
            step: Current step number

        Returns:
            Tuple of (should_whistleblow, incentive_value)
        """
        incentive = self.leniency_program.get_whistleblower_incentive(
            self.agent_id, current_fine, collusion_probability
        )

        # Adjust threshold based on risk aversion
        adjusted_threshold = self.whistleblow_threshold * self.risk_aversion

        should_whistleblow = incentive >= adjusted_threshold

        # Record decision
        self.whistleblow_history.append((step, should_whistleblow, incentive))

        return should_whistleblow, incentive

    def get_whistleblow_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about whistleblowing behavior.

        Returns:
            Dictionary containing whistleblowing statistics
        """
        if not self.whistleblow_history:
            return {
                "total_decisions": 0,
                "whistleblow_count": 0,
                "whistleblow_rate": 0.0,
                "average_incentive": 0.0,
                "max_incentive": 0.0,
            }

        decisions = len(self.whistleblow_history)
        whistleblow_count = sum(
            1 for _, whistled, _ in self.whistleblow_history if whistled
        )
        incentives = [incentive for _, _, incentive in self.whistleblow_history]

        return {
            "total_decisions": decisions,
            "whistleblow_count": whistleblow_count,
            "whistleblow_rate": whistleblow_count / decisions,
            "average_incentive": float(np.mean(incentives)),
            "max_incentive": float(np.max(incentives)),
            "min_incentive": float(np.min(incentives)),
        }

    def reset(self) -> None:
        """Reset the agent's whistleblowing history."""
        self.whistleblow_history.clear()
