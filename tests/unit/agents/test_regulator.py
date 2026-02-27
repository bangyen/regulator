"""
Unit tests for the minimalist Regulator.
"""

import numpy as np
from regulator.agents.regulator import Regulator


class TestRegulator:
    """Minimalist test suite for Regulator."""

    def test_initialization(self) -> None:
        """Test regulator initialization."""
        regulator = Regulator(parallel_threshold=2.0, parallel_steps=3)
        assert regulator.parallel_threshold == 2.0
        assert regulator.parallel_steps == 3
        assert len(regulator.price_history) == 0

    def test_parallel_pricing_detection(self) -> None:
        """Test detection of parallel pricing."""
        # Need 2 steps of parallel pricing to trigger
        regulator = Regulator(parallel_threshold=1.0, parallel_steps=2)

        # Step 1: Parallel
        res1 = regulator.monitor_step(np.array([20.0, 20.1]), step=0)
        assert not res1["parallel_violation"]

        # Step 2: Parallel -> Should trigger
        res2 = regulator.monitor_step(np.array([20.2, 20.3]), step=1)
        assert res2["parallel_violation"]

    def test_structural_break_detection(self) -> None:
        """Test detection of structural breaks."""
        regulator = Regulator(structural_break_threshold=10.0)

        # Step 1: Normal
        regulator.monitor_step(np.array([20.0, 20.0]), step=0)

        # Step 2: Sudden jump
        res = regulator.monitor_step(np.array([35.0, 35.0]), step=1)
        assert res["structural_break_violation"]

    def test_penalty_application(self) -> None:
        """Test that penalties reduce rewards."""
        # Disable leniency for simple test
        regulator = Regulator(fine_amount=50.0, leniency_enabled=False)
        rewards = np.array([100.0, 100.0])

        # Simulate violation results
        res = {
            "violation_detected": True,
            "parallel_violation": True,
            "structural_break_violation": False,
            "fines_applied": np.zeros(2),
            "step": 0,
        }

        # The penalty logic in regulator.py:
        # if any violation:
        #   if self.violation_start_step is None: self.violation_start_step = ...
        #   fine = self.fine_amount * (1 + 0.1 * duration)

        modified_rewards = regulator.apply_penalties(rewards, res)
        # Note: apply_penalties returns a NEW array
        assert np.all(modified_rewards < 100.0)
        assert np.all(res["fines_applied"] >= 50.0)

    def test_reset(self) -> None:
        """Test that reset clears histories."""
        regulator = Regulator(leniency_enabled=False)
        regulator.price_history.append(np.array([10.0]))
        regulator.reset()
        assert len(regulator.price_history) == 0

        # Test with leniency
        regulator_len = Regulator(leniency_enabled=True)
        regulator_len.reset(n_firms=2)
        assert len(regulator_len.price_history) == 0
