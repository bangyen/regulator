"""Tests for offline training of the MLRegulator collusion classifier."""

import numpy as np

from regulator.agents.ml_regulator import MLRegulator
from regulator.experiments.experiment_runner import create_regulator
from regulator.experiments.ml_training import (
    simulate_labeled_windows,
    train_collusion_classifier,
)


def test_windows_are_balanced_and_shaped() -> None:
    X, y = simulate_labeled_windows(n_episodes=6, steps=20, window=5, seed=0)

    assert X.shape == (6 * 16, 20)
    assert y.mean() == 0.5
    assert np.isfinite(X).all()


def test_windows_are_deterministic() -> None:
    a = simulate_labeled_windows(n_episodes=4, steps=15, seed=3)
    b = simulate_labeled_windows(n_episodes=4, steps=15, seed=3)

    np.testing.assert_array_equal(a[0], b[0])


def test_classifier_generalises_to_new_episodes() -> None:
    classifier = train_collusion_classifier(seed=0, n_episodes=40, steps=40)
    X, y = simulate_labeled_windows(n_episodes=20, steps=40, seed=123)

    assert (classifier.predict(X) == y).mean() > 0.75


def test_ml_config_uses_pretrained_classifier() -> None:
    regulator = create_regulator("ml", seed=1)

    assert isinstance(regulator, MLRegulator)
    assert regulator.get_ml_statistics()["classifier_fitted"] is True
