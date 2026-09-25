"""
Tests for the ML-enhanced regulator.

This module tests the MLRegulator class including ML-based detection,
feature extraction, and model training functionality.
"""

from unittest.mock import patch

import numpy as np

from regulator.agents.ml_regulator import MLRegulator


class TestMLRegulator:
    """Test suite for MLRegulator class."""

    def test_initialization_default_params(self) -> None:
        """Test ML regulator initialization with default parameters."""
        regulator = MLRegulator(seed=42)

        assert regulator.use_ml_detection is True
        assert regulator.ml_anomaly_threshold == 0.1
        assert regulator.ml_collusion_threshold == 0.7
        assert regulator.feature_window_size == 10
        assert regulator.retrain_frequency == 50
        assert regulator.anomaly_detector is not None
        # No classifier unless one is supplied or fit offline
        assert regulator.collusion_classifier is None

    def test_initialization_custom_params(self) -> None:
        """Test ML regulator initialization with custom parameters."""
        regulator = MLRegulator(
            use_ml_detection=False,
            ml_anomaly_threshold=0.2,
            ml_collusion_threshold=0.8,
            feature_window_size=15,
            retrain_frequency=100,
            seed=42,
        )

        assert regulator.use_ml_detection is False
        assert regulator.ml_anomaly_threshold == 0.2
        assert regulator.ml_collusion_threshold == 0.8
        assert regulator.feature_window_size == 15
        assert regulator.retrain_frequency == 100

    def test_feature_extraction(self) -> None:
        """Test feature extraction from price history."""
        regulator = MLRegulator(seed=42)

        # Create sample price history
        price_history = [
            np.array([50.0, 55.0, 60.0]),
            np.array([52.0, 54.0, 58.0]),
            np.array([48.0, 56.0, 62.0]),
            np.array([51.0, 53.0, 59.0]),
            np.array([49.0, 57.0, 61.0]),
        ]

        features = regulator._extract_features(price_history)

        # Check that features are extracted correctly
        assert len(features) == 20  # Expected feature vector size
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32

        # Check that features contain reasonable values
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_feature_extraction_insufficient_history(self) -> None:
        """Test feature extraction with insufficient price history."""
        regulator = MLRegulator(seed=42)

        # Test with empty history
        features = regulator._extract_features([])
        assert len(features) == 20
        assert np.all(features == 0.0)

        # Test with single step
        price_history = [np.array([50.0, 55.0, 60.0])]
        features = regulator._extract_features(price_history)
        assert len(features) == 20
        assert not np.any(np.isnan(features))

    def test_ml_anomaly_detection(self) -> None:
        """Test ML-based anomaly detection."""
        regulator = MLRegulator(seed=42)

        # Create sample features
        features = np.random.rand(20).astype(np.float32)

        # Test anomaly detection
        is_anomaly, anomaly_score = regulator._detect_ml_anomalies(features)

        assert isinstance(is_anomaly, bool)
        assert isinstance(anomaly_score, float)
        assert not np.isnan(anomaly_score)

    def test_ml_collusion_classification(self) -> None:
        """Test ML-based collusion classification."""
        regulator = MLRegulator(seed=42)

        # Create sample features
        features = np.random.rand(20).astype(np.float32)

        # Test collusion classification
        is_collusion, collusion_prob = regulator._classify_collusion(features)

        assert isinstance(is_collusion, bool)
        assert isinstance(collusion_prob, float)
        assert 0.0 <= collusion_prob <= 1.0

    def test_ml_detection_disabled(self) -> None:
        """Test ML detection when disabled."""
        regulator = MLRegulator(use_ml_detection=False, seed=42)

        features = np.random.rand(20).astype(np.float32)

        # Test that ML methods return default values when disabled
        is_anomaly, anomaly_score = regulator._detect_ml_anomalies(features)
        assert is_anomaly is False
        assert anomaly_score == 0.0

        is_collusion, collusion_prob = regulator._classify_collusion(features)
        assert is_collusion is False
        assert collusion_prob == 0.0

    def test_monitor_step_with_ml(self) -> None:
        """Test monitoring step with ML detection enabled."""
        regulator = MLRegulator(seed=42)

        # Create some price history first
        for i in range(5):
            prices = np.array([50.0 + i, 55.0 + i, 60.0 + i])
            regulator.monitor_step(prices, i)

        # Test monitoring with ML features
        prices = np.array([55.0, 60.0, 65.0])
        results = regulator.monitor_step(prices, 5)

        # Check that ML results are included
        assert "ml_anomaly_detected" in results
        assert "ml_anomaly_score" in results
        assert "ml_collusion_detected" in results
        assert "ml_collusion_probability" in results
        assert "ml_features" in results

        # Check that features are extracted
        assert len(results["ml_features"]) == 20
        assert isinstance(results["ml_features"], list)

    def test_training_data_update(self) -> None:
        """Online training data is unlabeled feature windows."""
        regulator = MLRegulator(seed=42)
        assert len(regulator.training_features) == 0

        for _ in range(11):
            regulator._update_training_data(np.random.rand(20).astype(np.float32))

        assert len(regulator.training_features) == 11
        assert not hasattr(regulator, "training_labels")

    def test_retraining_fits_only_the_anomaly_detector(self) -> None:
        regulator = MLRegulator(seed=42)
        for _ in range(25):
            regulator._update_training_data(np.random.rand(20).astype(np.float32))

        regulator._retrain_models()

        assert hasattr(regulator.anomaly_detector[-1], "estimators_")
        assert regulator.collusion_classifier is None
        is_anomaly, _ = regulator._detect_ml_anomalies(np.random.rand(20))
        assert isinstance(is_anomaly, (bool, np.bool_))

    def test_insufficient_training_data(self) -> None:
        """Retraining with too little data leaves the detector unfitted."""
        regulator = MLRegulator(seed=42)
        for _ in range(5):
            regulator._update_training_data(np.random.rand(20).astype(np.float32))

        regulator._retrain_models()

        assert not hasattr(regulator.anomaly_detector[-1], "estimators_")
        assert regulator._detect_ml_anomalies(np.random.rand(20)) == (False, 0.0)

    def test_rule_violations_do_not_train_the_classifier(self) -> None:
        """Rule-based flags must not become classifier labels (circularity)."""
        regulator = MLRegulator(seed=42, retrain_frequency=5, parallel_steps=2)
        for step in range(30):
            regulator.monitor_step(np.array([50.0, 50.0]), step)

        assert regulator.collusion_classifier is None

    def test_fit_classifier_enables_collusion_detection(self) -> None:
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (50, 20)), rng.normal(3, 1, (50, 20))])
        y = np.array([0] * 50 + [1] * 50)
        regulator = MLRegulator(seed=42, ml_collusion_threshold=0.5)

        regulator.fit_classifier(X, y)

        assert regulator._classify_collusion(np.full(20, 3.0))[0]
        assert not regulator._classify_collusion(np.zeros(20))[0]

    def test_supplied_classifier_survives_reset(self) -> None:
        rng = np.random.default_rng(0)
        regulator = MLRegulator(seed=42)
        regulator.fit_classifier(rng.normal(size=(40, 20)), np.arange(40) % 2)
        classifier = regulator.collusion_classifier

        regulator.reset(n_firms=2)

        assert regulator.collusion_classifier is classifier

    def test_apply_penalties_with_ml(self) -> None:
        """Test penalty application including ML-based fines."""
        regulator = MLRegulator(seed=42)

        # Create detection results with ML fines
        detection_results = {
            "parallel_violation": False,
            "structural_break_violation": False,
            "fines_applied": [0.0, 0.0, 0.0],
            "ml_fines_applied": [10.0, 15.0, 20.0],  # ML-detected fines
        }

        rewards = np.array([100.0, 120.0, 110.0])
        modified_rewards = regulator.apply_penalties(rewards, detection_results)

        # Check that ML fines are applied
        expected_rewards = rewards - np.array([10.0, 15.0, 20.0])
        assert np.allclose(modified_rewards, expected_rewards)

    def test_get_ml_statistics(self) -> None:
        """Test ML statistics retrieval."""
        regulator = MLRegulator(seed=42)

        # Add some training data
        for _ in range(10):
            regulator._update_training_data(np.random.rand(20).astype(np.float32))

        stats = regulator.get_ml_statistics()

        # Check that statistics are returned
        assert "ml_enabled" in stats
        assert "training_samples" in stats
        assert "anomaly_threshold" in stats
        assert "collusion_threshold" in stats
        assert "feature_window_size" in stats
        assert "retrain_frequency" in stats

        assert stats["ml_enabled"] is True
        assert stats["training_samples"] == 10
        assert stats["classifier_fitted"] is False

    def test_get_ml_statistics_disabled(self) -> None:
        """Test ML statistics when ML is disabled."""
        regulator = MLRegulator(use_ml_detection=False, seed=42)

        stats = regulator.get_ml_statistics()

        assert stats["ml_enabled"] is False

    def test_reset_functionality(self) -> None:
        """Test ML regulator reset functionality."""
        regulator = MLRegulator(seed=42)

        # Add some state
        for i in range(5):
            prices = np.array([50.0 + i, 55.0 + i, 60.0 + i])
            regulator.monitor_step(prices, i)

        regulator._update_training_data(np.random.rand(20).astype(np.float32))

        regulator.reset(n_firms=3)

        assert len(regulator.training_features) == 0
        assert regulator.step_count == 0
        assert regulator.anomaly_detector is not None

    def test_error_handling_in_ml_methods(self) -> None:
        """Test error handling in ML methods."""
        regulator = MLRegulator(seed=42)

        # Test with invalid features
        invalid_features = np.array([np.nan, np.inf, -np.inf])

        # Should not crash, should return default values
        is_anomaly, anomaly_score = regulator._detect_ml_anomalies(invalid_features)
        assert is_anomaly is False
        assert anomaly_score == 0.0

        is_collusion, collusion_prob = regulator._classify_collusion(invalid_features)
        assert is_collusion is False
        assert collusion_prob == 0.0

    def test_retrain_frequency_trigger(self) -> None:
        """Test that models are retrained at specified frequency."""
        regulator = MLRegulator(retrain_frequency=3, seed=42)

        # Mock the retrain method to track calls
        with patch.object(regulator, "_retrain_models") as mock_retrain:
            # Run steps up to retrain frequency
            for i in range(5):
                prices = np.array([50.0 + i, 55.0 + i, 60.0 + i])
                regulator.monitor_step(prices, i)

            # Should have been called at step 3 (step_count = 3)
            assert mock_retrain.call_count >= 1
