"""
Tests for the ML-enhanced regulator.

This module tests the MLRegulator class including ML-based detection,
feature extraction, and model training functionality.
"""

import numpy as np
from unittest.mock import patch

from src.agents.ml_regulator import MLRegulator


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
        assert regulator.collusion_classifier is not None

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
        """Test training data update functionality."""
        regulator = MLRegulator(seed=42)

        # Initially no training data
        assert len(regulator.training_features) == 0
        assert len(regulator.training_labels) == 0

        # Add some training data
        features = np.random.rand(20).astype(np.float32)
        regulator._update_training_data(features, True)

        assert len(regulator.training_features) == 1
        assert len(regulator.training_labels) == 1
        assert regulator.training_labels[0] == 1

        # Add more training data
        for i in range(10):
            features = np.random.rand(20).astype(np.float32)
            is_collusion = i % 2 == 0  # Alternate between collusion and normal
            regulator._update_training_data(features, is_collusion)

        assert len(regulator.training_features) == 11
        assert len(regulator.training_labels) == 11

    def test_model_retraining(self) -> None:
        """Test ML model retraining functionality."""
        regulator = MLRegulator(seed=42)

        # Add sufficient training data
        for i in range(25):
            features = np.random.rand(20).astype(np.float32)
            is_collusion = i % 3 == 0  # Some collusion cases
            regulator._update_training_data(features, is_collusion)

        # Test retraining
        regulator._retrain_models()

        # Models should still be available
        assert regulator.anomaly_detector is not None
        assert regulator.collusion_classifier is not None

    def test_insufficient_training_data(self) -> None:
        """Test retraining with insufficient training data."""
        regulator = MLRegulator(seed=42)

        # Add insufficient training data
        for i in range(5):
            features = np.random.rand(20).astype(np.float32)
            regulator._update_training_data(features, i % 2 == 0)

        # Retraining should not fail with insufficient data
        regulator._retrain_models()

        # Models should still be available
        assert regulator.anomaly_detector is not None
        assert regulator.collusion_classifier is not None

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
        for i in range(10):
            features = np.random.rand(20).astype(np.float32)
            regulator._update_training_data(features, i % 2 == 0)

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
        assert "collusion_rate" in stats
        assert "normal_rate" in stats

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

        # Add training data
        features = np.random.rand(20).astype(np.float32)
        regulator._update_training_data(features, True)

        # Reset
        regulator.reset(n_firms=3)

        # Check that state is reset
        assert len(regulator.training_features) == 0
        assert len(regulator.training_labels) == 0
        assert regulator.step_count == 0

        # Models should be reinitialized
        assert regulator.anomaly_detector is not None
        assert regulator.collusion_classifier is not None

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
