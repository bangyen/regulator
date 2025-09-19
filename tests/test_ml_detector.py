"""
Tests for ML-based collusion detector.

This module contains comprehensive tests for the feature extractor,
collusion detector, and synthetic data generation functionality.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union, cast

import numpy as np
import pytest

from src.detectors.ml_detector import (
    CollusionDetector,
    FeatureExtractor,
    generate_synthetic_labels,
)


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class."""

    def test_feature_extractor_initialization(self) -> None:
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor(min_steps=5)
        assert extractor.min_steps == 5

        # Test invalid min_steps
        with pytest.raises(ValueError, match="Minimum steps must be at least 3"):
            FeatureExtractor(min_steps=2)

    def create_mock_log_file(self, tmp_path: Path) -> Path:
        """Create a mock log file for testing."""
        log_file = tmp_path / "test_episode.jsonl"

        # Create episode header
        episode_header = {
            "type": "episode_header",
            "episode_id": "test_episode",
            "start_time": "2025-01-01T00:00:00",
            "n_firms": 2,
            "log_file": str(log_file),
        }

        # Create step data
        steps = cast(List[Dict[str, Any]], [])
        np.random.seed(42)
        n_steps = 20

        for step in range(1, n_steps + 1):
            # Simulate some price patterns
            if step < 10:
                # Competitive phase - more volatile prices
                prices = [50 + np.random.normal(0, 10), 55 + np.random.normal(0, 10)]
            else:
                # Collusive phase - more coordinated prices
                base_price = 70 + np.random.normal(0, 2)
                prices = [base_price, base_price + np.random.normal(0, 1)]

            profits = [(p - 10) * 10 for p in prices]  # Simple profit calculation
            demand_shock = np.random.normal(0, 5)
            market_price = np.mean(prices)
            total_demand = 100 - market_price + demand_shock
            individual_quantity = max(0, total_demand / 2)

            step_data = {
                "type": "step",
                "step": step,
                "timestamp": f"2025-01-01T00:00:{step:02d}",
                "prices": prices,
                "profits": profits,
                "demand_shock": demand_shock,
                "market_price": market_price,
                "total_demand": total_demand,
                "individual_quantity": individual_quantity,
                "total_profits": profits,
                "additional_info": {
                    "agent_types": ["random", "titfortat"],
                    "agent_prices": prices,
                },
            }
            steps.append(step_data)

        # Create episode end
        episode_end = {
            "type": "episode_end",
            "episode_id": "test_episode",
            "end_time": "2025-01-01T00:00:20",
            "duration_seconds": 20.0,
            "total_steps": n_steps,
            "terminated": False,
            "truncated": True,
            "final_rewards": profits,
            "episode_summary": {
                "total_steps": n_steps,
                "final_market_price": market_price,
                "total_profits": profits,
                "agent_types": ["random", "titfortat"],
                "environment_params": {
                    "n_firms": 2,
                    "max_steps": 100,
                    "marginal_cost": 10.0,
                    "demand_intercept": 100.0,
                    "demand_slope": -1.0,
                    "shock_std": 5.0,
                    "price_min": 1.0,
                    "price_max": 100.0,
                    "seed": 42,
                },
            },
        }

        # Write to file
        with open(log_file, "w") as f:
            f.write(json.dumps(episode_header) + "\n")
            for step in steps:  # type: ignore
                f.write(json.dumps(step) + "\n")
            f.write(json.dumps(episode_end) + "\n")

        return log_file

    def test_extract_features_from_log(self, tmp_path: Path) -> None:
        """Test feature extraction from a log file."""
        extractor = FeatureExtractor(min_steps=5)
        log_file = self.create_mock_log_file(tmp_path)

        features = extractor.extract_features_from_log(log_file)

        # Check that features are extracted
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) > 0

        # Check that all features are finite
        assert np.all(np.isfinite(features))

    def test_extract_features_insufficient_steps(self, tmp_path: Path) -> None:
        """Test feature extraction with insufficient steps."""
        extractor = FeatureExtractor(min_steps=50)  # High threshold
        log_file = self.create_mock_log_file(tmp_path)

        with pytest.raises(ValueError, match="Insufficient steps"):
            extractor.extract_features_from_log(log_file)

    def test_extract_features_nonexistent_file(self) -> None:
        """Test feature extraction with non-existent file."""
        extractor = FeatureExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract_features_from_log("nonexistent_file.jsonl")

    def test_extract_features_batch(self, tmp_path: Path) -> None:
        """Test batch feature extraction."""
        extractor = FeatureExtractor()

        # Create multiple log files
        log_files: List[Union[str, Path]] = []
        for i in range(3):
            log_files.append(self.create_mock_log_file(tmp_path))

        features = extractor.extract_features_batch(log_files)

        # Check batch extraction results
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 3  # 3 episodes
        assert features.shape[1] > 0  # Features extracted
        assert np.all(np.isfinite(features))

    def test_feature_extractor_output_shape(self, tmp_path: Path) -> None:
        """Test that feature extractor outputs expected shape."""
        extractor = FeatureExtractor()
        log_file = self.create_mock_log_file(tmp_path)

        features = extractor.extract_features_from_log(log_file)

        # Expected number of features based on implementation
        # This should match the number of features in _extract_features_from_steps
        expected_min_features = 20  # Conservative estimate
        assert len(features) >= expected_min_features


class TestCollusionDetector:
    """Test cases for CollusionDetector class."""

    def test_detector_initialization(self) -> None:
        """Test CollusionDetector initialization."""
        # Test logistic regression
        detector = CollusionDetector(model_type="logistic", random_state=42)
        assert detector.model_type == "logistic"
        assert detector.random_state == 42
        assert not detector.is_trained

        # Test LightGBM (if available)
        try:
            detector_lgb = CollusionDetector(model_type="lightgbm", random_state=42)
            assert detector_lgb.model_type == "lightgbm"
        except ImportError:
            # LightGBM not available, skip this test
            pass

        # Test invalid model type
        with pytest.raises(ValueError, match="model_type must be"):
            CollusionDetector(model_type="invalid")

    def test_detector_training(self) -> None:
        """Test detector training."""
        detector = CollusionDetector(model_type="logistic", random_state=42)

        # Create synthetic training data
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        # Train detector
        metrics = detector.train(X, y, test_size=0.2)

        # Check training results
        assert detector.is_trained
        assert "train_accuracy" in metrics
        assert "test_accuracy" in metrics
        assert "auroc" in metrics
        assert 0 <= metrics["auroc"] <= 1
        assert metrics["n_train_samples"] + metrics["n_test_samples"] == n_samples

    def test_detector_auroc_threshold(self) -> None:
        """Test that detector achieves ≥0.8 AUROC on synthetic dataset."""
        detector = CollusionDetector(model_type="logistic", random_state=42)

        # Create synthetic data with clear separation
        np.random.seed(42)
        n_samples = 200
        n_features = 20

        # Create two distinct clusters
        X_competitive = np.random.randn(n_samples // 2, n_features) * 2
        X_collusive = np.random.randn(n_samples // 2, n_features) * 0.5 + 3

        X = np.vstack([X_competitive, X_collusive])
        y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

        # Train detector
        metrics = detector.train(X, y, test_size=0.3)

        # Check AUROC threshold
        assert metrics["auroc"] >= 0.8, f"AUROC {metrics['auroc']} below threshold 0.8"

    def test_detector_predictions(self) -> None:
        """Test detector predictions."""
        detector = CollusionDetector(model_type="logistic", random_state=42)

        # Create and train on synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)

        detector.train(X_train, y_train, validation_split=False)

        # Test predictions
        X_test = np.random.randn(10, 20)
        predictions = detector.predict(X_test)
        probabilities = detector.predict_proba(X_test)

        # Check prediction format
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)

        # Check probability format
        assert probabilities.shape == (10, 2)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)

    def test_detector_feature_importance(self) -> None:
        """Test feature importance extraction."""
        # Use logistic regression for feature importance test
        detector = CollusionDetector(model_type="logistic", random_state=42)

        # Create and train on synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)

        detector.train(X_train, y_train, validation_split=False)

        # Get feature importance
        importance = detector.get_feature_importance()

        # Check importance format
        assert importance is not None
        assert len(importance) == 20
        assert np.all(importance >= 0)

    def test_detector_untrained_predictions(self) -> None:
        """Test that predictions fail when model is not trained."""
        detector = CollusionDetector()
        X_test = np.random.randn(10, 20)

        with pytest.raises(ValueError, match="Model must be trained"):
            detector.predict(X_test)

        with pytest.raises(ValueError, match="Model must be trained"):
            detector.predict_proba(X_test)

    def test_detector_invalid_training_data(self) -> None:
        """Test detector with invalid training data."""
        detector = CollusionDetector()

        # Mismatched lengths
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 50)

        with pytest.raises(ValueError, match="X and y must have the same length"):
            detector.train(X, y)

        # Invalid labels
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 3, 100)  # 3 classes instead of 2

        with pytest.raises(ValueError, match="y must contain exactly 2 unique values"):
            detector.train(X, y)


class TestSyntheticLabels:
    """Test cases for synthetic label generation."""

    def test_generate_synthetic_labels(self, tmp_path: Path) -> None:
        """Test synthetic label generation."""
        # Create mock log files
        log_files: List[Union[str, Path]] = []
        for i in range(5):
            log_file = tmp_path / f"episode_{i}.jsonl"
            # Create a simple mock log file
            with open(log_file, "w") as f:
                # Episode header
                header = {
                    "type": "episode_header",
                    "episode_id": f"episode_{i}",
                    "n_firms": 2,
                }
                f.write(json.dumps(header) + "\n")

                # Steps
                for step in range(1, 21):
                    step_data = {
                        "type": "step",
                        "step": step,
                        "prices": [50 + i * 5, 55 + i * 5],
                        "profits": [100 + i * 10, 110 + i * 10],
                        "demand_shock": 0.0,
                        "market_price": 52.5 + i * 5,
                        "total_demand": 50.0,
                        "individual_quantity": 25.0,
                        "total_profits": [100 + i * 10, 110 + i * 10],
                    }
                    f.write(json.dumps(step_data) + "\n")

            log_files.append(log_file)

        # Generate labels
        valid_files, labels = generate_synthetic_labels(
            log_files, collusion_ratio=0.4, random_state=42
        )

        # Check results
        assert len(valid_files) == len(labels)
        assert all(label in [0, 1] for label in labels)
        assert len(valid_files) <= len(log_files)

    def test_generate_synthetic_labels_invalid_ratio(self) -> None:
        """Test synthetic label generation with invalid ratio."""
        with pytest.raises(ValueError, match="collusion_ratio must be between 0 and 1"):
            generate_synthetic_labels([], collusion_ratio=1.5)

    def test_generate_synthetic_labels_empty_list(self) -> None:
        """Test synthetic label generation with empty file list."""
        valid_files, labels = generate_synthetic_labels([])
        assert len(valid_files) == 0
        assert len(labels) == 0


class TestIntegration:
    """Integration tests for the complete ML detector pipeline."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        """Test the complete ML detector pipeline."""
        # Create mock log files
        log_files: List[Union[str, Path]] = []
        for i in range(10):
            log_file = tmp_path / f"episode_{i}.jsonl"

            # Create episode header
            episode_header = {
                "type": "episode_header",
                "episode_id": f"episode_{i}",
                "start_time": "2025-01-01T00:00:00",
                "n_firms": 2,
                "log_file": str(log_file),
            }

            # Create step data with different patterns
            steps = []
            np.random.seed(42 + i)

            for step in range(1, 21):
                if i < 5:  # Competitive episodes
                    prices = [
                        50 + np.random.normal(0, 15),
                        55 + np.random.normal(0, 15),
                    ]
                else:  # Collusive episodes
                    base_price = 70 + np.random.normal(0, 3)
                    prices = [base_price, base_price + np.random.normal(0, 2)]

                profits = [(p - 10) * 10 for p in prices]
                demand_shock = np.random.normal(0, 5)
                market_price = np.mean(prices)
                total_demand = 100 - market_price + demand_shock
                individual_quantity = max(0, total_demand / 2)

                step_data = {
                    "type": "step",
                    "step": step,
                    "timestamp": f"2025-01-01T00:00:{step:02d}",
                    "prices": prices,
                    "profits": profits,
                    "demand_shock": demand_shock,
                    "market_price": market_price,
                    "total_demand": total_demand,
                    "individual_quantity": individual_quantity,
                    "total_profits": profits,
                    "additional_info": {
                        "agent_types": ["random", "titfortat"],
                        "agent_prices": prices,
                    },
                }
                steps.append(step_data)

            # Create episode end
            episode_end = {
                "type": "episode_end",
                "episode_id": f"episode_{i}",
                "end_time": "2025-01-01T00:00:20",
                "duration_seconds": 20.0,
                "total_steps": 20,
                "terminated": False,
                "truncated": True,
                "final_rewards": profits,
                "episode_summary": {
                    "total_steps": 20,
                    "final_market_price": market_price,
                    "total_profits": profits,
                    "agent_types": ["random", "titfortat"],
                    "environment_params": {
                        "n_firms": 2,
                        "max_steps": 100,
                        "marginal_cost": 10.0,
                        "demand_intercept": 100.0,
                        "demand_slope": -1.0,
                        "shock_std": 5.0,
                        "price_min": 1.0,
                        "price_max": 100.0,
                        "seed": 42,
                    },
                },
            }

            # Write to file
            with open(log_file, "w") as f:
                f.write(json.dumps(episode_header) + "\n")
                for step in steps:  # type: ignore
                    f.write(json.dumps(step) + "\n")
                f.write(json.dumps(episode_end) + "\n")

            log_files.append(log_file)

        # Generate synthetic labels
        valid_files, labels = generate_synthetic_labels(
            log_files, collusion_ratio=0.5, random_state=42
        )

        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_features_batch(valid_files)

        # Train detector
        detector = CollusionDetector(model_type="logistic", random_state=42)
        metrics = detector.train(features, labels, test_size=0.3)

        # Test predictions
        predictions = detector.predict(features)
        probabilities = detector.predict_proba(features)

        # Verify pipeline results
        assert len(features) == len(labels)
        assert len(predictions) == len(features)
        assert probabilities.shape[0] == len(features)
        assert metrics["auroc"] >= 0.5  # Should achieve reasonable performance

    def test_regulator_integration(self) -> None:
        """Test integration with Regulator class."""
        # This test would verify that the ML detector can be integrated
        # with the existing Regulator class for real-time monitoring

        # For now, we'll test that the detector can be instantiated
        # and used in a way that would be compatible with regulator integration
        detector = CollusionDetector(model_type="logistic", random_state=42)

        # Create mock features that might come from regulator monitoring
        mock_features = np.random.randn(1, 20)

        # Train on some data first
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        detector.train(X_train, y_train, validation_split=False)

        # Test that we can get predictions (which would be used by regulator)
        predictions = detector.predict(mock_features)
        probabilities = detector.predict_proba(mock_features)

        # Verify output format is suitable for regulator integration
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]
        assert probabilities.shape == (1, 2)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)
