"""
Tests for the trainer module.

This module tests the training functions for ML models and episode running.
"""

import pytest
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.experiments.trainer import (
    create_demo_episodes,
    train_and_evaluate_detector,
    run_episode,
    train_ml_detector,
)


class TestCreateDemoEpisodes:
    """Test the create_demo_episodes function."""

    def test_create_demo_episodes_basic(self):
        """Test basic demo episode creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            n_episodes = 5

            episode_files = create_demo_episodes(output_dir, n_episodes)

            assert len(episode_files) == n_episodes
            assert all(isinstance(f, Path) for f in episode_files)
            assert all(f.exists() for f in episode_files)

            # Check file names
            expected_names = [f"demo_episode_{i:03d}.jsonl" for i in range(n_episodes)]
            actual_names = [f.name for f in episode_files]
            assert actual_names == expected_names

    def test_create_demo_episodes_file_content(self):
        """Test that demo episode files contain expected content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            n_episodes = 3

            episode_files = create_demo_episodes(output_dir, n_episodes)

            # Check content of first file
            with open(episode_files[0], "r") as f:
                content = f.read().strip()
                data = json.loads(content)

                assert data["step"] == 0
                assert data["prices"] == [50.0]
                assert data["profits"] == [1000.0]
                assert data["market_price"] == 50.0

    def test_create_demo_episodes_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nested" / "directory"
            n_episodes = 2

            # Directory shouldn't exist initially
            assert not output_dir.exists()

            episode_files = create_demo_episodes(output_dir, n_episodes)

            # Directory should be created
            assert output_dir.exists()
            assert len(episode_files) == n_episodes

    def test_create_demo_episodes_zero_episodes(self):
        """Test creating zero episodes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            n_episodes = 0

            episode_files = create_demo_episodes(output_dir, n_episodes)

            assert len(episode_files) == 0

    def test_create_demo_episodes_large_number(self):
        """Test creating a large number of episodes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            n_episodes = 100

            episode_files = create_demo_episodes(output_dir, n_episodes)

            assert len(episode_files) == n_episodes
            assert all(f.exists() for f in episode_files)


class TestTrainAndEvaluateDetector:
    """Test the train_and_evaluate_detector function."""

    @pytest.fixture
    def mock_log_files(self):
        """Create mock log files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            log_files = []

            for i in range(5):
                log_file = log_dir / f"episode_{i}.jsonl"
                with open(log_file, "w") as f:
                    f.write('{"step": 0, "prices": [50.0], "profits": [1000.0]}\n')
                log_files.append(log_file)

            yield log_files

    @patch("src.experiments.trainer.generate_synthetic_labels")
    @patch("src.experiments.trainer.FeatureExtractor")
    @patch("src.experiments.trainer.CollusionDetector")
    def test_train_and_evaluate_detector_basic(
        self,
        mock_detector_class,
        mock_extractor_class,
        mock_generate_labels,
        mock_log_files,
    ):
        """Test basic detector training and evaluation."""
        # Setup mocks
        mock_labels = [0, 1, 0, 1, 0]
        mock_generate_labels.return_value = (mock_log_files, mock_labels)

        mock_extractor = Mock()
        mock_extractor.extract_features_batch.return_value = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
            ]
        )
        mock_extractor_class.return_value = mock_extractor

        mock_detector = Mock()
        mock_detector.train.return_value = {
            "train_accuracy": 0.85,
            "test_accuracy": 0.80,
            "auroc": 0.82,
            "n_train_samples": 4,
            "n_test_samples": 1,
        }
        mock_detector_class.return_value = mock_detector

        # Test function
        detector, metrics = train_and_evaluate_detector(mock_log_files)

        # Verify calls
        mock_generate_labels.assert_called_once_with([str(f) for f in mock_log_files])
        mock_extractor.extract_features_batch.assert_called_once_with(
            [str(f) for f in mock_log_files]
        )
        mock_detector.train.assert_called_once()

        # Verify return values
        assert detector == mock_detector
        assert metrics["train_accuracy"] == 0.85
        assert metrics["test_accuracy"] == 0.80
        assert metrics["auroc"] == 0.82

    @patch("src.experiments.trainer.generate_synthetic_labels")
    @patch("src.experiments.trainer.FeatureExtractor")
    @patch("src.experiments.trainer.CollusionDetector")
    def test_train_and_evaluate_detector_custom_params(
        self,
        mock_detector_class,
        mock_extractor_class,
        mock_generate_labels,
        mock_log_files,
    ):
        """Test detector training with custom parameters."""
        # Setup mocks
        mock_labels = [0, 1, 0, 1, 0]
        mock_generate_labels.return_value = (mock_log_files, mock_labels)

        mock_extractor = Mock()
        mock_extractor.extract_features_batch.return_value = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
            ]
        )
        mock_extractor_class.return_value = mock_extractor

        mock_detector = Mock()
        mock_detector.train.return_value = {
            "train_accuracy": 0.90,
            "test_accuracy": 0.85,
            "auroc": 0.88,
            "n_train_samples": 4,
            "n_test_samples": 1,
        }
        mock_detector_class.return_value = mock_detector

        # Test with custom parameters
        detector, metrics = train_and_evaluate_detector(
            mock_log_files, model_type="lightgbm", test_size=0.3, random_state=123
        )

        # Verify detector was created with custom parameters
        mock_detector_class.assert_called_once_with(
            model_type="lightgbm", random_state=123
        )

        # Verify train was called with custom test_size
        mock_detector.train.assert_called_once()
        call_args = mock_detector.train.call_args
        assert call_args[1]["test_size"] == 0.3

    @patch("src.experiments.trainer.generate_synthetic_labels")
    def test_train_and_evaluate_detector_empty_log_files(self, mock_generate_labels):
        """Test detector training with empty log files."""
        mock_generate_labels.return_value = ([], [])

        with pytest.raises(ValueError):
            train_and_evaluate_detector([])


class TestRunEpisode:
    """Test the run_episode function."""

    @patch("src.experiments.trainer.print")
    def test_run_episode_basic(self, mock_print):
        """Test basic episode running."""
        firms = ["firm1", "firm2", "firm3"]
        steps = 50
        seed = 42
        log_dir = "test_logs"

        result = run_episode(firms, steps, seed=seed, log_dir=log_dir)

        # Verify print calls
        assert mock_print.call_count >= 2

        # Verify result structure
        assert result["episode_id"] == f"episode_{seed}"
        assert result["firms"] == firms
        assert result["steps"] == steps
        assert result["seed"] == seed
        assert result["log_file"] == f"{log_dir}/episode_{seed}.jsonl"

    @patch("src.experiments.trainer.print")
    def test_run_episode_custom_params(self, mock_print):
        """Test episode running with custom parameters."""
        firms = ["firm1", "firm2"]
        steps = 100
        n_firms = 2
        seed = 123
        log_dir = "custom_logs"

        result = run_episode(firms, steps, n_firms=n_firms, seed=seed, log_dir=log_dir)

        assert result["episode_id"] == f"episode_{seed}"
        assert result["firms"] == firms
        assert result["steps"] == steps
        assert result["seed"] == seed
        assert result["log_file"] == f"{log_dir}/episode_{seed}.jsonl"

    @patch("src.experiments.trainer.print")
    def test_run_episode_empty_firms(self, mock_print):
        """Test episode running with empty firms list."""
        firms = []
        steps = 10

        result = run_episode(firms, steps)

        assert result["firms"] == []
        assert result["steps"] == steps


class TestTrainMlDetector:
    """Test the train_ml_detector function."""

    @patch("src.experiments.trainer.create_demo_episodes")
    @patch("src.experiments.trainer.train_and_evaluate_detector")
    @patch("src.experiments.trainer.print")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_train_ml_detector_with_demo_episodes(
        self, mock_json_dump, mock_file, mock_print, mock_train_eval, mock_create_demo
    ):
        """Test ML detector training with demo episodes."""
        # Setup mocks
        mock_log_files = [
            Path("demo_episode_000.jsonl"),
            Path("demo_episode_001.jsonl"),
        ]
        mock_create_demo.return_value = mock_log_files

        mock_detector = Mock()
        mock_detector.get_feature_importance.return_value = np.array([0.1, 0.2, 0.3])
        mock_metrics = {
            "train_accuracy": 0.85,
            "test_accuracy": 0.80,
            "auroc": 0.82,
            "n_train_samples": 40,
            "n_test_samples": 10,
        }
        mock_train_eval.return_value = (mock_detector, mock_metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "ml_detector_output"

            train_ml_detector(
                n_episodes=50,
                model_type="logistic",
                existing_logs=None,
                output_dir=str(output_dir),
            )

            # Verify demo episodes were created
            mock_create_demo.assert_called_once_with(output_dir, 50)

            # Verify detector was trained
            mock_train_eval.assert_called_once()

            # Verify results were saved
            assert mock_json_dump.call_count == 2  # metrics and feature importance

            # Verify print calls
            assert mock_print.call_count > 0

    @patch("src.experiments.trainer.train_and_evaluate_detector")
    @patch("src.experiments.trainer.print")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_train_ml_detector_with_existing_logs(
        self, mock_json_dump, mock_file, mock_print, mock_train_eval
    ):
        """Test ML detector training with existing log files."""
        # Setup mocks
        mock_detector = Mock()
        mock_detector.get_feature_importance.return_value = np.array([0.1, 0.2, 0.3])
        mock_metrics = {
            "train_accuracy": 0.90,
            "test_accuracy": 0.85,
            "auroc": 0.88,
            "n_train_samples": 80,
            "n_test_samples": 20,
        }
        mock_train_eval.return_value = (mock_detector, mock_metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some mock log files
            log_dir = Path(temp_dir) / "existing_logs"
            log_dir.mkdir()
            (log_dir / "episode1.jsonl").touch()
            (log_dir / "episode2.jsonl").touch()

            train_ml_detector(
                n_episodes=50,
                model_type="lightgbm",
                existing_logs=str(log_dir),
                output_dir=str(temp_dir),
            )

            # Verify detector was trained
            mock_train_eval.assert_called_once()

            # Verify results were saved
            assert mock_json_dump.call_count == 2

    @patch("src.experiments.trainer.create_demo_episodes")
    @patch("src.experiments.trainer.train_and_evaluate_detector")
    @patch("src.experiments.trainer.print")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_train_ml_detector_auroc_threshold_met(
        self, mock_json_dump, mock_file, mock_print, mock_train_eval, mock_create_demo
    ):
        """Test ML detector training when AUROC threshold is met."""
        # Setup mocks
        mock_log_files = [Path("demo_episode_000.jsonl")]
        mock_create_demo.return_value = mock_log_files

        mock_detector = Mock()
        mock_detector.get_feature_importance.return_value = np.array([0.1, 0.2, 0.3])
        mock_metrics = {
            "train_accuracy": 0.90,
            "test_accuracy": 0.85,
            "auroc": 0.85,  # Above threshold
            "n_train_samples": 40,
            "n_test_samples": 10,
        }
        mock_train_eval.return_value = (mock_detector, mock_metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            train_ml_detector(
                n_episodes=50,
                model_type="logistic",
                existing_logs=None,
                output_dir=str(temp_dir),
            )

            # Verify success message was printed
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any("✅ AUROC threshold" in call for call in print_calls)

    @patch("src.experiments.trainer.create_demo_episodes")
    @patch("src.experiments.trainer.train_and_evaluate_detector")
    @patch("src.experiments.trainer.print")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_train_ml_detector_auroc_threshold_not_met(
        self, mock_json_dump, mock_file, mock_print, mock_train_eval, mock_create_demo
    ):
        """Test ML detector training when AUROC threshold is not met."""
        # Setup mocks
        mock_log_files = [Path("demo_episode_000.jsonl")]
        mock_create_demo.return_value = mock_log_files

        mock_detector = Mock()
        mock_detector.get_feature_importance.return_value = np.array([0.1, 0.2, 0.3])
        mock_metrics = {
            "train_accuracy": 0.70,
            "test_accuracy": 0.65,
            "auroc": 0.65,  # Below threshold
            "n_train_samples": 40,
            "n_test_samples": 10,
        }
        mock_train_eval.return_value = (mock_detector, mock_metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            train_ml_detector(
                n_episodes=50,
                model_type="logistic",
                existing_logs=None,
                output_dir=str(temp_dir),
            )

            # Verify warning message was printed
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any("⚠️  AUROC below threshold" in call for call in print_calls)

    @patch("src.experiments.trainer.create_demo_episodes")
    @patch("src.experiments.trainer.train_and_evaluate_detector")
    @patch("src.experiments.trainer.print")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_train_ml_detector_feature_importance_none(
        self, mock_json_dump, mock_file, mock_print, mock_train_eval, mock_create_demo
    ):
        """Test ML detector training when feature importance is None."""
        # Setup mocks
        mock_log_files = [Path("demo_episode_000.jsonl")]
        mock_create_demo.return_value = mock_log_files

        mock_detector = Mock()
        mock_detector.get_feature_importance.return_value = None
        mock_metrics = {
            "train_accuracy": 0.85,
            "test_accuracy": 0.80,
            "auroc": 0.82,
            "n_train_samples": 40,
            "n_test_samples": 10,
        }
        mock_train_eval.return_value = (mock_detector, mock_metrics)

        with tempfile.TemporaryDirectory() as temp_dir:
            train_ml_detector(
                n_episodes=50,
                model_type="logistic",
                existing_logs=None,
                output_dir=str(temp_dir),
            )

            # Verify feature importance was saved as empty list
            json_calls = mock_json_dump.call_args_list
            feature_importance_call = next(
                call for call in json_calls if "feature_importance" in str(call)
            )
            assert feature_importance_call[0][0]["feature_importance"] == []

    def test_train_ml_detector_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nested" / "output" / "directory"

            # Directory shouldn't exist initially
            assert not output_dir.exists()

            with patch(
                "src.experiments.trainer.create_demo_episodes"
            ) as mock_create_demo, patch(
                "src.experiments.trainer.train_and_evaluate_detector"
            ) as mock_train_eval, patch(
                "src.experiments.trainer.print"
            ), patch(
                "builtins.open", new_callable=mock_open
            ), patch(
                "json.dump"
            ):

                mock_create_demo.return_value = []
                mock_train_eval.return_value = (
                    Mock(),
                    {
                        "train_accuracy": 0.85,
                        "test_accuracy": 0.80,
                        "auroc": 0.82,
                        "n_train_samples": 8,
                        "n_test_samples": 2,
                    },
                )

                train_ml_detector(n_episodes=10, output_dir=str(output_dir))

                # Directory should be created
                assert output_dir.exists()
