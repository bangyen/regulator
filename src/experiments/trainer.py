"""
Training functions for the Regulator package.

This module contains functions for training ML models and running episodes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.detectors.ml_detector import (
    CollusionDetector,
    FeatureExtractor,
    generate_synthetic_labels,
)


def create_demo_episodes(output_dir: Path, n_episodes: int = 50) -> List[Path]:
    """
    Create demo episode log files for training.

    Args:
        output_dir: Directory to save demo episodes
        n_episodes: Number of episodes to create

    Returns:
        List of paths to created episode files
    """
    # This is a simplified version - in practice, you'd want to generate
    # actual episode data. For now, we'll create empty files as placeholders.
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_files = []
    for i in range(n_episodes):
        episode_file = output_dir / f"demo_episode_{i:03d}.jsonl"
        # Create a minimal episode file
        with open(episode_file, "w") as f:
            f.write(
                '{"step": 0, "prices": [50.0], "profits": [1000.0], "market_price": 50.0}\n'
            )
        episode_files.append(episode_file)

    return episode_files


def train_and_evaluate_detector(
    log_files: List[Path],
    model_type: str = "logistic",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[CollusionDetector, Dict[str, Any]]:
    """
    Train and evaluate a collusion detector.

    Args:
        log_files: List of log file paths
        model_type: Type of model to use ("logistic" or "lightgbm")
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducibility

    Returns:
        Tuple of (trained_detector, evaluation_metrics)
    """
    print(f"Training {model_type} collusion detector...")

    # Generate synthetic labels
    print("Generating synthetic labels...")
    log_files_with_labels, labels = generate_synthetic_labels(
        [str(f) for f in log_files]
    )

    print(f"Generated labels for {len(log_files)} episodes")
    print(f"Collusive episodes: {sum(labels)}")
    print(f"Competitive episodes: {len(labels) - sum(labels)}")

    # Extract features
    print("Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_features_batch([str(f) for f in log_files])
    print(f"Extracted {features.shape[1]} features from {len(log_files)} episodes")

    # Train detector
    print("Training detector...")
    detector = CollusionDetector(model_type=model_type, random_state=random_state)
    metrics = detector.train(features, labels, test_size=test_size)

    return detector, metrics


def run_episode(
    firms: List[str],
    steps: int = 50,
    n_firms: Optional[int] = None,
    seed: int = 42,
    log_dir: str = "logs",
) -> Dict[str, Any]:
    """
    Run a single episode with the specified parameters.

    Args:
        firms: List of agent types
        steps: Number of steps to run
        n_firms: Number of firms (auto-detected if not specified)
        seed: Random seed for reproducibility
        log_dir: Directory to save log files

    Returns:
        Dictionary containing episode results
    """
    # This is a simplified version - in practice, you'd want to use
    # the actual episode running logic from the scripts
    print(f"Running episode with {len(firms)} firms: {', '.join(firms)}")
    print(f"Steps: {steps}, Seed: {seed}")

    # Create a simple result
    result = {
        "episode_id": f"episode_{seed}",
        "firms": firms,
        "steps": steps,
        "seed": seed,
        "log_file": f"{log_dir}/episode_{seed}.jsonl",
    }

    return result


def train_ml_detector(
    n_episodes: int = 50,
    model_type: str = "logistic",
    existing_logs: Optional[str] = None,
    output_dir: str = "ml_detector_output",
) -> None:
    """
    Train the ML collusion detector.

    Args:
        n_episodes: Number of episodes to train on
        model_type: Model type to use
        existing_logs: Path to existing log files
        output_dir: Output directory for results
    """
    print("ML-Based Collusion Detector Training")
    print("=" * 50)

    output_path = Path(output_dir)

    if existing_logs:
        # Use existing log files
        log_files = list(Path(existing_logs).glob("*.jsonl"))
        print(f"Using {len(log_files)} existing log files from {existing_logs}")
    else:
        # Create demo episodes
        print(f"Creating {n_episodes} demo episodes...")
        log_files = create_demo_episodes(output_path, n_episodes)
        print(f"Created {n_episodes} demo episodes in {output_dir}")

    # Train and evaluate detector
    detector, metrics = train_and_evaluate_detector(
        log_files,
        model_type=model_type,
        test_size=0.2,
        random_state=42,
    )

    # Print results
    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    print(f"Model Type: {model_type}")
    print(f"Training Accuracy: {metrics['train_accuracy']:.3f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"AUROC: {metrics['auroc']:.3f}")
    print(f"Training Samples: {metrics['train_samples']}")
    print(f"Test Samples: {metrics['test_samples']}")

    if metrics["auroc"] >= 0.8:
        print("✅ AUROC threshold (≥0.8) achieved!")
    else:
        print("⚠️  AUROC below threshold (≥0.8)")

    # Save results
    output_path.mkdir(parents=True, exist_ok=True)

    # Save training metrics
    import json

    metrics_file = output_path / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance
    importance = detector.get_feature_importance()
    importance_file = output_path / "feature_importance.json"
    importance_data = {
        "feature_importance": importance.tolist() if importance is not None else [],
        "model_type": detector.model_type,
    }
    with open(importance_file, "w") as f:
        json.dump(importance_data, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"  - Training metrics: {output_dir}/training_metrics.json")
    print(f"  - Feature importance: {output_dir}/feature_importance.json")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Final AUROC: {metrics['auroc']:.3f}")

    if metrics["auroc"] >= 0.8:
        print("✅ Model meets AUROC threshold requirement!")
    else:
        print("⚠️  Model below AUROC threshold - consider:")
        print("   - Increasing training data")
        print("   - Trying different model types")
        print("   - Adjusting feature extraction")
