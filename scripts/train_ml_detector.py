#!/usr/bin/env python3
"""
Training script for ML-based collusion detector.

This script demonstrates how to train and evaluate the ML collusion detector
using synthetic data generated from episode logs.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np

# Import from the package
from detectors.ml_detector import (
    CollusionDetector,
    FeatureExtractor,
    generate_synthetic_labels,
)


def create_demo_episodes(output_dir: Path, n_episodes: int = 50) -> List[Path]:
    """
    Create demo episode log files for training.

    Args:
        output_dir: Directory to save log files
        n_episodes: Number of episodes to create

    Returns:
        List of created log file paths
    """
    output_dir.mkdir(exist_ok=True)
    log_files = []

    print(f"Creating {n_episodes} demo episodes...")

    for i in range(n_episodes):
        log_file = output_dir / f"demo_episode_{i:03d}.jsonl"

        # Create episode header
        episode_header = {
            "type": "episode_header",
            "episode_id": f"demo_episode_{i:03d}",
            "start_time": "2025-01-01T00:00:00",
            "n_firms": 2,
            "log_file": str(log_file),
        }

        # Create step data with different patterns
        steps = cast(List[Dict[str, Any]], [])
        np.random.seed(42 + i)
        n_steps = 30

        # Determine if this episode should be collusive
        is_collusive = i % 2 == 0  # Alternate between collusive and competitive

        for step in range(1, n_steps + 1):
            if is_collusive:
                # Collusive pattern: coordinated prices, low volatility
                if step < 10:
                    # Initial competitive phase
                    prices = [
                        50 + np.random.normal(0, 10),
                        55 + np.random.normal(0, 10),
                    ]
                else:
                    # Collusive phase: coordinated high prices
                    base_price = 75 + np.random.normal(0, 2)
                    prices = [base_price, base_price + np.random.normal(0, 1)]
            else:
                # Competitive pattern: volatile prices, price wars
                if step < 15:
                    # Price war phase
                    prices = [
                        40 + np.random.normal(0, 15),
                        45 + np.random.normal(0, 15),
                    ]
                else:
                    # Recovery phase
                    prices = [60 + np.random.normal(0, 8), 65 + np.random.normal(0, 8)]

            # Ensure prices are within bounds
            prices = [max(1.0, min(100.0, p)) for p in prices]

            # Calculate profits
            marginal_cost = 10.0
            profits = [(p - marginal_cost) * 10 for p in prices]
            profits = [max(0, p) for p in profits]

            # Demand shock
            demand_shock = np.random.normal(0, 5)

            # Market calculations
            market_price = np.mean(prices)
            total_demand = max(0.0, 100.0 - market_price + demand_shock)
            individual_quantity = max(0.0, float(str(total_demand)) / 2.0)

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
                    "agent_types": [
                        "collusive" if is_collusive else "competitive",
                        "titfortat",
                    ],
                    "agent_prices": prices,
                },
            }
            steps.append(step_data)

        # Create episode end
        episode_end = {
            "type": "episode_end",
            "episode_id": f"demo_episode_{i:03d}",
            "end_time": "2025-01-01T00:00:30",
            "duration_seconds": 30.0,
            "total_steps": n_steps,
            "terminated": False,
            "truncated": True,
            "final_rewards": profits,
            "episode_summary": {
                "total_steps": n_steps,
                "final_market_price": market_price,
                "total_profits": profits,
                "agent_types": [
                    "collusive" if is_collusive else "competitive",
                    "titfortat",
                ],
                "environment_params": {
                    "n_firms": 2,
                    "max_steps": 100,
                    "marginal_cost": marginal_cost,
                    "demand_intercept": 100.0,
                    "demand_slope": -1.0,
                    "shock_std": 5.0,
                    "price_min": 1.0,
                    "price_max": 100.0,
                    "seed": 42 + i,
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

    print(f"Created {len(log_files)} demo episodes in {output_dir}")
    return log_files


def train_and_evaluate_detector(
    log_files: List[Path],
    model_type: str = "logistic",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[CollusionDetector, dict]:
    """
    Train and evaluate the collusion detector.

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
    valid_files, labels = generate_synthetic_labels(
        list(log_files), collusion_ratio=0.5, random_state=random_state
    )

    print(f"Generated labels for {len(valid_files)} episodes")
    print(f"Collusive episodes: {np.sum(labels)}")
    print(f"Competitive episodes: {len(labels) - np.sum(labels)}")

    # Extract features
    print("Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_features_batch(valid_files)

    print(f"Extracted {features.shape[1]} features from {features.shape[0]} episodes")

    # Train detector
    print("Training detector...")
    detector = CollusionDetector(model_type=model_type, random_state=random_state)
    metrics = detector.train(features, labels, test_size=test_size)

    # Print results
    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    print(f"Model Type: {model_type}")
    print(f"Training Accuracy: {metrics['train_accuracy']:.3f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"AUROC: {metrics['auroc']:.3f}")
    print(f"Training Samples: {metrics['n_train_samples']}")
    print(f"Test Samples: {metrics['n_test_samples']}")

    # Check AUROC threshold
    if metrics["auroc"] >= 0.8:
        print("✅ AUROC threshold (≥0.8) achieved!")
    else:
        print("⚠️  AUROC below threshold (≥0.8)")

    # Feature importance (if available)
    importance = detector.get_feature_importance()
    if importance is not None:
        print("\nTop 5 Most Important Features:")
        top_indices = np.argsort(importance)[-5:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Feature {idx}: {importance[idx]:.3f}")

    return detector, metrics


def demonstrate_predictions(detector: CollusionDetector, log_files: List[Path]) -> None:
    """
    Demonstrate predictions on sample episodes.

    Args:
        detector: Trained collusion detector
        log_files: List of log file paths
    """
    print("\n" + "=" * 50)
    print("PREDICTION DEMONSTRATION")
    print("=" * 50)

    # Select a few episodes for demonstration
    sample_files = log_files[:5]

    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_features_batch(list(sample_files))

    # Make predictions
    predictions = detector.predict(features)
    probabilities = detector.predict_proba(features)

    print("Episode Predictions:")
    for i, (log_file, pred, prob) in enumerate(
        zip(sample_files, predictions, probabilities)
    ):
        collusion_prob = prob[1]  # Probability of collusion
        status = "COLLUSIVE" if pred == 1 else "COMPETITIVE"
        print(f"  {log_file.name}: {status} (confidence: {collusion_prob:.3f})")


def save_model_and_results(
    detector: CollusionDetector, metrics: dict, output_dir: Path
) -> None:
    """
    Save the trained model and results.

    Args:
        detector: Trained collusion detector
        metrics: Evaluation metrics
        output_dir: Directory to save outputs
    """
    output_dir.mkdir(exist_ok=True)

    # Save metrics
    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance
    importance = detector.get_feature_importance()
    if importance is not None:
        importance_file = output_dir / "feature_importance.json"
        importance_data = {
            "feature_importance": importance.tolist(),
            "model_type": detector.model_type,
        }
        with open(importance_file, "w") as f:
            json.dump(importance_data, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"  - Training metrics: {metrics_file}")
    if importance is not None:
        print(f"  - Feature importance: {importance_file}")


def main() -> None:
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train ML-based collusion detector")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml_detector_output"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=50, help="Number of demo episodes to create"
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "lightgbm"],
        default="logistic",
        help="Type of ML model to use",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--existing-logs",
        type=Path,
        help="Path to existing log files (if not provided, demo episodes will be created)",
    )

    args = parser.parse_args()

    print("ML-Based Collusion Detector Training")
    print("=" * 50)

    # Create or use existing log files
    if args.existing_logs:
        if args.existing_logs.is_dir():
            log_files = list(args.existing_logs.glob("*.jsonl"))
        else:
            log_files = [args.existing_logs]
        print(f"Using {len(log_files)} existing log files")
    else:
        log_files = create_demo_episodes(args.output_dir, args.n_episodes)

    if not log_files:
        print("Error: No log files found!")
        sys.exit(1)

    # Train and evaluate detector
    detector, metrics = train_and_evaluate_detector(
        log_files,
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Demonstrate predictions
    demonstrate_predictions(detector, log_files)

    # Save results
    save_model_and_results(detector, metrics, args.output_dir)

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


if __name__ == "__main__":
    main()
