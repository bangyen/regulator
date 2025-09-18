"""
ML-based collusion detector for market competition analysis.

This module implements machine learning models to detect collusive behavior
by analyzing price patterns, profit margins, and market dynamics from episode logs.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMClassifier = None


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate correlation between two arrays safely, handling zero variance.

    Args:
        x: First array
        y: Second array

    Returns:
        Correlation coefficient or 0.0 if calculation fails
    """
    try:
        # Check for zero variance
        if np.var(x) == 0 or np.var(y) == 0:
            return 0.0

        # Suppress warnings for this calculation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr = np.corrcoef(x, y)[0, 1]

        return corr if not np.isnan(corr) else 0.0
    except (ValueError, IndexError):
        return 0.0


class FeatureExtractor:
    """
    Extracts features from episode logs for collusion detection.

    This class processes JSONL episode logs to extract statistical features
    that can distinguish between collusive and competitive behavior patterns.
    """

    def __init__(self, min_steps: int = 10) -> None:
        """
        Initialize the feature extractor.

        Args:
            min_steps: Minimum number of steps required for feature extraction
        """
        if min_steps < 3:
            raise ValueError("Minimum steps must be at least 3 for meaningful features")
        self.min_steps = min_steps

    def extract_features_from_log(self, log_file: Union[str, Path]) -> np.ndarray:
        """
        Extract features from a single episode log file.

        Args:
            log_file: Path to the JSONL log file

        Returns:
            Array of extracted features

        Raises:
            ValueError: If log file has insufficient data or invalid format
        """
        log_path = Path(log_file)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")

        # Parse log file
        steps_data = []
        episode_info = None

        with open(log_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("type") == "step":
                        steps_data.append(data)
                    elif data.get("type") == "episode_header":
                        episode_info = data
                except json.JSONDecodeError:
                    continue

        if len(steps_data) < self.min_steps:
            raise ValueError(
                f"Insufficient steps in log: {len(steps_data)} < {self.min_steps}"
            )

        return self._extract_features_from_steps(steps_data, episode_info)

    def _extract_features_from_steps(
        self, steps_data: List[Dict[str, Any]], episode_info: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Extract features from parsed step data.

        Args:
            steps_data: List of step data dictionaries
            episode_info: Episode header information

        Returns:
            Array of extracted features
        """
        # Extract time series data
        prices = np.array([step["prices"] for step in steps_data])
        profits = np.array([step["profits"] for step in steps_data])
        demand_shocks = np.array([step["demand_shock"] for step in steps_data])
        market_prices = np.array([step["market_price"] for step in steps_data])

        n_firms = prices.shape[1]
        n_steps = len(steps_data)

        features = []

        # 1. Price variance features
        price_variance = np.var(prices, axis=0)  # Variance for each firm
        features.extend(price_variance.tolist())
        features.append(np.mean(price_variance))  # Average price variance
        features.append(np.std(price_variance))  # Std of price variance across firms

        # 2. Price autocorrelation features
        for i in range(n_firms):
            if n_steps > 1:
                # Lag-1 autocorrelation
                autocorr_1 = _safe_correlation(prices[:-1, i], prices[1:, i])
                features.append(autocorr_1)
            else:
                features.append(0.0)

        # Average autocorrelation across firms
        autocorrs = []
        for i in range(n_firms):
            if n_steps > 1:
                autocorr = _safe_correlation(prices[:-1, i], prices[1:, i])
                if autocorr != 0.0:  # Only add non-zero correlations
                    autocorrs.append(autocorr)
        features.append(np.mean(autocorrs) if autocorrs else 0.0)

        # 3. Profit margin features
        # Calculate profit margins (assuming marginal cost from episode info)
        marginal_cost = 10.0  # Default
        if episode_info and "environment_params" in episode_info.get(
            "episode_summary", {}
        ):
            marginal_cost = episode_info["episode_summary"]["environment_params"].get(
                "marginal_cost", 10.0
            )

        profit_margins = (prices - marginal_cost) / prices
        profit_margins = np.clip(profit_margins, 0, 1)  # Clip to [0, 1]

        avg_profit_margin = np.mean(profit_margins, axis=0)
        features.extend(avg_profit_margin.tolist())
        features.append(np.mean(avg_profit_margin))  # Average across firms
        features.append(np.std(avg_profit_margin))  # Std across firms

        # 4. Price coordination features
        # Price correlation between firms
        price_correlations = []
        for i in range(n_firms):
            for j in range(i + 1, n_firms):
                corr = _safe_correlation(prices[:, i], prices[:, j])
                if corr != 0.0:  # Only add non-zero correlations
                    price_correlations.append(corr)
        features.append(np.mean(price_correlations) if price_correlations else 0.0)
        features.append(np.std(price_correlations) if price_correlations else 0.0)

        # 5. Market concentration features
        # Herfindahl-Hirschman Index (HHI) based on market shares
        market_shares = prices / np.sum(prices, axis=1, keepdims=True)
        hhi = np.sum(market_shares**2, axis=1)
        features.append(np.mean(hhi))
        features.append(np.std(hhi))

        # 6. Price stability features
        price_changes = np.diff(prices, axis=0)
        price_volatility = np.std(price_changes, axis=0)
        features.extend(price_volatility.tolist())
        features.append(np.mean(price_volatility))

        # 7. Demand shock response features
        # Correlation between demand shocks and price changes
        if n_steps > 1:
            demand_shock_changes = np.diff(demand_shocks)
            price_change_correlations = []
            for i in range(n_firms):
                price_changes_i = np.diff(prices[:, i])
                if len(price_changes_i) > 0 and len(demand_shock_changes) > 0:
                    corr = _safe_correlation(price_changes_i, demand_shock_changes)
                    if corr != 0.0:  # Only add non-zero correlations
                        price_change_correlations.append(corr)
            features.append(
                np.mean(price_change_correlations) if price_change_correlations else 0.0
            )
        else:
            features.append(0.0)

        # 8. Profit stability features
        profit_volatility = np.std(profits, axis=0)
        features.extend(profit_volatility.tolist())
        features.append(np.mean(profit_volatility))

        # 9. Market price features
        market_price_volatility = np.std(market_prices)
        features.append(market_price_volatility)
        features.append(np.mean(market_prices))

        # 10. Episode length and firm count
        features.append(n_steps)
        features.append(n_firms)

        return np.array(features, dtype=np.float32)  # type: ignore[no-any-return]

    def extract_features_batch(self, log_files: List[Union[str, Path]]) -> np.ndarray:
        """
        Extract features from multiple log files.

        Args:
            log_files: List of paths to JSONL log files

        Returns:
            Array of shape (n_episodes, n_features)
        """
        features_list = []
        valid_files = []

        for log_file in log_files:
            try:
                features = self.extract_features_from_log(log_file)
                features_list.append(features)
                valid_files.append(log_file)
            except (ValueError, FileNotFoundError) as e:
                print(f"Warning: Skipping {log_file}: {e}")
                continue

        if not features_list:
            raise ValueError("No valid log files found")

        return np.array(features_list)  # type: ignore[no-any-return]


class CollusionDetector:
    """
    Machine learning-based collusion detector.

    This class trains and uses ML models to classify episodes as collusive
    or competitive based on extracted features.
    """

    def __init__(
        self,
        model_type: str = "logistic",
        random_state: Optional[int] = None,
        **model_kwargs: Any,
    ) -> None:
        """
        Initialize the collusion detector.

        Args:
            model_type: Type of model to use ("logistic" or "lightgbm")
            random_state: Random state for reproducibility
            **model_kwargs: Additional arguments for the model
        """
        if model_type not in ["logistic", "lightgbm"]:
            raise ValueError("model_type must be 'logistic' or 'lightgbm'")

        if model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not available. Please install it or use 'logistic' model type."
            )

        self.model_type = model_type
        self.random_state = random_state
        self.model_kwargs = model_kwargs

        # Initialize model
        if model_type == "logistic":
            self.model = LogisticRegression(
                random_state=random_state, max_iter=1000, **model_kwargs
            )
        else:  # lightgbm
            self.model = LGBMClassifier(
                random_state=random_state, verbose=-1, **model_kwargs
            )

        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names: Optional[List[str]] = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        validation_split: bool = True,
    ) -> Dict[str, float]:
        """
        Train the collusion detector.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Binary labels (1 for collusive, 0 for competitive)
            test_size: Fraction of data to use for testing
            validation_split: Whether to split data for validation

        Returns:
            Dictionary containing training metrics
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(np.unique(y)) != 2:
            raise ValueError("y must contain exactly 2 unique values")

        # Split data if requested
        if validation_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        # Calculate AUROC
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        auroc = roc_auc_score(y_test, y_pred_proba)

        self.is_trained = True

        return {
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "auroc": auroc,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict collusion for new episodes.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Binary predictions (1 for collusive, 0 for competitive)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        return np.array(self.model.predict(X_scaled))  # type: ignore[no-any-return]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict collusion probabilities for new episodes.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Probability matrix of shape (n_samples, 2) [prob_competitive, prob_collusive]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        return np.array(self.model.predict_proba(X_scaled))  # type: ignore[no-any-return]

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.

        Returns:
            Feature importance array or None if not available
        """
        if not self.is_trained:
            return None

        if hasattr(self.model, "feature_importances_"):
            return np.array(self.model.feature_importances_)  # type: ignore[no-any-return]
        elif hasattr(self.model, "coef_"):
            return np.array(np.abs(self.model.coef_[0]))  # type: ignore[no-any-return]
        else:
            return None


def generate_synthetic_labels(
    log_files: List[Union[str, Path]],
    collusion_ratio: float = 0.5,
    random_state: Optional[int] = None,
) -> Tuple[List[Union[str, Path]], np.ndarray]:
    """
    Generate synthetic labels for collusion detection.

    This function creates synthetic labels by analyzing agent types and
    market behavior patterns to simulate collusive vs competitive episodes.

    Args:
        log_files: List of log file paths
        collusion_ratio: Fraction of episodes to label as collusive
        random_state: Random state for reproducibility

    Returns:
        Tuple of (valid_log_files, labels) where labels are 1 for collusive, 0 for competitive
    """
    if not 0 <= collusion_ratio <= 1:
        raise ValueError("collusion_ratio must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    valid_files = []
    labels = []

    for log_file in log_files:
        try:
            # Parse log file to extract agent types and behavior patterns
            with open(log_file, "r") as f:
                lines = f.readlines()

            # Extract episode header
            episode_header = None
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    if data.get("type") == "episode_header":
                        episode_header = data
                        break
                except json.JSONDecodeError:
                    continue

            if not episode_header:
                continue

            # Extract step data
            steps_data = []
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    if data.get("type") == "step":
                        steps_data.append(data)
                except json.JSONDecodeError:
                    continue

            if len(steps_data) < 10:  # Need sufficient data
                continue

            # Determine label based on agent types and behavior
            # agent_types = episode_header.get("n_firms", 2)  # Not used in current logic

            # Extract prices for analysis
            prices = np.array([step["prices"] for step in steps_data])

            # Calculate price coordination metrics
            price_correlation = 0.0
            if prices.shape[1] > 1:
                # Calculate average correlation between firm prices
                correlations = []
                for i in range(prices.shape[1]):
                    for j in range(i + 1, prices.shape[1]):
                        corr = _safe_correlation(prices[:, i], prices[:, j])
                        if corr != 0.0:  # Only add non-zero correlations
                            correlations.append(corr)
                price_correlation = (
                    float(np.mean(correlations)) if correlations else 0.0
                )

            # Calculate price stability
            price_volatility = np.std(prices)

            # Calculate profit margins
            profits = np.array([step["profits"] for step in steps_data])
            avg_profit = np.mean(profits)

            # Synthetic labeling logic
            is_collusive = False

            # High price correlation suggests collusion
            if price_correlation > 0.7:
                is_collusive = True

            # Low price volatility with high profits suggests collusion
            if price_volatility < 20 and avg_profit > 1000:
                is_collusive = True

            # Random component based on collusion_ratio
            if rng.random() < collusion_ratio:
                is_collusive = True

            valid_files.append(log_file)
            labels.append(1 if is_collusive else 0)

        except Exception as e:
            print(f"Warning: Error processing {log_file}: {e}")
            continue

    return valid_files, np.array(labels, dtype=int)
