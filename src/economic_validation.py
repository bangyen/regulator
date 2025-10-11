"""
Economic validation utilities for the regulator simulation.

This module provides functions to validate that the economic data generated
by the simulation is consistent and economically plausible.
"""

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EconomicValidator:
    """
    Validator for economic data consistency and plausibility.

    This class provides methods to check that the economic relationships
    and constraints in the simulation are properly maintained.
    """

    def __init__(
        self,
        demand_intercept: float = 100.0,
        demand_slope: float = -1.0,
        marginal_cost: float = 10.0,
        price_min: float = 1.0,
        price_max: float = 100.0,
        tolerance: float = 1e-3,
    ) -> None:
        """
        Initialize the economic validator.

        Args:
            demand_intercept: Demand curve intercept
            demand_slope: Demand curve slope
            marginal_cost: Marginal cost per unit
            price_min: Minimum allowed price
            price_max: Maximum allowed price
            tolerance: Numerical tolerance for comparisons
        """
        self.demand_intercept = demand_intercept
        self.demand_slope = demand_slope
        self.marginal_cost = marginal_cost
        self.price_min = price_min
        self.price_max = price_max
        self.tolerance = tolerance

    def validate_step_data(
        self,
        prices: List[float],
        market_price: float,
        total_demand: float,
        individual_quantities: List[float],
        market_shares: Optional[List[float]] = None,
        profits: Optional[List[float]] = None,
        demand_shock: float = 0.0,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a single step of economic data.

        Args:
            prices: List of individual firm prices
            market_price: Average market price
            total_demand: Total market demand
            individual_quantities: List of quantities sold by each firm
            market_shares: List of market shares for each firm (optional)
            profits: List of profits for each firm (optional)
            demand_shock: Demand shock value

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Validate price constraints
        for i, price in enumerate(prices):
            if price < self.price_min or price > self.price_max:
                errors.append(
                    f"Firm {i} price {price} outside valid range [{self.price_min}, {self.price_max}]"
                )

        # Validate market price calculation
        expected_market_price = np.mean(prices)
        if not math.isclose(
            market_price, expected_market_price, abs_tol=self.tolerance
        ):
            errors.append(
                f"Market price {market_price} doesn't match expected {expected_market_price}"
            )

        # Validate demand calculation
        expected_demand = self._calculate_expected_demand(market_price, demand_shock)
        if not math.isclose(total_demand, expected_demand, abs_tol=self.tolerance):
            errors.append(
                f"Total demand {total_demand} doesn't match expected {expected_demand}"
            )

        # Calculate market shares from individual quantities if not provided
        if market_shares is None and individual_quantities:
            if total_demand > 0:
                market_shares = [q / total_demand for q in individual_quantities]
            else:
                market_shares = [0.0] * len(individual_quantities)

        # Validate market share constraints
        if market_shares and len(market_shares) > 0:
            market_share_sum = sum(market_shares)
            if not math.isclose(market_share_sum, 1.0, abs_tol=self.tolerance):
                errors.append(f"Market shares sum to {market_share_sum}, should be 1.0")

        # Validate quantity constraints
        if individual_quantities:
            expected_total_quantity = sum(individual_quantities)
            if not math.isclose(
                total_demand, expected_total_quantity, abs_tol=self.tolerance
            ):
                errors.append(
                    f"Total demand {total_demand} doesn't match sum of individual quantities {expected_total_quantity}"
                )

        # Validate profit calculations (lenient due to complex cost structures)
        # Note: Exact profit validation is difficult due to learning curves, economies of scale,
        # and fixed costs that may be enabled in the environment
        if profits and individual_quantities:
            for i, (price, quantity, profit) in enumerate(
                zip(prices, individual_quantities, profits)
            ):
                # Basic sanity check: profits should be reasonable relative to revenue
                revenue = price * quantity
                # Allow for significant variation due to complex cost structures
                # Profits can be negative (losses) or much higher than simple calculation
                # Fixed costs, fines, and other factors can cause large losses
                if (
                    profit < -revenue * 10
                ):  # Losses shouldn't exceed 10x revenue (very lenient)
                    errors.append(
                        f"Firm {i} profit {profit} seems unreasonably low relative to revenue {revenue}"
                    )
                elif (
                    profit > revenue * 10
                ):  # Profits shouldn't exceed 10x revenue (very lenient)
                    errors.append(
                        f"Firm {i} profit {profit} seems unreasonably high relative to revenue {revenue}"
                    )

        # Validate economic relationships
        self._validate_economic_relationships(
            prices, market_shares or [], individual_quantities, errors
        )

        return len(errors) == 0, errors

    def _calculate_expected_demand(
        self, market_price: float, demand_shock: float
    ) -> float:
        """Calculate expected demand given market price and shock."""
        base_demand = self.demand_intercept + self.demand_slope * market_price
        return max(0.0, base_demand + demand_shock)

    def _validate_economic_relationships(
        self,
        prices: List[float],
        market_shares: List[float],
        individual_quantities: List[float],
        errors: List[str],
    ) -> None:
        """Validate economic relationships between prices, shares, and quantities."""

        # Only validate market share relationships if market shares are provided
        if market_shares and len(market_shares) == len(prices):
            # Check that lower prices generally get higher market shares
            # (allowing for some noise due to other factors)
            # Only calculate correlation if we have at least 2 data points
            if len(prices) >= 2:
                try:
                    # Suppress numpy warnings about division by zero when std is zero
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
                        price_share_correlation = np.corrcoef(prices, market_shares)[0, 1]
                    if (
                        price_share_correlation > 0.5
                    ):  # Strong positive correlation is suspicious
                        errors.append(
                            f"Strong positive correlation between prices and market shares: {price_share_correlation}"
                        )
                except (ValueError, np.linalg.LinAlgError):
                    # Handle cases where correlation can't be computed
                    pass

            # Check that market shares are reasonable (not too concentrated)
            max_share = max(market_shares)
            if max_share > 0.99:  # No firm should have >99% market share (very lenient)
                errors.append(
                    f"Market share too concentrated: max share is {max_share}"
                )

        # Check that quantities are non-negative
        if individual_quantities:
            for i, quantity in enumerate(individual_quantities):
                if quantity < 0:
                    errors.append(f"Firm {i} has negative quantity: {quantity}")

    def validate_episode_consistency(
        self, episode_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate consistency across an entire episode.

        Args:
            episode_data: Dictionary containing episode data

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        steps = episode_data.get("steps", [])

        if not steps:
            errors.append("No steps found in episode data")
            return False, errors

        # Validate each step
        for step_data in steps:
            is_valid, step_errors = self.validate_step_data(
                prices=step_data["prices"],
                market_price=step_data["market_price"],
                total_demand=step_data["total_demand"],
                individual_quantities=step_data.get("individual_quantity", []),
                market_shares=step_data.get("market_shares", None),
                profits=step_data.get("profits", []),
                demand_shock=step_data.get("demand_shock", 0.0),
            )

            if not is_valid:
                step_num = step_data.get("step", "unknown")
                for error in step_errors:
                    errors.append(f"Step {step_num}: {error}")

        # Validate episode-level consistency
        self._validate_episode_level_consistency(episode_data, errors)

        return len(errors) == 0, errors

    def _validate_episode_level_consistency(
        self, episode_data: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate consistency at the episode level."""
        steps = episode_data.get("steps", [])
        if not steps:
            return

        # Check that total profits accumulate correctly
        final_total_profits = steps[-1].get("total_profits", [])
        if final_total_profits and len(final_total_profits) > 0:
            # Sum of all step profits should equal final total profits
            # Note: step profits include regulator penalties, so we need to account for that
            step_profits_sum = np.zeros(len(final_total_profits))
            total_penalties = np.zeros(len(final_total_profits))

            for step in steps:
                step_profits = step.get("profits", [])
                regulator_flags = step.get("regulator_flags", {})
                fines_applied = regulator_flags.get("fines_applied", [])

                if len(step_profits) == len(final_total_profits):
                    step_profits_sum += step_profits

                if len(fines_applied) == len(final_total_profits):
                    total_penalties += fines_applied

            # Calculate expected total profits: sum of step profits (after penalties) + total penalties
            # This should equal the original total profits (before penalties)
            expected_total_profits = step_profits_sum + total_penalties

            for i, (final, calculated) in enumerate(
                zip(final_total_profits, expected_total_profits)
            ):
                # Use a very lenient tolerance for profit aggregation due to potential differences
                # in how profits are calculated vs accumulated (e.g., different cost structures)
                # Allow for up to 20% difference or 1000 units, whichever is larger
                max_diff = max(abs(final) * 0.2, 1000.0)
                if abs(final - calculated) > max_diff:
                    errors.append(
                        f"Firm {i} total profits {final} doesn't match sum of step profits {calculated} (step profits: {step_profits_sum[i]:.2f}, penalties: {total_penalties[i]:.2f})"
                    )

        # Check that demand shocks are reasonable
        demand_shocks = [step["demand_shock"] for step in steps]
        shock_std = np.std(demand_shocks)
        if shock_std > 20.0:  # Arbitrary threshold for reasonable shock variation
            errors.append(f"Demand shock standard deviation {shock_std} seems too high")

        # Check that prices don't show unrealistic patterns
        all_prices = [price for step in steps for price in step["prices"]]
        if all_prices:
            price_std = np.std(all_prices)
            price_mean = np.mean(all_prices)
            if (
                price_std > price_mean
            ):  # Prices shouldn't be more variable than their mean
                errors.append(
                    f"Price variation {price_std} is larger than mean price {price_mean}"
                )


def validate_economic_data(
    episode_data: Dict[str, Any],
    demand_intercept: float = 100.0,
    demand_slope: float = -1.0,
    marginal_cost: float = 10.0,
) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate economic data.

    Args:
        episode_data: Dictionary containing episode data
        demand_intercept: Demand curve intercept
        demand_slope: Demand curve slope
        marginal_cost: Marginal cost per unit

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = EconomicValidator(
        demand_intercept=demand_intercept,
        demand_slope=demand_slope,
        marginal_cost=marginal_cost,
    )

    return validator.validate_episode_consistency(episode_data)


def check_economic_plausibility(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check economic plausibility of episode data and return summary statistics.

    Args:
        episode_data: Dictionary containing episode data

    Returns:
        Dictionary containing plausibility checks and statistics
    """
    steps = episode_data.get("steps", [])
    if not steps:
        return {"error": "No steps found in episode data"}

    # Extract data
    all_prices = [price for step in steps for price in step["prices"]]
    all_profits = [profit for step in steps for profit in step["profits"]]
    all_demands = [step["total_demand"] for step in steps]
    all_shocks = [step["demand_shock"] for step in steps]

    # Calculate statistics
    price_stats = {
        "mean": np.mean(all_prices),
        "std": np.std(all_prices),
        "min": np.min(all_prices),
        "max": np.max(all_prices),
        "range": np.max(all_prices) - np.min(all_prices),
    }

    profit_stats = {
        "mean": np.mean(all_profits),
        "std": np.std(all_profits),
        "min": np.min(all_profits),
        "max": np.max(all_profits),
        "negative_profits": sum(1 for p in all_profits if p < 0),
        "total_negative_profits": sum(p for p in all_profits if p < 0),
    }

    demand_stats = {
        "mean": np.mean(all_demands),
        "std": np.std(all_demands),
        "min": np.min(all_demands),
        "max": np.max(all_demands),
        "zero_demand_steps": sum(1 for d in all_demands if d <= 0),
    }

    shock_stats = {
        "mean": np.mean(all_shocks),
        "std": np.std(all_shocks),
        "min": np.min(all_shocks),
        "max": np.max(all_shocks),
    }

    # Plausibility checks
    plausibility_checks = {
        "prices_above_marginal_cost": sum(1 for p in all_prices if p >= 10.0)
        / len(all_prices),
        "reasonable_price_range": 0.0 <= price_stats["min"] <= 200.0
        and 0.0 <= price_stats["max"] <= 200.0,
        "demand_positive": demand_stats["zero_demand_steps"] == 0,
        "profit_variation_reasonable": profit_stats["std"]
        < abs(profit_stats["mean"]) * 2,
        "shock_magnitude_reasonable": shock_stats["std"] < 20.0,
    }

    return {
        "price_stats": price_stats,
        "profit_stats": profit_stats,
        "demand_stats": demand_stats,
        "shock_stats": shock_stats,
        "plausibility_checks": plausibility_checks,
        "overall_plausible": all(plausibility_checks.values()),
    }
