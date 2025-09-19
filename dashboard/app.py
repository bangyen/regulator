"""
Streamlit dashboard for visualizing regulator experiments.

This dashboard provides interactive visualization of episode logs including:
- Price trajectories over time
- Regulator flags and monitoring results
- Consumer surplus vs producer surplus analysis
- Individual firm profit analysis
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Suppress numpy warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(divide="ignore", invalid="ignore")

# Dashboard imports


def load_episode_data(log_file: Path) -> Dict[str, Any]:
    """
    Load episode data from a JSONL log file.

    Args:
        log_file: Path to the JSONL log file

    Returns:
        Dictionary containing episode header, steps, and summary data
    """
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    episode_data: Dict[str, Any] = {
        "header": None,
        "steps": [],
        "summary": None,
    }

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())

                if data["type"] == "episode_header":
                    episode_data["header"] = data
                elif data["type"] == "step":
                    episode_data["steps"].append(data)
                elif data["type"] in ["episode_summary", "episode_end"]:
                    if data["type"] == "episode_end" and "episode_summary" in data:
                        episode_data["summary"] = data["episode_summary"]
                    else:
                        episode_data["summary"] = data

    return episode_data


def calculate_surplus(
    prices: List[float],
    market_price: float,
    total_demand: float,
    demand_intercept: float = 100.0,
    demand_slope: float = -1.0,
    marginal_cost: float = 10.0,
    individual_quantities: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Calculate consumer surplus and producer surplus with enhanced economic accuracy.

    Args:
        prices: List of individual firm prices
        market_price: Average market price
        total_demand: Total market demand
        demand_intercept: Demand curve intercept
        demand_slope: Demand curve slope
        marginal_cost: Marginal cost per unit
        individual_quantities: List of quantities sold by each firm (optional)

    Returns:
        Tuple of (consumer_surplus, producer_surplus)
    """
    # Enhanced Consumer Surplus Calculation
    # For linear demand Q = a + b*P, consumer surplus is the area under the demand curve
    # above the market price: CS = 0.5 * (a - P_market) * Q_market
    # This represents the difference between what consumers are willing to pay
    # and what they actually pay

    if demand_slope < 0:  # Normal downward-sloping demand
        # Calculate the price where demand would be zero (choke price)
        # Avoid division by zero
        if abs(demand_slope) < 1e-10:
            choke_price = float("inf")  # Very flat demand curve
        else:
            choke_price = demand_intercept / abs(demand_slope)

        # Consumer surplus = area of triangle above market price
        if market_price < choke_price and total_demand > 0:
            consumer_surplus = 0.5 * (choke_price - market_price) * total_demand
        else:
            consumer_surplus = (
                0.0  # No consumer surplus if price is at or above choke price
            )
    else:
        # Upward-sloping demand (unusual but possible)
        if total_demand > 0:
            consumer_surplus = 0.5 * (market_price - demand_intercept) * total_demand
        else:
            consumer_surplus = 0.0

    # Ensure non-negative consumer surplus
    consumer_surplus = max(0.0, consumer_surplus)

    # Enhanced Producer Surplus Calculation
    # Producer surplus = Total revenue - Total variable costs
    # This represents economic profit above opportunity cost

    if individual_quantities is not None and len(individual_quantities) == len(prices):
        # Calculate using individual firm data for more accuracy
        total_revenue = sum(
            price * quantity
            for price, quantity in zip(prices, individual_quantities)
            if quantity > 0  # Only include positive quantities
        )
        total_variable_costs = marginal_cost * sum(individual_quantities)
        producer_surplus = total_revenue - total_variable_costs
    else:
        # Fallback to market-level calculation
        if total_demand > 0:
            total_revenue = market_price * total_demand
            total_variable_costs = marginal_cost * total_demand
            producer_surplus = total_revenue - total_variable_costs
        else:
            producer_surplus = 0.0

    # Producer surplus can be negative if firms are selling below marginal cost
    # This is economically realistic (firms may accept losses temporarily)

    return consumer_surplus, producer_surplus


def create_price_trajectory_plot(steps: List[Dict[str, Any]]) -> go.Figure:
    """
    Create price trajectory plot showing individual firm prices and market price.

    Args:
        steps: List of step data dictionaries

    Returns:
        Plotly figure object
    """
    if not steps:
        return go.Figure()

    # Extract data
    step_numbers = [step["step"] for step in steps]
    market_prices = [step["market_price"] for step in steps]

    # Get number of firms from first step
    n_firms = len(steps[0]["prices"])

    fig = go.Figure()

    # Add individual firm price lines
    for i in range(n_firms):
        firm_prices = [step["prices"][i] for step in steps]
        fig.add_trace(
            go.Scatter(
                x=step_numbers,
                y=firm_prices,
                mode="lines+markers",
                name=f"Firm {i+1}",
                line=dict(width=2),
                marker=dict(size=4),
            )
        )

    # Add market price line
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=market_prices,
            mode="lines+markers",
            name="Market Price",
            line=dict(width=3, dash="dash"),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title="Price Trajectories",
        title_font_size=24,
        xaxis_title="Step",
        yaxis_title="Price",
        hovermode="x unified",
        height=500,
    )

    return fig


def create_regulator_flags_plot(steps: List[Dict[str, Any]]) -> go.Figure:
    """
    Create plot showing regulator flags and violations over time.

    Args:
        steps: List of step data dictionaries

    Returns:
        Plotly figure object
    """
    if not steps:
        return go.Figure()

    step_numbers = [step["step"] for step in steps]

    # Check if any step has regulator monitoring data
    has_regulator_data = any(
        "price_monitoring" in step
        or "chat_monitoring" in step
        or "regulator_flags" in step
        for step in steps
    )

    if not has_regulator_data:
        # Create a message plot when no regulator data is available
        fig = go.Figure()
        fig.add_annotation(
            text="No regulator monitoring data available for this episode.<br>This episode was run without regulator monitoring enabled.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title="Regulator Monitoring Results",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        return fig

    # Initialize fines tracking
    total_fines = []

    for step in steps:
        # Handle different data formats
        if "regulator_flags" in step:
            # New format: regulator_flags contains all monitoring data
            regulator_flags = step.get("regulator_flags", {})
            # Total fines from regulator_flags
            total_fines.append(sum(regulator_flags.get("fines_applied", [])))
        else:
            # Legacy format: separate price_monitoring and chat_monitoring
            price_monitoring = step.get("price_monitoring", {})
            # Chat monitoring violations
            chat_monitoring = step.get("chat_monitoring", {})
            # Total fines
            price_fines = sum(price_monitoring.get("fines_applied", []))
            chat_fines = chat_monitoring.get("fines_applied", 0.0)
            total_fines.append(price_fines + chat_fines)

    # Create fines-only plot
    fig = go.Figure()

    # Fines plot
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=total_fines,
            mode="lines+markers",
            name="Total Fines",
            line=dict(color="darkred", width=2),
            fill="tonexty",
        )
    )

    # Update layout and axes
    fig.update_layout(
        title="Regulator Flags",
        title_font_size=24,
        height=400,
        showlegend=True,
        xaxis_title="Step",
        yaxis_title="Fines",
    )

    return fig


def create_surplus_plot(steps: List[Dict[str, Any]]) -> go.Figure:
    """
    Create plot showing consumer surplus vs producer surplus over time.

    Args:
        steps: List of step data dictionaries

    Returns:
        Plotly figure object
    """
    if not steps:
        return go.Figure()

    step_numbers = [step["step"] for step in steps]
    consumer_surpluses = []
    producer_surpluses = []

    # Get demand parameters from header if available
    demand_intercept = 100.0
    demand_slope = -1.0
    marginal_cost = 10.0

    for step in steps:
        prices = step["prices"]
        market_price = step["market_price"]
        total_demand = step.get("total_demand", 0.0)
        individual_quantities = step.get("individual_quantity", None)

        consumer_surplus, producer_surplus = calculate_surplus(
            prices,
            market_price,
            total_demand,
            demand_intercept,
            demand_slope,
            marginal_cost,
            individual_quantities,
        )

        consumer_surpluses.append(consumer_surplus)
        producer_surpluses.append(producer_surplus)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=consumer_surpluses,
            mode="lines+markers",
            name="Consumer Surplus",
            line=dict(color="green", width=2),
            fill="tonexty",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=producer_surpluses,
            mode="lines+markers",
            name="Producer Surplus",
            line=dict(color="blue", width=2),
            fill="tozeroy",
        )
    )

    fig.update_layout(
        title="Surplus Analysis",
        title_font_size=24,
        xaxis_title="Step",
        yaxis_title="Surplus",
        hovermode="x unified",
        height=500,
    )

    return fig


def create_profit_plot(steps: List[Dict[str, Any]]) -> go.Figure:
    """
    Create plot showing individual firm profits over time.

    Args:
        steps: List of step data dictionaries

    Returns:
        Plotly figure object
    """
    if not steps:
        return go.Figure()

    step_numbers = [step["step"] for step in steps]
    n_firms = len(steps[0]["profits"])

    fig = go.Figure()

    for i in range(n_firms):
        firm_profits = [step["profits"][i] for step in steps]
        fig.add_trace(
            go.Scatter(
                x=step_numbers,
                y=firm_profits,
                mode="lines+markers",
                name=f"Firm {i+1} Profits",
                line=dict(width=2),
                marker=dict(size=4),
            )
        )

    fig.update_layout(
        title="Profit Analysis",
        title_font_size=24,
        xaxis_title="Step",
        yaxis_title="Profit",
        hovermode="x unified",
        height=500,
    )

    return fig


def display_episode_summary(episode_data: Dict[str, Any]) -> None:
    """
    Display comprehensive episode summary information including enhanced monitoring data.

    Args:
        episode_data: Dictionary containing episode data
    """
    header = episode_data.get("header", {})
    steps = episode_data.get("steps", [])

    if not header:
        st.error("No episode header found")
        return

    # Enhanced episode information - same 4 metrics but more informative
    col1, col2, col3, col4 = st.columns(4)

    # Check if this is an enhanced episode (has market volatility data)
    has_enhanced_data = any(
        step.get("regulator_flags", {}).get("market_volatility") is not None
        for step in steps
    )

    with col1:
        if has_enhanced_data:
            # Show total violations for enhanced episodes
            parallel_violations = sum(
                1
                for step in steps
                if step.get("regulator_flags", {}).get("parallel_violation", False)
            )
            structural_violations = sum(
                1
                for step in steps
                if step.get("regulator_flags", {}).get(
                    "structural_break_violation", False
                )
            )
            total_violations = parallel_violations + structural_violations
            st.metric("Total Violations", total_violations)
        else:
            # Show steps for standard episodes
            st.metric("Steps", len(steps))

    with col2:
        if has_enhanced_data:
            # Show average market volatility for enhanced episodes
            market_volatilities = [
                step.get("regulator_flags", {}).get("market_volatility", 0)
                for step in steps
            ]
            avg_volatility = (
                sum(market_volatilities) / len(market_volatilities)
                if market_volatilities
                else 0
            )
            st.metric("Avg Market Volatility", f"{avg_volatility:.3f}")
        else:
            # Show total violations for standard episodes
            parallel_violations = sum(
                1
                for step in steps
                if step.get("regulator_flags", {}).get("parallel_violation", False)
            )
            structural_violations = sum(
                1
                for step in steps
                if step.get("regulator_flags", {}).get(
                    "structural_break_violation", False
                )
            )
            total_violations = parallel_violations + structural_violations
            st.metric("Total Violations", total_violations)

    with col3:
        if has_enhanced_data:
            # Show total fines for enhanced episodes
            total_fines = 0
            for step in steps:
                fines = step.get("regulator_flags", {}).get("fines_applied", [0])
                if isinstance(fines, list):
                    total_fines += sum(fines)
                else:
                    total_fines += fines
            st.metric("Total Fines", f"{total_fines:.2f}")
        else:
            # Show total fines for standard episodes
            total_fines = 0
            for step in steps:
                fines = step.get("regulator_flags", {}).get("fines_applied", [0])
                if isinstance(fines, list):
                    total_fines += sum(fines)
                else:
                    total_fines += fines
            st.metric("Total Fines", f"{total_fines:.2f}")

    with col4:
        if has_enhanced_data:
            # Show average penalty multiplier for enhanced episodes
            penalty_multipliers = []
            for step in steps:
                flags = step.get("regulator_flags", {})
                if "penalty_multipliers" in flags:
                    penalty_multipliers.extend(flags["penalty_multipliers"])
            avg_multiplier = (
                sum(penalty_multipliers) / len(penalty_multipliers)
                if penalty_multipliers
                else 0
            )
            st.metric("Avg Penalty Multiplier", f"{avg_multiplier:.2f}x")
        else:
            # Show violation rate for standard episodes
            parallel_violations = sum(
                1
                for step in steps
                if step.get("regulator_flags", {}).get("parallel_violation", False)
            )
            structural_violations = sum(
                1
                for step in steps
                if step.get("regulator_flags", {}).get(
                    "structural_break_violation", False
                )
            )
            total_violations = parallel_violations + structural_violations
            violation_rate = (total_violations / len(steps)) * 100 if steps else 0
            st.metric("Violation Rate", f"{violation_rate:.1f}%")


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Regulator Experiment Dashboard", page_icon="📊", layout="wide"
    )

    st.title("📊 Regulator Experiment Dashboard")
    st.markdown("Visualize and analyze regulator experiment episodes")

    # Sidebar for file selection
    st.sidebar.header("Episode Selection")

    # Get available log files
    logs_dir = Path("logs")
    if not logs_dir.exists():
        st.error(
            "Logs directory not found. Please ensure you have episode logs in the 'logs' directory."
        )
        return

    log_files = list(logs_dir.glob("*.jsonl"))
    if not log_files:
        st.error("No JSONL log files found in the logs directory.")
        return

    # File selector
    selected_file = st.sidebar.selectbox(
        "Select Episode Log File", options=log_files, format_func=lambda x: x.name
    )

    if not selected_file:
        st.warning("Please select a log file to continue.")
        return

    try:
        # Load episode data
        episode_data = load_episode_data(selected_file)

        if not episode_data["steps"]:
            st.error("No step data found in the selected log file.")
            return

        # Create tabs for different visualizations
        tab_names = [
            "📈 Price Trajectories",
            "🚨 Regulator Flags",
            "💰 Surplus Analysis",
            "💵 Profit Analysis",
        ]

        tabs = st.tabs(tab_names)
        tab1, tab2, tab3, tab4 = tabs

        with tab1:
            price_fig = create_price_trajectory_plot(episode_data["steps"])
            st.plotly_chart(price_fig, use_container_width=True)

        with tab2:
            flags_fig = create_regulator_flags_plot(episode_data["steps"])
            st.plotly_chart(flags_fig, use_container_width=True)

        with tab3:
            surplus_fig = create_surplus_plot(episode_data["steps"])
            st.plotly_chart(surplus_fig, use_container_width=True)

        with tab4:
            profit_fig = create_profit_plot(episode_data["steps"])
            st.plotly_chart(profit_fig, use_container_width=True)

        # Display episode summary at the bottom
        st.subheader("Episode Summary")
        display_episode_summary(episode_data)

    except Exception as e:
        st.error(f"Error loading episode data: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
