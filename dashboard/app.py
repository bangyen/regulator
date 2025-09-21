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


def create_market_overview_plot(steps: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a comprehensive market overview showing key market metrics.

    Args:
        steps: List of step data dictionaries

    Returns:
        Plotly figure object with subplots
    """
    if not steps:
        return go.Figure()

    step_numbers = [step["step"] for step in steps]

    # Extract market data
    market_prices = [step["market_price"] for step in steps]
    total_demands = [step.get("total_demand", 0.0) for step in steps]
    demand_shocks = [step.get("demand_shock", 0.0) for step in steps]

    # Calculate price volatility (rolling standard deviation)
    price_volatility = []
    window_size = min(5, len(market_prices))
    for i in range(len(market_prices)):
        start_idx = max(0, i - window_size + 1)
        window_prices = market_prices[start_idx : i + 1]
        if len(window_prices) > 1:
            volatility = np.std(window_prices)
        else:
            volatility = 0.0
        price_volatility.append(volatility)

    # Calculate market concentration (Herfindahl-Hirschman Index)
    market_concentration = []
    for step in steps:
        prices = step["prices"]
        if len(prices) > 0:
            # Calculate market shares based on prices (simplified)
            total_price = sum(prices)
            if total_price > 0:
                shares = [p / total_price for p in prices]
                hhi = sum(s**2 for s in shares) * 10000  # Scale to 0-10000
            else:
                hhi = 0
        else:
            hhi = 0
        market_concentration.append(hhi)

    # Create subplots
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Market Price & Demand",
            "Price Volatility",
            "Demand Shocks",
            "Market Concentration (HHI)",
        ),
        specs=[
            [{"secondary_y": True}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Market Price & Demand
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=market_prices,
            name="Market Price",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=total_demands,
            name="Total Demand",
            line=dict(color="green", width=2),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Price Volatility
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=price_volatility,
            name="Price Volatility",
            line=dict(color="red", width=2),
            fill="tozeroy",
        ),
        row=1,
        col=2,
    )

    # Demand Shocks
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=demand_shocks,
            name="Demand Shocks",
            line=dict(color="orange", width=2),
            fill="tonexty" if len(demand_shocks) > 0 else None,
        ),
        row=2,
        col=1,
    )

    # Market Concentration
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=market_concentration,
            name="Market Concentration",
            line=dict(color="purple", width=2),
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title="Market Overview", title_font_size=24, height=600, showlegend=True
    )

    # Update axes
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=2)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Demand", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Volatility", row=1, col=2)
    fig.update_yaxes(title_text="Shock", row=2, col=1)
    fig.update_yaxes(title_text="HHI", row=2, col=2)

    return fig


def create_enhanced_regulator_plot(steps: List[Dict[str, Any]]) -> go.Figure:
    """
    Create enhanced regulator monitoring plot with continuous risk scores and ML detection.

    Args:
        steps: List of step data dictionaries

    Returns:
        Plotly figure object
    """
    if not steps:
        return go.Figure()

    step_numbers = [step["step"] for step in steps]

    # Check if any step has regulator monitoring data
    has_regulator_data = any("regulator_flags" in step for step in steps)

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
            title="Enhanced Regulator Monitoring",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        return fig

    # Create subplots for comprehensive monitoring view
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Fines Applied",
            "Risk Scores",
            "Market Volatility",
            "Violation Types",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Extract monitoring data
    total_fines = []
    risk_scores = []
    market_volatilities = []
    parallel_violations = []
    structural_violations = []

    for step in steps:
        regulator_flags = step.get("regulator_flags", {})

        # Total fines
        fines = regulator_flags.get("fines_applied", [0])
        if isinstance(fines, list):
            total_fines.append(sum(fines))
        else:
            total_fines.append(fines)

        # Risk scores (if available)
        risk_score = regulator_flags.get("risk_score", 0.0)
        risk_scores.append(risk_score)

        # Market volatility (if available)
        market_volatility = regulator_flags.get("market_volatility", 0.0)
        market_volatilities.append(market_volatility)

        # Violation types
        parallel_violations.append(
            1 if regulator_flags.get("parallel_violation", False) else 0
        )
        structural_violations.append(
            1 if regulator_flags.get("structural_break_violation", False) else 0
        )

    # Fines Applied
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=total_fines,
            mode="lines+markers",
            name="Total Fines",
            line=dict(color="darkred", width=2),
            fill="tonexty",
        ),
        row=1,
        col=1,
    )

    # Risk Scores
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=risk_scores,
            mode="lines+markers",
            name="Risk Score",
            line=dict(color="orange", width=2),
        ),
        row=1,
        col=2,
    )

    # Market Volatility
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=market_volatilities,
            mode="lines+markers",
            name="Market Volatility",
            line=dict(color="purple", width=2),
        ),
        row=2,
        col=1,
    )

    # Violation Types
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=parallel_violations,
            mode="markers",
            name="Parallel Violations",
            marker=dict(color="red", size=8, symbol="circle"),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=structural_violations,
            mode="markers",
            name="Structural Violations",
            marker=dict(color="blue", size=8, symbol="diamond"),
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title="Enhanced Regulator Monitoring",
        title_font_size=24,
        height=600,
        showlegend=True,
    )

    # Update axes
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=2)
    fig.update_yaxes(title_text="Fines", row=1, col=1)
    fig.update_yaxes(title_text="Risk Score", row=1, col=2)
    fig.update_yaxes(title_text="Volatility", row=2, col=1)
    fig.update_yaxes(title_text="Violations", row=2, col=2)

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
            "🚨 Regulator Monitoring",
            "📊 Market Overview",
        ]

        tabs = st.tabs(tab_names)
        tab1, tab2, tab3 = tabs

        with tab1:
            price_fig = create_price_trajectory_plot(episode_data["steps"])
            st.plotly_chart(price_fig, use_container_width=True)

        with tab2:
            enhanced_flags_fig = create_enhanced_regulator_plot(episode_data["steps"])
            st.plotly_chart(enhanced_flags_fig, use_container_width=True)

        with tab3:
            market_overview_fig = create_market_overview_plot(episode_data["steps"])
            st.plotly_chart(market_overview_fig, use_container_width=True)

        # Display episode summary at the bottom
        st.subheader("Episode Summary")
        display_episode_summary(episode_data)

    except Exception as e:
        st.error(f"Error loading episode data: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
