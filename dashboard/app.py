"""
Streamlit dashboard for visualizing regulator experiments.

This dashboard provides interactive visualization of episode logs including:
- Price trajectories over time
- Regulator flags and monitoring results
- Consumer surplus vs producer surplus analysis
- Individual firm profit analysis
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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
) -> Tuple[float, float]:
    """
    Calculate consumer surplus and producer surplus.

    Args:
        prices: List of individual firm prices
        market_price: Average market price
        total_demand: Total market demand
        demand_intercept: Demand curve intercept
        demand_slope: Demand curve slope
        marginal_cost: Marginal cost per unit

    Returns:
        Tuple of (consumer_surplus, producer_surplus)
    """
    # Consumer surplus: area under demand curve above market price
    # For demand curve Q = a + b*P where b < 0 (downward sloping)
    # Consumer surplus = 0.5 * (a - P) * Q
    if demand_slope < 0:  # Normal downward-sloping demand
        consumer_surplus = 0.5 * (demand_intercept - market_price) * total_demand
    else:
        # Upward-sloping demand (unusual but possible)
        consumer_surplus = 0.5 * (market_price - demand_intercept) * total_demand

    # Ensure non-negative consumer surplus
    consumer_surplus = max(0.0, consumer_surplus)

    # Producer surplus: total revenue minus variable costs
    # Producer surplus = (market_price - marginal_cost) * total_demand
    producer_surplus = (market_price - marginal_cost) * total_demand

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
        title="Price Trajectories Over Time",
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
        "price_monitoring" in step or "chat_monitoring" in step for step in steps
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

    # Initialize violation tracking
    parallel_violations = []
    structural_break_violations = []
    chat_violations = []
    total_fines = []

    for step in steps:
        # Price monitoring violations
        price_monitoring = step.get("price_monitoring", {})
        parallel_violations.append(
            1 if price_monitoring.get("parallel_violation", False) else 0
        )
        structural_break_violations.append(
            1 if price_monitoring.get("structural_break_violation", False) else 0
        )

        # Chat monitoring violations
        chat_monitoring = step.get("chat_monitoring", {})
        chat_violations.append(
            1 if chat_monitoring.get("collusive_messages", 0) > 0 else 0
        )

        # Total fines
        price_fines = sum(price_monitoring.get("fines_applied", []))
        chat_fines = chat_monitoring.get("fines_applied", 0.0)
        total_fines.append(price_fines + chat_fines)

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Violations Over Time", "Fines Over Time"),
        vertical_spacing=0.1,
    )

    # Violations plot
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=parallel_violations,
            mode="lines+markers",
            name="Parallel Violations",
            line=dict(color="red", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=structural_break_violations,
            mode="lines+markers",
            name="Structural Break Violations",
            line=dict(color="orange", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=chat_violations,
            mode="lines+markers",
            name="Chat Violations",
            line=dict(color="purple", width=2),
        ),
        row=1,
        col=1,
    )

    # Fines plot
    fig.add_trace(
        go.Scatter(
            x=step_numbers,
            y=total_fines,
            mode="lines+markers",
            name="Total Fines",
            line=dict(color="darkred", width=2),
            fill="tonexty",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(title="Regulator Monitoring Results", height=600, showlegend=True)

    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_yaxes(title_text="Violation (0/1)", row=1, col=1)
    fig.update_yaxes(title_text="Fines", row=2, col=1)

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

        consumer_surplus, producer_surplus = calculate_surplus(
            prices,
            market_price,
            total_demand,
            demand_intercept,
            demand_slope,
            marginal_cost,
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
        title="Consumer vs Producer Surplus Over Time",
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
        title="Individual Firm Profits Over Time",
        xaxis_title="Step",
        yaxis_title="Profit",
        hovermode="x unified",
        height=500,
    )

    return fig


def display_episode_summary(episode_data: Dict[str, Any]) -> None:
    """
    Display essential episode summary information.

    Args:
        episode_data: Dictionary containing episode data
    """
    header = episode_data.get("header", {})
    summary = episode_data.get("summary", {})
    steps = episode_data.get("steps", [])

    if not header:
        st.error("No episode header found")
        return

    # Show only essential metrics in a compact format
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Episode ID", header.get("episode_id", "N/A"))

    with col2:
        st.metric("Firms", header.get("n_firms", "N/A"))

    with col3:
        st.metric("Steps", len(steps))

    with col4:
        if summary:
            final_market_price = summary.get("final_market_price")
            if final_market_price is not None:
                st.metric("Final Price", f"{final_market_price:.2f}")
            else:
                st.metric("Final Price", "N/A")


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
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Price Trajectories",
                "🚨 Regulator Flags",
                "💰 Surplus Analysis",
                "💵 Profit Analysis",
            ]
        )

        with tab1:
            st.header("Price Trajectories")
            price_fig = create_price_trajectory_plot(episode_data["steps"])
            st.plotly_chart(price_fig, use_container_width=True)

        with tab2:
            st.header("Regulator Monitoring")
            flags_fig = create_regulator_flags_plot(episode_data["steps"])
            st.plotly_chart(flags_fig, use_container_width=True)

        with tab3:
            st.header("Surplus Analysis")
            surplus_fig = create_surplus_plot(episode_data["steps"])
            st.plotly_chart(surplus_fig, use_container_width=True)

        with tab4:
            st.header("Profit Analysis")
            profit_fig = create_profit_plot(episode_data["steps"])
            st.plotly_chart(profit_fig, use_container_width=True)

        # Display episode summary at the bottom
        st.header("Episode Summary")
        display_episode_summary(episode_data)

    except Exception as e:
        st.error(f"Error loading episode data: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
