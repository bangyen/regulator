"""
Streamlit dashboard for visualizing regulator experiments.

This dashboard provides interactive visualization of episode logs including:
- Price trajectories over time
- Regulator flags and monitoring results
- Consumer surplus vs producer surplus analysis
- Episode replay functionality
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
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
                    episode_data["summary"] = data

    return episode_data


def calculate_surplus(
    prices: List[float],
    market_price: float,
    total_demand: float,
    demand_intercept: float = 100.0,
    demand_slope: float = -1.0,
) -> Tuple[float, float]:
    """
    Calculate consumer surplus and producer surplus.

    Args:
        prices: List of individual firm prices
        market_price: Average market price
        total_demand: Total market demand
        demand_intercept: Demand curve intercept
        demand_slope: Demand curve slope

    Returns:
        Tuple of (consumer_surplus, producer_surplus)
    """
    # Consumer surplus: area under demand curve above market price
    # Assuming linear demand: Q = a - b*P, so P = (a - Q)/b
    if demand_slope != 0:
        max_price = demand_intercept / abs(demand_slope)
        consumer_surplus = 0.5 * (max_price - market_price) * total_demand
    else:
        consumer_surplus = 0.0

    # Producer surplus: total revenue minus variable costs
    # Assuming marginal cost = 10 (from environment params)
    marginal_cost = 10.0
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

    for step in steps:
        prices = step["prices"]
        market_price = step["market_price"]
        total_demand = step.get("total_demand", 0.0)

        consumer_surplus, producer_surplus = calculate_surplus(
            prices, market_price, total_demand, demand_intercept, demand_slope
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
    Display episode summary information.

    Args:
        episode_data: Dictionary containing episode data
    """
    header = episode_data.get("header", {})
    summary = episode_data.get("summary", {})
    steps = episode_data.get("steps", [])

    if not header:
        st.error("No episode header found")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Episode ID", header.get("episode_id", "N/A"))
        st.metric("Number of Firms", header.get("n_firms", "N/A"))

    with col2:
        st.metric("Total Steps", len(steps))
        if summary:
            st.metric("Duration (s)", f"{summary.get('duration_seconds', 0):.3f}")

    with col3:
        if summary:
            st.metric("Total Reward", f"{summary.get('total_reward', 0):.2f}")
            final_prices = summary.get("final_prices", [])
            if final_prices:
                st.metric("Final Market Price", f"{np.mean(final_prices):.2f}")

    # Agent types
    agent_types = header.get("agent_types", [])
    if agent_types:
        st.subheader("Agent Types")
        for i, agent_type in enumerate(agent_types):
            st.write(f"Firm {i+1}: {agent_type}")

    # Environment parameters
    env_params = header.get("environment_params", {})
    if env_params:
        st.subheader("Environment Parameters")
        for key, value in env_params.items():
            st.write(f"**{key}**: {value}")


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

        # Display episode summary
        st.header("Episode Summary")
        display_episode_summary(episode_data)

        st.divider()

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

        # Episode replay section
        st.divider()
        st.header("Episode Replay")

        # Create replay controls
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🎬 Start Replay"):
                st.session_state.replay_active = True
                st.session_state.replay_step = 0

        with col2:
            if st.button("⏸️ Pause"):
                st.session_state.replay_active = False

        with col3:
            if st.button("⏹️ Stop"):
                st.session_state.replay_active = False
                st.session_state.replay_step = 0

        with col4:
            if st.button("🔄 Reset"):
                st.session_state.replay_step = 0

        # Replay controls
        if "replay_active" not in st.session_state:
            st.session_state.replay_active = False
        if "replay_step" not in st.session_state:
            st.session_state.replay_step = 0

        # Step selector
        max_steps = len(episode_data["steps"]) - 1
        selected_step = st.slider(
            "Select Step",
            min_value=0,
            max_value=max_steps,
            value=st.session_state.replay_step,
            key="step_slider",
        )

        # Update replay step when slider changes
        if selected_step != st.session_state.replay_step:
            st.session_state.replay_step = selected_step

        # Display current step information
        if episode_data["steps"]:
            current_step_data = episode_data["steps"][st.session_state.replay_step]

            st.subheader(f"Step {st.session_state.replay_step + 1}")

            # Create columns for step information
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Prices", f"{current_step_data.get('prices', 'N/A')}")
                st.metric("Profits", f"{current_step_data.get('profits', 'N/A')}")

            with col2:
                st.metric(
                    "Consumer Surplus",
                    f"{current_step_data.get('consumer_surplus', 'N/A')}",
                )
                st.metric(
                    "Producer Surplus",
                    f"{current_step_data.get('producer_surplus', 'N/A')}",
                )

            with col3:
                st.metric("Total Fines", f"{current_step_data.get('total_fines', 0)}")
                st.metric(
                    "Violations", f"{current_step_data.get('violations_detected', 0)}"
                )

            # Display step-specific visualizations
            if st.checkbox("Show Step-by-Step Visualization"):
                # Create a focused plot for the current step
                import plotly.graph_objects as go

                # Price trajectory up to current step
                steps_to_show = episode_data["steps"][
                    : st.session_state.replay_step + 1
                ]
                step_numbers = list(range(1, len(steps_to_show) + 1))

                fig = go.Figure()

                # Add price lines for each firm
                n_firms = len(steps_to_show[0].get("prices", []))
                for firm_id in range(n_firms):
                    prices = [
                        step.get("prices", [0] * n_firms)[firm_id]
                        for step in steps_to_show
                    ]
                    fig.add_trace(
                        go.Scatter(
                            x=step_numbers,
                            y=prices,
                            mode="lines+markers",
                            name=f"Firm {firm_id + 1}",
                            line=dict(width=2),
                            marker=dict(size=6),
                        )
                    )

                # Highlight current step
                if steps_to_show:
                    current_prices = steps_to_show[-1].get("prices", [])
                    fig.add_trace(
                        go.Scatter(
                            x=[len(steps_to_show)],
                            y=current_prices,
                            mode="markers",
                            name="Current Step",
                            marker=dict(size=12, color="red", symbol="star"),
                            showlegend=True,
                        )
                    )

                fig.update_layout(
                    title=f"Price Trajectory (Steps 1-{len(steps_to_show)})",
                    xaxis_title="Step",
                    yaxis_title="Price",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

            # Auto-advance replay
            if (
                st.session_state.replay_active
                and st.session_state.replay_step < max_steps
            ):
                import time

                time.sleep(0.5)  # Pause between steps
                st.session_state.replay_step += 1
                st.rerun()

        # Export functionality
        if st.button("📥 Download Episode Data"):
            import json

            # Create downloadable JSON
            json_data = json.dumps(episode_data, indent=2, default=str)

            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"episode_{episode_data.get('episode_id', 'unknown')}.json",
                mime="application/json",
            )

    except Exception as e:
        st.error(f"Error loading episode data: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
