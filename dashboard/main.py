"""Flask application for real-time regulator monitoring dashboard.

This dashboard provides real-time monitoring of cartel detection simulations,
including price trajectories, violations, and regulatory enforcement metrics.
"""

import json
import logging
import os
import random
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flask import Flask, Response, jsonify, render_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Track running experiments
running_experiments = {"status": "idle", "progress": None, "error_message": None}


def load_latest_experiment() -> Optional[Dict[str, Any]]:
    """Load most recent experiment data from logs directory.
    
    Returns parsed experiment data including step-by-step metrics,
    or None if no experiment logs are found.
    """
    log_dir = Path(__file__).parent.parent / "logs"
    if not log_dir.exists():
        logger.warning(f"Log directory does not exist: {log_dir}")
        return None

    log_files = sorted(log_dir.glob("*.jsonl"), key=os.path.getmtime, reverse=True)
    if not log_files:
        logger.info("No experiment log files found")
        return None

    steps = []
    n_firms = 2  # Default
    try:
        with open(log_files[0], "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    steps.append(data)
                    # Get n_firms from header
                    if data.get("type") == "episode_header":
                        n_firms = data.get("n_firms", 2)
        
        logger.info(f"Loaded {len(steps)} steps from {log_files[0].name}")
        return {"steps": steps, "file": str(log_files[0]), "n_firms": n_firms}
    except Exception as e:
        logger.error(f"Failed to load experiment data: {e}")
        return None


def calculate_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate aggregate metrics from experiment data.
    
    Computes summary statistics including price averages, violation counts,
    total fines, and risk scores across all simulation steps.
    """
    steps = data.get("steps", [])
    if not steps:
        return {}

    n_firms = data.get("n_firms", 2)
    max_fines = n_firms * 50.0  # Each firm can be fined $50

    # Filter to only actual step entries
    step_entries = [
        s for s in steps if s.get("type") == "step" and s.get("step", 0) > 0
    ]

    prices = [s.get("market_price", 0) for s in step_entries if "market_price" in s]

    violations = sum(
        1
        for s in step_entries
        if s.get("regulator_flags", {}).get("parallel_violation", False)
        or s.get("regulator_flags", {}).get("structural_break_violation", False)
        or s.get("regulator_flags", {}).get("flagged", False)
    )

    total_fines = sum(
        sum(s.get("regulator_flags", {}).get("fines_applied", []))
        for s in step_entries
        if "regulator_flags" in s
    )

    # Calculate risk scores from fines or use existing risk_score
    risk_scores = []
    for s in step_entries:
        if "regulator_flags" in s:
            flags = s["regulator_flags"]
            # Use risk_score if available, otherwise calculate from fines
            if "risk_score" in flags and flags["risk_score"] is not None:
                risk_scores.append(flags["risk_score"])
            else:
                fines = sum(flags.get("fines_applied", []))
                risk_scores.append(min(fines / max_fines, 1.0))

    return {
        "total_steps": len(step_entries),
        "avg_price": round(np.mean(prices), 2) if prices else 0,
        "price_volatility": round(np.std(prices), 2) if prices else 0,
        "total_violations": violations,
        "total_fines": round(total_fines, 2),
        "avg_risk_score": round(np.mean(risk_scores), 2) if risk_scores else 0,
        "current_risk": round(risk_scores[-1], 2) if risk_scores else 0,
    }


def extract_time_series(data: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Extract time series data for visualization.
    
    Transforms step-by-step experiment data into time series arrays
    suitable for charting prices, profits, violations, and fines.
    """
    steps = data.get("steps", [])

    prices = []
    profits = []
    violations = []
    fines = []
    cumulative_fines = []

    total_fines = 0.0

    for step in steps:
        # Skip non-step entries (headers, summaries)
        if step.get("type") != "step":
            continue

        step_num = step.get("step", 0)

        # Skip step 0 if it exists
        if step_num == 0:
            continue

        prices.append({"x": step_num, "y": step.get("market_price", 0)})

        # Average profit across firms
        step_profits = step.get("profits", [])
        avg_profit = sum(step_profits) / len(step_profits) if step_profits else 0
        profits.append({"x": step_num, "y": avg_profit})

        regulator_flags = step.get("regulator_flags", {})

        # Calculate fines for this step
        step_fines = sum(regulator_flags.get("fines_applied", []))
        fines.append({"x": step_num, "y": step_fines})

        # Track cumulative fines
        total_fines += step_fines
        cumulative_fines.append({"x": step_num, "y": total_fines})

        # Violations now show fines amount (for combined chart)
        violations.append({"x": step_num, "y": step_fines})

    return {
        "prices": prices,
        "profits": profits,
        "violations": violations,
        "fines": fines,
        "cumulative_fines": cumulative_fines,
    }


@app.route("/")
def index() -> str:
    """Render the main dashboard interface.
    
    Serves the HTML template for the interactive monitoring dashboard.
    """
    return render_template("dashboard.html")


@app.route("/api/data")
def get_data() -> Union[Tuple[Response, int], Response]:
    """API endpoint for dashboard data.
    
    Returns combined metrics and time series data from the most recent
    experiment run. Used by the dashboard for real-time visualization.
    """
    data = load_latest_experiment()
    if not data:
        logger.warning("No experiment data available for dashboard")
        return jsonify({"error": "No experiment data found"}), 404

    metrics = calculate_metrics(data)
    time_series = extract_time_series(data)

    return jsonify({"metrics": metrics, "time_series": time_series})


@app.route("/api/experiments")
def list_experiments() -> Response:
    """List all available experiment log files.
    
    Returns metadata for the 10 most recent experiment runs,
    including filenames and modification timestamps.
    """
    log_dir = Path(__file__).parent.parent / "logs"
    if not log_dir.exists():
        return jsonify([])

    log_files = sorted(log_dir.glob("*.jsonl"), key=os.path.getmtime, reverse=True)
    experiments = []

    for log_file in log_files[:10]:
        experiments.append(
            {
                "name": log_file.name,
                "path": str(log_file),
                "modified": datetime.fromtimestamp(
                    os.path.getmtime(log_file)
                ).isoformat(),
            }
        )

    return jsonify(experiments)


def run_experiment_background(steps: int, firms: List[str]) -> None:
    """Run experiment in background thread.
    
    Executes the experiment runner script as a subprocess to avoid
    blocking the Flask server. Updates global status for polling.
    """
    global running_experiments
    try:
        running_experiments["status"] = "running"
        running_experiments["progress"] = 0
        logger.info(f"Starting experiment: {steps} steps, firms={firms}")

        project_root = Path(__file__).parent.parent

        # Use venv Python if available, otherwise fall back to current interpreter
        venv_python = project_root / "venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else sys.executable

        # Generate random seed for each experiment
        seed = random.randint(1, 999999)

        cmd = [
            python_cmd,
            "scripts/run_experiment.py",
            "--steps",
            str(steps),
            "--firms",
            ",".join(firms),
            "--seed",
            str(seed),
        ]

        # Set up environment with project root in PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root.resolve())

        subprocess.run(
            cmd, cwd=project_root, env=env, check=True, capture_output=True, text=True
        )

        running_experiments["status"] = "completed"
        running_experiments["progress"] = 100
        logger.info("Experiment completed successfully")
    except subprocess.CalledProcessError as e:
        running_experiments["status"] = "error"
        running_experiments["progress"] = None
        running_experiments["error_message"] = f"Process failed: {e.stderr[:200]}"
        logger.error(f"Experiment failed: {e.stderr}")
    except Exception as e:
        running_experiments["status"] = "error"
        running_experiments["progress"] = None
        running_experiments["error_message"] = str(e)
        logger.error(f"Experiment error: {e}")


@app.route("/api/experiment/run", methods=["POST"])
def run_experiment() -> Response:
    """Start a new experiment in the background.
    
    Initiates a new simulation run with default parameters. The experiment
    runs asynchronously while status can be polled via /api/experiment/status.
    """
    global running_experiments

    if running_experiments["status"] == "running":
        logger.warning("Attempted to start experiment while one is already running")
        return jsonify({"error": "Experiment already running"}), 400

    # Reset state
    running_experiments = {"status": "idle", "progress": None, "error_message": None}

    # Default experiment parameters
    steps = 50
    firms = ["random", "titfortat"]

    # Start experiment in background thread
    thread = threading.Thread(target=run_experiment_background, args=(steps, firms))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "steps": steps, "firms": firms})


@app.route("/api/experiment/status")
def experiment_status() -> Response:
    """Get status of running experiment.
    
    Returns the current state of any running experiment, including
    progress percentage and error messages if applicable.
    """
    return jsonify(running_experiments)


@app.route("/api/healthz")
def health_check() -> Response:
    """Health check endpoint.
    
    Returns service health status for monitoring and load balancing.
    """
    return jsonify({"status": "healthy", "service": "regulator-dashboard"})


if __name__ == "__main__":
    logger.info("Starting Regulator Dashboard on http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
