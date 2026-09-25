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
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from regulator.experiments.experiment_runner import AGENT_TYPES, REGULATOR_CONFIGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Track running experiments (mutated in place; guarded by run_lock on start)
running_experiments: dict[str, Any] = {
    "status": "idle",
    "progress": None,
    "error_message": None,
}
run_lock = threading.Lock()


def get_log_dir() -> Path:
    """Directory holding experiment logs (override with REGULATOR_LOG_DIR)."""
    default = Path(__file__).parent.parent / "logs"
    return Path(os.environ.get("REGULATOR_LOG_DIR", default))


def load_latest_experiment() -> dict[str, Any] | None:
    """Load most recent experiment data from logs directory.

    Returns parsed experiment data including step-by-step metrics,
    or None if no experiment logs are found.
    """
    log_dir = get_log_dir()
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
        with open(log_files[0]) as f:
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


def calculate_metrics(data: dict[str, Any]) -> dict[str, Any]:
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


def extract_time_series(data: dict[str, Any]) -> dict[str, list[Any]]:
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
def get_data() -> tuple[Response, int] | Response:
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
    log_dir = get_log_dir()
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


MAX_STEPS = 1000
MAX_FIRMS = 5
DEFAULT_RUN: dict[str, Any] = {
    "firms": ["random", "titfortat"],
    "regulator": "rule_based",
    "steps": 50,
    "seed": None,
    "chat": False,
}


def parse_run_request(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Validate an experiment request, filling defaults for missing fields.

    Raises:
        ValueError: with a user-facing message when a field is invalid
    """
    payload = payload or {}
    config = {**DEFAULT_RUN, **{k: v for k, v in payload.items() if v is not None}}

    firms = config["firms"]
    if not isinstance(firms, list) or not 1 <= len(firms) <= MAX_FIRMS:
        raise ValueError(f"firms must be a list of 1-{MAX_FIRMS} agent types")
    for firm in firms:
        if (
            not isinstance(firm, str)
            or firm.lower().replace("_", "").replace("-", "") not in AGENT_TYPES
        ):
            raise ValueError(
                f"Unknown agent type {firm!r}; valid: {', '.join(AGENT_TYPES)}"
            )

    if config["regulator"] not in REGULATOR_CONFIGS:
        raise ValueError(
            f"Unknown regulator {config['regulator']!r}; "
            f"valid: {', '.join(REGULATOR_CONFIGS)}"
        )

    steps = config["steps"]
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise ValueError("steps must be an integer")
    if not 5 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be between 5 and {MAX_STEPS}")

    if not isinstance(config["chat"], bool):
        raise ValueError("chat must be true or false")

    seed = config["seed"]
    if seed is None:
        seed = random.randint(1, 999999)
    elif not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    return {
        "firms": firms,
        "regulator": config["regulator"],
        "steps": steps,
        "seed": seed,
        "chat": config["chat"],
    }


def run_experiment_background(config: dict[str, Any]) -> None:
    """Run experiment in background thread.

    Executes the experiment runner script as a subprocess to avoid
    blocking the Flask server. Updates global status for polling.

    Args:
        config: Validated request from parse_run_request
    """
    try:
        running_experiments["progress"] = 0
        logger.info(f"Starting experiment: {config}")

        project_root = Path(__file__).parent.parent

        # Use venv Python if available, otherwise fall back to current interpreter
        venv_python = project_root / "venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else sys.executable

        cmd = [
            python_cmd,
            "scripts/run_experiment.py",
            "--steps",
            str(config["steps"]),
            "--firms",
            ",".join(config["firms"]),
            "--regulator",
            config["regulator"],
            "--seed",
            str(config["seed"]),
            "--log-dir",
            str(get_log_dir()),
        ]
        if config.get("chat"):
            cmd.append("--chat")

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
        output = (e.stderr or e.stdout or "").strip()
        running_experiments["status"] = "error"
        running_experiments["progress"] = None
        running_experiments["error_message"] = f"Process failed: {output[-200:]}"
        logger.error(f"Experiment failed: {output}")
    except Exception as e:
        running_experiments["status"] = "error"
        running_experiments["progress"] = None
        running_experiments["error_message"] = str(e)
        logger.error(f"Experiment error: {e}")


@app.route("/api/options")
def run_options() -> Response:
    """Choices and defaults for the Run Experiment controls."""
    return jsonify(
        {
            "agent_types": list(AGENT_TYPES),
            "regulator_configs": list(REGULATOR_CONFIGS),
            "max_steps": MAX_STEPS,
            "max_firms": MAX_FIRMS,
            "defaults": DEFAULT_RUN,
        }
    )


@app.route("/api/experiment/run", methods=["POST"])
def run_experiment() -> tuple[Response, int] | Response:
    """Start a new experiment in the background.

    Accepts an optional JSON body with firms, regulator, steps and seed (see
    parse_run_request); missing fields use DEFAULT_RUN. Status can be polled
    via /api/experiment/status.
    """
    with run_lock:
        if running_experiments["status"] == "running":
            logger.warning("Attempted to start experiment while one is already running")
            return jsonify({"error": "Experiment already running"}), 409

        try:
            config = parse_run_request(request.get_json(silent=True))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Mark running before the thread starts so a second request is refused
        running_experiments.update(
            {"status": "running", "progress": None, "error_message": None}
        )

    thread = threading.Thread(target=run_experiment_background, args=(config,))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", **config})


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
    # Debug mode exposes the Werkzeug debugger (arbitrary code execution), so
    # it is opt-in and the server binds to localhost unless told otherwise.
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("DASHBOARD_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    logger.info(f"Starting Regulator Dashboard on http://{host}:{port}")
    app.run(debug=debug, host=host, port=port)
