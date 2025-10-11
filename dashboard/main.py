"""Flask application for real-time regulator monitoring dashboard."""

import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flask import Flask, Response, jsonify, render_template

app = Flask(__name__)

# Track running experiments
running_experiments = {"status": "idle", "progress": None, "error_message": None}


def load_latest_experiment() -> Optional[Dict[str, Any]]:
    """Load most recent experiment data from logs directory."""
    log_dir = Path(__file__).parent.parent / "logs"
    if not log_dir.exists():
        return None

    log_files = sorted(log_dir.glob("*.jsonl"), key=os.path.getmtime, reverse=True)
    if not log_files:
        return None

    steps = []
    with open(log_files[0], "r") as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))

    return {"steps": steps, "file": str(log_files[0])}


def calculate_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate aggregate metrics from experiment data."""
    steps = data.get("steps", [])
    if not steps:
        return {}

    prices = [s.get("market_price", 0) for s in steps if "market_price" in s]

    violations = sum(
        1
        for s in steps
        if s.get("regulator_flags", {}).get("parallel_violation", False)
        or s.get("regulator_flags", {}).get("structural_break_violation", False)
        or s.get("regulator_flags", {}).get("flagged", False)
    )

    total_fines = sum(
        sum(s.get("regulator_flags", {}).get("fines_applied", []))
        for s in steps
        if "regulator_flags" in s
    )

    # Calculate risk scores from fines or use existing risk_score
    risk_scores = []
    for s in steps:
        if "regulator_flags" in s:
            flags = s["regulator_flags"]
            # Use risk_score if available, otherwise calculate from fines
            if "risk_score" in flags and flags["risk_score"] is not None:
                risk_scores.append(flags["risk_score"])
            else:
                fines = sum(flags.get("fines_applied", []))
                risk_scores.append(min(fines / 150.0, 1.0))

    return {
        "total_steps": len(steps),
        "avg_price": round(np.mean(prices), 2) if prices else 0,
        "price_volatility": round(np.std(prices), 2) if prices else 0,
        "total_violations": violations,
        "total_fines": round(total_fines, 2),
        "avg_risk_score": round(np.mean(risk_scores), 2) if risk_scores else 0,
        "current_risk": round(risk_scores[-1], 2) if risk_scores else 0,
    }


def extract_time_series(data: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Extract time series data for visualization."""
    steps = data.get("steps", [])

    prices = []
    violations = []
    risk_scores = []
    fines = []

    for step in steps:
        step_num = step.get("step", 0)
        prices.append({"x": step_num, "y": step.get("market_price", 0)})

        regulator_flags = step.get("regulator_flags", {})
        is_violation = (
            regulator_flags.get("parallel_violation", False)
            or regulator_flags.get("structural_break_violation", False)
            or regulator_flags.get("flagged", False)
        )
        violations.append({"x": step_num, "y": 1 if is_violation else 0})

        # Calculate fines for this step
        step_fines = sum(regulator_flags.get("fines_applied", []))
        fines.append({"x": step_num, "y": step_fines})

        # Use risk_score if available, otherwise calculate from fines
        if (
            "risk_score" in regulator_flags
            and regulator_flags["risk_score"] is not None
        ):
            risk_score = regulator_flags["risk_score"]
        else:
            risk_score = min(step_fines / 150.0, 1.0) if step_fines > 0 else 0.0
        risk_scores.append({"x": step_num, "y": risk_score})

    return {
        "prices": prices,
        "violations": violations,
        "risk_scores": risk_scores,
        "fines": fines,
    }


@app.route("/")
def index() -> str:
    """Render dashboard template."""
    return render_template("dashboard.html")


@app.route("/api/data")
def get_data() -> Union[Tuple[Response, int], Response]:
    """API endpoint for dashboard data."""
    data = load_latest_experiment()
    if not data:
        return jsonify({"error": "No experiment data found"}), 404

    metrics = calculate_metrics(data)
    time_series = extract_time_series(data)

    return jsonify({"metrics": metrics, "time_series": time_series})


@app.route("/api/experiments")
def list_experiments() -> Response:
    """List all available experiment log files."""
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
    """Run experiment in background thread."""
    global running_experiments
    try:
        running_experiments["status"] = "running"
        running_experiments["progress"] = 0

        project_root = Path(__file__).parent.parent

        # Use sys.executable to get current Python interpreter
        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--steps",
            str(steps),
            "--firms",
            ",".join(firms),
        ]

        # Set up environment with project root in PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)

        result = subprocess.run(
            cmd, cwd=project_root, env=env, check=True, capture_output=True, text=True
        )

        running_experiments["status"] = "completed"
        running_experiments["progress"] = 100
    except subprocess.CalledProcessError as e:
        running_experiments["status"] = "error"
        running_experiments["progress"] = None
        running_experiments["error_message"] = f"Process failed: {e.stderr[:200]}"
        print(f"Experiment error: {e.stderr}")
    except Exception as e:
        running_experiments["status"] = "error"
        running_experiments["progress"] = None
        running_experiments["error_message"] = str(e)
        print(f"Experiment error: {e}")


@app.route("/api/experiment/run", methods=["POST"])
def run_experiment() -> Response:
    """Start a new experiment in the background."""
    global running_experiments

    if running_experiments["status"] == "running":
        return jsonify({"error": "Experiment already running"}), 400

    # Reset state
    running_experiments = {"status": "idle", "progress": None, "error_message": None}

    # Default experiment parameters
    steps = 50
    firms = ["random", "tit_for_tat"]

    # Start experiment in background thread
    thread = threading.Thread(target=run_experiment_background, args=(steps, firms))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "steps": steps, "firms": firms})


@app.route("/api/experiment/status")
def experiment_status() -> Response:
    """Get status of running experiment."""
    return jsonify(running_experiments)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
