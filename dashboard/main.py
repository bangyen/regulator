"""Flask application for real-time regulator monitoring dashboard."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from flask import Flask, Response, jsonify, render_template

app = Flask(__name__)


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
    violations = sum(1 for s in steps if s.get("regulator_flags", {}).get("flagged"))
    total_fines = sum(
        sum(s.get("regulator_flags", {}).get("fines_applied", []))
        for s in steps
        if "regulator_flags" in s
    )

    risk_scores = [
        s.get("regulator_flags", {}).get("risk_score", 0)
        for s in steps
        if s.get("regulator_flags", {}).get("risk_score") is not None
    ]

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

        is_violation = step.get("regulator_flags", {}).get("flagged", False)
        violations.append({"x": step_num, "y": 1 if is_violation else 0})

        risk_score = step.get("regulator_flags", {}).get("risk_score")
        if risk_score is not None:
            risk_scores.append({"x": step_num, "y": risk_score})

        step_fines = sum(step.get("regulator_flags", {}).get("fines_applied", []))
        fines.append({"x": step_num, "y": step_fines})

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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
