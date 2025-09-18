"""
Command-line interface for the Regulator package.

This module provides CLI commands for running experiments, training models,
and analyzing results.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_experiment import main as run_experiment_main
from scripts.train_ml_detector import main as train_ml_main
from scripts.run_episode import run_episode


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """
    Regulator: Market Competition & Collusion Detection

    A Python package for simulating market competition and detecting
    collusive behavior using machine learning.
    """
    # Load environment variables
    load_dotenv()


@main.command()
@click.option("--steps", default=100, help="Number of steps to run")
@click.option(
    "--firms", default="random,tit_for_tat", help="Comma-separated agent types"
)
@click.option(
    "--regulator",
    default="rule_based",
    help="Regulator configuration (ml, rule_based, none)",
)
@click.option("--seed", default=42, help="Random seed")
@click.option("--log-dir", default="logs", help="Output directory for logs")
@click.option("--episode-id", help="Custom episode ID (auto-generated if not provided)")
def experiment(
    steps: int,
    firms: str,
    regulator: str,
    seed: int,
    log_dir: str,
    episode_id: Optional[str],
) -> None:
    """Run a single episode experiment."""
    click.echo("Running experiment...")

    # Set up arguments for run_experiment
    sys.argv = [
        "run_experiment.py",
        "--firms",
        firms,
        "--steps",
        str(steps),
        "--regulator",
        regulator,
        "--seed",
        str(seed),
        "--log-dir",
        log_dir,
    ]

    if episode_id:
        sys.argv.extend(["--episode-id", episode_id])

    try:
        run_experiment_main()
        click.echo("✅ Experiment completed successfully!")
    except Exception as e:
        click.echo(f"❌ Experiment failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--n-episodes", default=50, help="Number of episodes to train on")
@click.option(
    "--model-type", default="logistic", help="Model type (logistic, lightgbm)"
)
@click.option("--existing-logs", help="Path to existing log files")
@click.option("--output-dir", default="ml_detector_output", help="Output directory")
def train(
    n_episodes: int,
    model_type: str,
    existing_logs: Optional[str],
    output_dir: str,
) -> None:
    """Train the ML collusion detector."""
    click.echo(f"Training ML detector with {model_type}...")

    # Set up arguments for train_ml_detector
    sys.argv = [
        "train_ml_detector.py",
        "--n-episodes",
        str(n_episodes),
        "--model-type",
        model_type,
        "--output-dir",
        output_dir,
    ]

    if existing_logs:
        sys.argv.extend(["--existing-logs", existing_logs])

    try:
        train_ml_main()
        click.echo("✅ Training completed successfully!")
    except Exception as e:
        click.echo(f"❌ Training failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--firms", default="random,tit_for_tat", help="Comma-separated agent types"
)
@click.option("--steps", default=50, help="Number of steps")
@click.option("--n-firms", help="Number of firms (auto-detected if not specified)")
@click.option("--seed", default=42, help="Random seed")
@click.option("--log-dir", default="logs", help="Log directory")
def episode(
    firms: str,
    steps: int,
    n_firms: Optional[int],
    seed: int,
    log_dir: str,
) -> None:
    """Run a single episode."""
    click.echo("Running single episode...")

    # Parse agent types
    agent_types = [agent.strip() for agent in firms.split(",")]

    try:
        result = run_episode(
            firms=agent_types,
            steps=steps,
            n_firms=n_firms,
            seed=seed,
            log_dir=log_dir,
        )
        click.echo("✅ Episode completed successfully!")
        if isinstance(result, dict):
            click.echo(f"Episode ID: {result.get('episode_id', 'unknown')}")
        else:
            click.echo("Episode completed successfully!")
    except Exception as e:
        click.echo(f"❌ Episode failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--port", default=8501, help="Port for the dashboard")
def dashboard(port: int) -> None:
    """Launch the Streamlit dashboard."""
    click.echo(f"Launching dashboard on port {port}...")

    import subprocess
    import sys

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "dashboard/app.py",
                "--server.port",
                str(port),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Dashboard failed to start: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n👋 Dashboard stopped.")


if __name__ == "__main__":
    main()
