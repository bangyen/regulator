"""
Command-line interface for the Regulator package.

This module provides CLI commands for running experiments, training models,
and analyzing results.
"""

import sys
from typing import Optional

import click
from dotenv import load_dotenv

# Import from the package
from src.experiments.experiment_runner import run_experiment


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """
    Regulator: Market Competition & Collusion Detection

    A Python package for simulating market competition and detecting
    collusive behavior using LLM-based chat analysis and rule-based monitoring.
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

    try:
        # Parse firms string into list
        firms_list = [firm.strip() for firm in firms.split(",")]

        # Call the function directly
        run_experiment(
            firms=firms_list,
            steps=steps,
            regulator_config=regulator,
            seed=seed,
            log_dir=log_dir,
            episode_id=episode_id,
        )
        click.echo("✅ Experiment completed successfully!")
    except Exception as e:
        click.echo(f"❌ Experiment failed: {e}", err=True)
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
