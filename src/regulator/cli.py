"""
Command-line interface for the Regulator package.

This module provides CLI commands for running experiments, training models,
and analyzing results.
"""

import logging
import sys

import click
from dotenv import load_dotenv

# Import from the package
from regulator.experiments.experiment_runner import REGULATOR_CONFIGS, run_experiment


@click.group()
@click.version_option(package_name="regulator")
def main() -> None:
    """
    Regulator: can label-free screens detect algorithmic collusion?

    Simulate oligopoly markets with competing, colluding and learning firms,
    and measure how well collusion screens and regulators tell them apart.
    """
    # Load environment variables
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@main.command()
@click.option("--steps", default=100, help="Number of steps to run")
@click.option(
    "--firms", default="random,tit_for_tat", help="Comma-separated agent types"
)
@click.option(
    "--regulator",
    default="rule_based",
    type=click.Choice([*REGULATOR_CONFIGS, "disabled"], case_sensitive=False),
    help="Regulator configuration",
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
    episode_id: str | None,
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
@click.option("--episodes", default=40, help="Test episodes per population")
@click.option("--steps", default=100, help="Steps per episode")
@click.option("--alpha", default=0.05, help="Target false-positive rate")
@click.option("--q-pairs", default=4, help="Q-learning pairs per discount factor")
@click.option("--q-train-steps", default=150_000, help="Training periods per pair")
@click.option("--seed", default=0, help="Base seed")
def screen(
    episodes: int,
    steps: int,
    alpha: float,
    q_pairs: int,
    q_train_steps: int,
    seed: int,
) -> None:
    """Calibrate label-free screens on competition; measure FPR and detection."""
    from regulator.experiments.screening import format_study, run_study

    logging.getLogger("regulator").setLevel(logging.WARNING)
    click.echo("Training Q-learners and running the screening study...")
    result = run_study(
        episodes=episodes,
        steps=steps,
        alpha=alpha,
        q_pairs=q_pairs,
        q_train_steps=q_train_steps,
        seed=seed,
    )
    click.echo(format_study(result))


@main.command()
@click.option(
    "--lineups",
    default="random,random;bestresponse,bestresponse;stealth,stealth",
    help="Firm line-ups: semicolon-separated, each comma-separated",
)
@click.option(
    "--regulators",
    default="none,rule_based,ml",
    help="Comma-separated regulator configs",
)
@click.option("--seeds", default=10, help="Seeds per combination (0..N-1)")
@click.option("--steps", default=100, help="Steps per episode")
@click.option("--jobs", type=int, help="Worker processes (default: CPU count)")
@click.option("--csv", "csv_path", type=click.Path(), help="Write per-run rows here")
@click.option("--log-dir", help="Keep episode logs here (default: discard)")
def batch(
    lineups: str,
    regulators: str,
    seeds: int,
    steps: int,
    jobs: int | None,
    csv_path: str | None,
    log_dir: str | None,
) -> None:
    """Run seeds x line-ups x regulators and compare outcomes with 95% CIs."""
    from regulator.experiments.batch import format_summary, run_batch, summarize

    lineup_list = [
        [firm.strip() for firm in lineup.split(",")]
        for lineup in lineups.split(";")
        if lineup.strip()
    ]
    regulator_list = [r.strip() for r in regulators.split(",") if r.strip()]
    click.echo(
        f"Running {len(lineup_list) * len(regulator_list) * seeds} episodes "
        f"({len(lineup_list)} line-ups x {len(regulator_list)} regulators x "
        f"{seeds} seeds)..."
    )
    runs = run_batch(
        lineup_list, regulator_list, range(seeds), steps, log_dir=log_dir, jobs=jobs
    )
    if csv_path:
        runs.to_csv(csv_path, index=False)
        click.echo(f"Per-run results written to {csv_path}")
    click.echo(format_summary(summarize(runs)))
    invalid = int((~runs["economically_valid"]).sum())
    if invalid:
        click.echo(f"\n{invalid} run(s) failed economic validation", err=True)


@main.command()
@click.option("--port", default=5000, help="Port for the dashboard")
@click.option("--host", default="127.0.0.1", help="Host address for the dashboard")
def dashboard(port: int, host: str) -> None:
    """Launch the Flask dashboard."""
    click.echo(f"Launching dashboard on {host}:{port}...")

    import os
    import subprocess

    try:
        env = os.environ.copy()
        env["FLASK_APP"] = "dashboard.main"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "flask",
                "run",
                "--host",
                host,
                "--port",
                str(port),
            ],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Dashboard failed to start: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n👋 Dashboard stopped.")


if __name__ == "__main__":
    main()
