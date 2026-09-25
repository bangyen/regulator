"""
Batch experiments: many seeds x firm line-ups x regulator configs.

Runs every combination, collects one row of outcome metrics per run, and
summarises each (line-up, regulator) cell with a mean and a 95% confidence
interval across seeds.
"""

import itertools
import logging
import tempfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from regulator.experiments.experiment_runner import run_experiment
from regulator.experiments.q_learning import collusion_index

METRICS = [
    "mean_price",
    "collusion_index",
    "consumer_surplus",
    "producer_surplus",
    "total_welfare",
    "total_fines",
    "violations",
]


def _run_one(job: tuple[tuple[str, ...], str, int, int, str]) -> dict[str, Any]:
    firms, regulator_config, seed, steps, log_dir = job
    logging.getLogger("regulator").setLevel(logging.WARNING)
    results = run_experiment(
        firms=list(firms),
        steps=steps,
        regulator_config=regulator_config,
        seed=seed,
        log_dir=log_dir,
        episode_id=f"batch_{'-'.join(firms)}_{regulator_config}_{seed}",
        verbose=False,
    )
    data = results["episode_data"]
    welfare = results["welfare_metrics"]
    mean_price = float(np.mean(data["episode_prices"]))
    return {
        "firms": ",".join(firms),
        "regulator": regulator_config,
        "seed": seed,
        "mean_price": mean_price,
        "collusion_index": collusion_index(mean_price),
        "consumer_surplus": welfare["consumer_surplus"],
        "producer_surplus": welfare["producer_surplus"],
        "total_welfare": welfare["total_welfare"],
        "total_fines": float(data["total_fines"]),
        "violations": int(sum(data["violations"].values())),
        "economically_valid": results["economic_validation"]["valid"],
    }


def run_batch(
    lineups: Sequence[Sequence[str]],
    regulator_configs: Sequence[str],
    seeds: Sequence[int],
    steps: int = 100,
    log_dir: str | None = None,
    jobs: int | None = None,
) -> pd.DataFrame:
    """
    Run every (line-up, regulator, seed) combination.

    Args:
        lineups: Firm line-ups, e.g. [["random", "random"], ["stealth", "stealth"]]
        regulator_configs: Regulator configs (see REGULATOR_CONFIGS)
        seeds: Seeds; each combination runs once per seed
        steps: Steps per episode
        log_dir: Where to keep episode logs (a temporary directory if None)
        jobs: Worker processes (default: CPU count; 1 runs in-process)

    Returns:
        One row per run with the columns in METRICS plus firms, regulator,
        seed and economically_valid
    """
    with tempfile.TemporaryDirectory() as tmp:
        directory = log_dir or tmp
        Path(directory).mkdir(parents=True, exist_ok=True)
        combos = [
            (tuple(lineup), regulator, seed, steps, directory)
            for lineup, regulator, seed in itertools.product(
                lineups, regulator_configs, seeds
            )
        ]
        if jobs == 1:
            rows = [_run_one(job) for job in combos]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                rows = list(pool.map(_run_one, combos))
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """
    Mean and confidence-interval half-width per (firms, regulator) cell.

    Uses a t-interval across seeds; cells with one seed get a NaN interval.

    Returns:
        Columns ``<metric>`` and ``<metric>_ci`` for each metric, plus ``n``
    """
    rows = []
    for (firms, regulator), group in runs.groupby(["firms", "regulator"], sort=False):
        row: dict[str, Any] = {"firms": firms, "regulator": regulator, "n": len(group)}
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[metric] = float(values.mean())
            if len(values) > 1:
                sem = stats.sem(values)
                t = stats.t.ppf((1 + confidence) / 2, len(values) - 1)
                row[f"{metric}_ci"] = float(t * sem)
            else:
                row[f"{metric}_ci"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def format_summary(summary: pd.DataFrame) -> str:
    """Markdown table of the summary, each metric shown as mean ± CI."""
    columns = [
        ("mean_price", "Mean price", 2),
        ("collusion_index", "Collusion idx", 2),
        ("consumer_surplus", "Consumer surplus", 0),
        ("producer_surplus", "Producer surplus", 0),
        ("total_welfare", "Total welfare", 0),
        ("total_fines", "Fines", 0),
    ]
    header = "| Firms | Regulator | n | " + " | ".join(c[1] for c in columns) + " |"
    rule = "|" + "---|" * (3 + len(columns))
    lines = [header, rule]
    for _, row in summary.iterrows():
        cells = []
        for metric, _, digits in columns:
            ci = row[f"{metric}_ci"]
            text = f"{row[metric]:.{digits}f}"
            if not np.isnan(ci):
                text += f" ± {ci:.{digits}f}"
            cells.append(text)
        lines.append(
            f"| {row['firms']} | {row['regulator']} | {row['n']} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)
