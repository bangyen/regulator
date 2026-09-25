"""Tests for batch experiments and their summaries."""

import math

import pandas as pd
import pytest
from click.testing import CliRunner

from regulator.cli import main
from regulator.experiments.batch import (
    METRICS,
    format_summary,
    run_batch,
    summarize,
)


@pytest.fixture(scope="module")
def runs() -> pd.DataFrame:
    return run_batch(
        [["bestresponse", "bestresponse"], ["stealth", "stealth"]],
        ["none", "rule_based"],
        seeds=[0, 1, 2],
        steps=20,
        jobs=1,
    )


def test_one_row_per_combination(runs: pd.DataFrame) -> None:
    assert len(runs) == 2 * 2 * 3
    assert set(METRICS) <= set(runs.columns)
    assert runs["economically_valid"].all()


def test_no_regulator_means_no_fines(runs: pd.DataFrame) -> None:
    assert (runs.loc[runs["regulator"] == "none", "total_fines"] == 0).all()


def test_colluders_price_above_competitors(runs: pd.DataFrame) -> None:
    mean = runs.groupby("firms")["mean_price"].mean()

    assert mean["stealth,stealth"] > mean["bestresponse,bestresponse"] + 5


def test_summary_mean_and_t_interval(runs: pd.DataFrame) -> None:
    summary = summarize(runs)
    cell = summary[
        (summary["firms"] == "stealth,stealth") & (summary["regulator"] == "none")
    ].iloc[0]
    prices = runs[(runs["firms"] == "stealth,stealth") & (runs["regulator"] == "none")][
        "mean_price"
    ]

    assert cell["n"] == 3
    assert cell["mean_price"] == pytest.approx(prices.mean())
    # t(0.975, df=2) = 4.303
    expected = 4.303 * prices.std(ddof=1) / math.sqrt(3)
    assert cell["mean_price_ci"] == pytest.approx(expected, rel=1e-3)


def test_single_seed_has_no_interval() -> None:
    runs = run_batch([["random"]], ["none"], seeds=[0], steps=10, jobs=1)

    assert math.isnan(summarize(runs)["mean_price_ci"].iloc[0])
    assert "±" not in format_summary(summarize(runs)).splitlines()[-1]


def test_format_summary_table(runs: pd.DataFrame) -> None:
    table = format_summary(summarize(runs))

    assert table.splitlines()[0].startswith("| Firms | Regulator | n |")
    assert len(table.splitlines()) == 2 + 4
    assert "±" in table


def test_parallel_matches_serial() -> None:
    kwargs = {"lineups": [["random", "random"]], "regulator_configs": ["none"]}
    serial = run_batch(**kwargs, seeds=[0, 1], steps=10, jobs=1)
    parallel = run_batch(**kwargs, seeds=[0, 1], steps=10, jobs=2)

    pd.testing.assert_frame_equal(serial, parallel)


def test_cli_batch(tmp_path) -> None:
    csv = tmp_path / "runs.csv"
    result = CliRunner().invoke(
        main,
        [
            "batch",
            "--lineups",
            "random,random",
            "--regulators",
            "none",
            "--seeds",
            "2",
            "--steps",
            "10",
            "--jobs",
            "1",
            "--csv",
            str(csv),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "| random,random | none | 2 |" in result.output
    assert len(pd.read_csv(csv)) == 2
