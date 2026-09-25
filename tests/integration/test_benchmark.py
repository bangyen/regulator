"""Smoke tests for the reproducible detection benchmark script."""

import pytest

from scripts.benchmark import SCENARIOS, benchmark_ml

METRICS = {"accuracy", "precision", "recall", "f1", "roc_auc", "n_test"}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_benchmark_ml_scenarios_report_all_metrics(scenario: str) -> None:
    result = benchmark_ml(
        n_episodes=20, steps=20, model_type="logistic", seed=0, scenario=scenario
    )

    assert set(result) == METRICS
    assert result["n_test"] == 6
    assert 0.0 <= result["accuracy"] <= 1.0


def test_benchmark_is_deterministic() -> None:
    kwargs = {"n_episodes": 20, "steps": 20, "model_type": "logistic", "seed": 1}

    assert benchmark_ml(**kwargs, scenario="noisy") == benchmark_ml(
        **kwargs, scenario="noisy"
    )


def test_benchmark_tacit_smoke() -> None:
    from scripts.benchmark import benchmark_tacit

    result = benchmark_tacit(
        n_pairs=2,
        train_steps=2_000,
        episodes_per_pair=3,
        steps=15,
        model_type="logistic",
        seed=0,
    )

    assert result["n_test"] == 6  # one patient + one myopic pair held out
    assert 38.0 < result["mean_price_patient"] < 57.0
    assert "collusion_index_myopic" in result
