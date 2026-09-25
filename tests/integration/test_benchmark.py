"""Smoke test for the reproducible detection benchmark script."""

from scripts.benchmark import benchmark_llm, benchmark_ml

METRICS = {"accuracy", "precision", "recall", "f1", "roc_auc", "n_test"}


def test_benchmark_ml_reports_all_metrics() -> None:
    result = benchmark_ml(n_episodes=20, steps=20, model_type="logistic", seed=0)

    assert set(result) == METRICS
    assert result["n_test"] == 6
    assert 0.0 <= result["accuracy"] <= 1.0


def test_benchmark_is_deterministic() -> None:
    first = benchmark_ml(n_episodes=20, steps=20, model_type="logistic", seed=1)
    second = benchmark_ml(n_episodes=20, steps=20, model_type="logistic", seed=1)

    assert first == second


def test_benchmark_llm_reports_all_metrics() -> None:
    result = benchmark_llm(n_messages=40, seed=0)

    assert set(result) == METRICS
    assert result["n_test"] == 40
