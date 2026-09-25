"""Smoke tests for the reproducible detection benchmark script."""

import pytest

from regulator.detectors.llm_detector import LLMDetector
from scripts.benchmark import (
    HARD_MESSAGES,
    SCENARIOS,
    benchmark_llm,
    benchmark_ml,
    template_messages,
)

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


def test_benchmark_llm_templates() -> None:
    messages = template_messages(40, seed=0)
    result = benchmark_llm(messages, LLMDetector(model_type="stubbed", seed=0))

    assert set(result) == METRICS
    assert result["n_test"] == 40


def test_hard_messages_are_balanced() -> None:
    labels = [label for _, label in HARD_MESSAGES]

    assert labels.count(1) == labels.count(0)


def test_benchmark_llm_real_model_path_reports_usage() -> None:
    """The --llm-model path, with the OpenAI client mocked."""
    import os
    from unittest.mock import Mock, patch

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = '{"is_collusive": true, "confidence": 0.8}'
    response.usage = Mock(prompt_tokens=100, completion_tokens=20)

    with (
        patch("regulator.detectors.llm_detector.openai") as mock_openai,
        patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True),
    ):
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = response
        detector = LLMDetector(model_type="llm", model_name="test-model")
        result = benchmark_llm(HARD_MESSAGES[:4], detector)

    assert result["fallbacks"] == 0
    assert result["prompt_tokens"] == 400
    assert result["completion_tokens"] == 80
    assert result["mean_latency_s"] >= 0
