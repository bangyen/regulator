"""Tests for the dashboard's Run Experiment API."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

import dashboard.main as dash
from dashboard.main import DEFAULT_RUN, app, parse_run_request


@pytest.fixture
def client() -> Iterator[Any]:
    dash.running_experiments.update(
        {"status": "idle", "progress": None, "error_message": None}
    )
    with app.test_client() as client:
        yield client
    dash.running_experiments["status"] = "idle"


class TestParseRunRequest:
    def test_defaults(self) -> None:
        config = parse_run_request(None)

        assert config["firms"] == DEFAULT_RUN["firms"]
        assert config["regulator"] == "rule_based"
        assert config["steps"] == 50
        assert isinstance(config["seed"], int)

    def test_explicit_values(self) -> None:
        config = parse_run_request(
            {
                "firms": ["stealth", "stealth", "bestresponse"],
                "regulator": "ml",
                "steps": 200,
                "seed": 7,
            }
        )

        assert config == {
            "firms": ["stealth", "stealth", "bestresponse"],
            "regulator": "ml",
            "steps": 200,
            "seed": 7,
        }

    def test_null_seed_means_random(self) -> None:
        assert isinstance(parse_run_request({"seed": None})["seed"], int)

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ({"firms": []}, "firms must be"),
            ({"firms": "random"}, "firms must be"),
            ({"firms": ["random"] * 6}, "firms must be"),
            ({"firms": ["random", "cartel"]}, "Unknown agent type"),
            ({"regulator": "fbi"}, "Unknown regulator"),
            ({"steps": 2}, "between 5"),
            ({"steps": 10_000}, "between 5"),
            ({"steps": "50"}, "integer"),
            ({"steps": True}, "integer"),
            ({"seed": -1}, "non-negative"),
            ({"seed": 1.5}, "non-negative"),
        ],
    )
    def test_invalid(self, payload: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_run_request(payload)


class TestRunEndpoints:
    def test_options(self, client: Any) -> None:
        options = client.get("/api/options").get_json()

        assert "stealth" in options["agent_types"]
        assert options["regulator_configs"] == [
            "rule_based",
            "ml",
            "enhanced",
            "none",
        ]
        assert options["defaults"] == DEFAULT_RUN

    def test_run_starts_with_config(self, client: Any) -> None:
        with patch.object(dash, "run_experiment_background") as background:
            response = client.post(
                "/api/experiment/run",
                json={"firms": ["stealth", "random"], "regulator": "none", "seed": 3},
            )

        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "started"
        assert body["regulator"] == "none"
        background.assert_called_once()
        assert background.call_args.args[0]["firms"] == ["stealth", "random"]
        assert dash.running_experiments["status"] == "running"

    def test_run_without_body_uses_defaults(self, client: Any) -> None:
        with patch.object(dash, "run_experiment_background"):
            response = client.post("/api/experiment/run")

        assert response.get_json()["firms"] == DEFAULT_RUN["firms"]

    def test_invalid_config_is_rejected(self, client: Any) -> None:
        with patch.object(dash, "run_experiment_background") as background:
            response = client.post("/api/experiment/run", json={"steps": 1})

        assert response.status_code == 400
        assert "steps" in response.get_json()["error"]
        background.assert_not_called()
        assert dash.running_experiments["status"] == "idle"

    def test_second_run_is_refused_while_running(self, client: Any) -> None:
        with patch.object(dash, "run_experiment_background"):
            client.post("/api/experiment/run")
            response = client.post("/api/experiment/run")

        assert response.status_code == 409

    def test_background_passes_regulator_and_seed(self, tmp_path: Any) -> None:
        config = {"firms": ["random"], "regulator": "ml", "steps": 5, "seed": 9}
        with (
            patch.dict("os.environ", {"REGULATOR_LOG_DIR": str(tmp_path)}),
            patch.object(dash.subprocess, "run") as run,
        ):
            dash.run_experiment_background(config)

        cmd = run.call_args.args[0]
        assert cmd[cmd.index("--regulator") + 1] == "ml"
        assert cmd[cmd.index("--seed") + 1] == "9"
        assert cmd[cmd.index("--log-dir") + 1] == str(tmp_path)
        assert dash.running_experiments["status"] == "completed"
