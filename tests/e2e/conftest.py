"""Fixtures for browser tests of the Flask dashboard (run with `pytest -m e2e`)."""

import io
import os
import threading
from collections.abc import Iterator
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest
from werkzeug.serving import make_server

from dashboard.main import app, calculate_metrics, load_latest_experiment
from regulator.experiments.experiment_runner import run_experiment

FIXTURE_STEPS = 40


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args: dict[str, Any]) -> dict:
    """Allow a preinstalled Chromium via PLAYWRIGHT_CHROMIUM_EXECUTABLE."""
    executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    if executable:
        return {**browser_type_launch_args, "executable_path": executable}
    return browser_type_launch_args


@pytest.fixture(scope="session")
def log_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A log directory holding one seeded experiment, used by the server."""
    path = tmp_path_factory.mktemp("logs")
    with redirect_stdout(io.StringIO()):
        run_experiment(
            firms=["stealth", "stealth"],
            steps=FIXTURE_STEPS,
            regulator_config="rule_based",
            seed=7,
            log_dir=str(path),
            episode_id="fixture_episode",
        )
    previous = os.environ.get("REGULATOR_LOG_DIR")
    os.environ["REGULATOR_LOG_DIR"] = str(path)
    yield path
    if previous is None:
        os.environ.pop("REGULATOR_LOG_DIR", None)
    else:
        os.environ["REGULATOR_LOG_DIR"] = previous


@pytest.fixture(scope="session")
def expected_metrics(log_dir: Path) -> dict[str, Any]:
    data = load_latest_experiment()
    assert data is not None
    return calculate_metrics(data)


@pytest.fixture(scope="session")
def server_url(log_dir: Path) -> Iterator[str]:
    """Serve the dashboard on a free local port for the whole session."""
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def dashboard(page: Any, server_url: str) -> Iterator[Any]:
    """Open the dashboard; fail the test on any console error or page error."""
    errors: list[str] = []
    # Web fonts are the only external resource; don't depend on the network
    page.route("https://fonts.googleapis.com/**", lambda route: route.abort())
    page.route("https://fonts.gstatic.com/**", lambda route: route.abort())
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on(
        "console",
        lambda msg: (
            errors.append(msg.text)
            if msg.type == "error"
            and "fonts.g" not in msg.text
            and "ERR_FAILED" not in msg.text
            else None
        ),
    )
    page.goto(server_url)
    page.wait_for_function("() => window.Chart !== undefined")
    yield page
    assert errors == []
