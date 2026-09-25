"""Browser tests for the dashboard front end (dashboard.js + template)."""

import os
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conftest import FIXTURE_STEPS

pytest.importorskip("playwright")

pytestmark = pytest.mark.e2e

from playwright.sync_api import expect  # noqa: E402


def _chart_points(page: Any, canvas_id: str) -> int:
    return page.evaluate(
        "id => Chart.getChart(id).data.datasets[0].data.length", canvas_id
    )


def test_metric_tiles_show_fixture_values(
    dashboard: Any, expected_metrics: dict[str, Any]
) -> None:
    expect(dashboard.locator("#avg-price")).to_have_text(
        f"{expected_metrics['avg_price']:.2f}"
    )
    expect(dashboard.locator("#total-fines")).to_have_text(
        f"{expected_metrics['total_fines']:.2f}"
    )
    expect(dashboard.locator("#violations")).not_to_have_text("—")


def test_main_chart_plots_every_step(dashboard: Any) -> None:
    dashboard.wait_for_function(
        f"() => Chart.getChart('main-chart').data.datasets[0].data.length"
        f" === {FIXTURE_STEPS}"
    )
    assert _chart_points(dashboard, "violations-chart") > 0


def test_metric_toggle_switches_main_chart(dashboard: Any) -> None:
    dashboard.wait_for_function(
        "() => Chart.getChart('main-chart').data.datasets[0].data.length > 0"
    )
    before = dashboard.evaluate("Chart.getChart('main-chart').data.datasets[0].data")

    dashboard.click("[data-metric=profit]")

    expect(dashboard.locator("[data-metric=profit]")).to_have_class("toggle-btn active")
    after = dashboard.evaluate("Chart.getChart('main-chart').data.datasets[0].data")
    assert after != before


def test_navigation_between_views(dashboard: Any) -> None:
    dashboard.click("[data-view=analytics]")

    expect(dashboard.locator("#analytics-view")).to_be_visible()
    expect(dashboard.locator("#overview-view")).to_be_hidden()
    expect(dashboard.locator(".page-title")).to_have_text("Analytics")
    assert _chart_points(dashboard, "cumulative-chart") > 0


def test_enforcement_table_shows_recent_steps(dashboard: Any) -> None:
    dashboard.click("[data-view=enforcement]")

    rows = dashboard.locator("#activity-table tr")
    expect(rows).to_have_count(15)
    expect(rows.first.locator("td").first).to_have_text(str(FIXTURE_STEPS))


def test_experiments_view_lists_log_files(dashboard: Any) -> None:
    dashboard.click("[data-view=experiments]")

    expect(dashboard.locator(".experiment-name")).to_contain_text(
        ["fixture_episode.jsonl"]
    )


def test_empty_log_dir_shows_error_state(
    page: Any, server_url: str, tmp_path: Path
) -> None:
    previous = os.environ["REGULATOR_LOG_DIR"]
    os.environ["REGULATOR_LOG_DIR"] = str(tmp_path)
    try:
        page.route("https://fonts.g*/**", lambda route: route.abort())
        page.goto(server_url)

        expect(page.locator("#sidebar-status")).to_have_text("Error")
        expect(page.locator("#avg-price")).to_have_text("—")
    finally:
        os.environ["REGULATOR_LOG_DIR"] = previous


def test_run_controls_are_populated(dashboard: Any) -> None:
    expect(dashboard.locator("#firm-1")).to_have_value("random")
    expect(dashboard.locator("#firm-2")).to_have_value("titfortat")
    expect(dashboard.locator("#firm-3")).to_have_value("")
    expect(dashboard.locator("#regulator-select")).to_have_value("rule_based")
    expect(dashboard.locator("#steps-input")).to_have_value("50")
    options = dashboard.locator("#regulator-select option").all_inner_texts()
    assert options == ["Rule-based", "ML", "Enhanced", "None"]


def test_run_uses_selected_configuration(dashboard: Any, log_dir: Path) -> None:
    before = set(log_dir.glob("*.jsonl"))
    dashboard.select_option("#firm-1", "stealth")
    dashboard.select_option("#firm-2", "stealth")
    dashboard.select_option("#firm-3", "bestresponse")
    dashboard.select_option("#regulator-select", "enhanced")
    dashboard.fill("#steps-input", "12")
    dashboard.fill("#seed-input", "5")

    with dashboard.expect_request("**/api/experiment/run") as request_info:
        dashboard.click("#run-btn")

    assert request_info.value.post_data_json == {
        "firms": ["stealth", "stealth", "bestresponse"],
        "regulator": "enhanced",
        "steps": 12,
        "seed": 5,
    }
    expect(dashboard.locator("#run-btn")).to_be_enabled(timeout=60_000)
    (new_log,) = set(log_dir.glob("*.jsonl")) - before
    steps = [line for line in new_log.read_text().splitlines() if '"step"' in line]
    assert len([s for s in steps if '"type": "step"' in s]) == 12
    assert '"n_firms": 3' in new_log.read_text().splitlines()[0]


def test_invalid_steps_show_server_error(
    page: Any, server_url: str, log_dir: Path
) -> None:
    page.route("https://fonts.g*/**", lambda route: route.abort())
    page.goto(server_url)
    expect(page.locator("#regulator-select")).to_have_value("rule_based")
    page.fill("#steps-input", "2")
    messages: list[str] = []
    page.on("dialog", lambda dialog: (messages.append(dialog.message), dialog.accept()))

    page.click("#run-btn")

    expect(page.locator("#sidebar-status")).to_have_text("Error")
    assert messages and "steps must be between 5" in messages[0]


def test_run_experiment_button(dashboard: Any, log_dir: Path) -> None:
    before = set(log_dir.glob("*.jsonl"))

    dashboard.click("#run-btn")

    expect(dashboard.locator("#run-btn")).to_be_disabled()
    expect(dashboard.locator("#sidebar-status")).to_have_text("Running")
    # The button re-enables once the status poll reports completion
    expect(dashboard.locator("#run-btn")).to_be_enabled(timeout=60_000)
    expect(dashboard.locator("#sidebar-status")).to_have_text("Live")
    assert len(set(log_dir.glob("*.jsonl")) - before) == 1
