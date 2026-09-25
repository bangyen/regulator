"""
Unit tests for experiment runner functionality.

This module tests the core experiment execution functions including
agent creation, regulator creation, welfare calculations, and experiment execution.
"""

import json
from pathlib import Path
from typing import Any

import pytest

# Import the actual classes that experiment_runner uses
from regulator.agents.enhanced_regulator import EnhancedRegulator
from regulator.agents.firm_agents import BaseAgent
from regulator.agents.ml_regulator import MLRegulator
from regulator.agents.regulator import Regulator
from regulator.cartel.cartel_env import CartelEnv
from regulator.experiments.experiment_runner import (
    REGULATOR_CONFIGS,
    calculate_welfare_metrics,
    create_agent,
    create_regulator,
    print_experiment_summary,
    run_experiment,
    validate_episode,
)


class TestCreateAgent:
    """Test agent creation functionality."""

    def test_create_random_agent(self) -> None:
        """Test creating a random agent."""
        agent = create_agent("random", agent_id=0, seed=42)
        assert isinstance(agent, BaseAgent)
        assert agent.agent_id == 0

    def test_create_bestresponse_agent(self) -> None:
        """Test creating a best response agent."""
        agent = create_agent("bestresponse", agent_id=1, seed=42)
        assert isinstance(agent, BaseAgent)
        assert agent.agent_id == 1

    def test_create_titfortat_agent(self) -> None:
        """Test creating a tit-for-tat agent."""
        agent = create_agent("titfortat", agent_id=2, seed=42)
        assert isinstance(agent, BaseAgent)
        assert agent.agent_id == 2

    def test_create_agent_case_insensitive(self) -> None:
        """Test that agent type is case insensitive."""
        agent1 = create_agent("RANDOM", agent_id=0, seed=42)
        agent2 = create_agent("Random", agent_id=1, seed=42)
        agent3 = create_agent("random", agent_id=2, seed=42)

        assert isinstance(agent1, BaseAgent)
        assert isinstance(agent2, BaseAgent)
        assert isinstance(agent3, BaseAgent)

    def test_create_agent_with_seed(self) -> None:
        """Test creating agents with different seeds."""
        agent1 = create_agent("random", agent_id=0, seed=42)
        agent2 = create_agent("random", agent_id=1, seed=123)

        assert agent1.agent_id == 0
        assert agent2.agent_id == 1
        # Agents should be different instances
        assert agent1 is not agent2

    def test_create_agent_invalid_type(self) -> None:
        """Test creating agent with invalid type raises error."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("invalid_type", agent_id=0, seed=42)

    def test_create_agent_empty_type(self) -> None:
        """Test creating agent with empty type raises error."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("", agent_id=0, seed=42)


class TestCreateRegulator:
    """Test regulator creation functionality."""

    def test_rule_based(self) -> None:
        regulator = create_regulator("rule_based", seed=42)
        assert type(regulator) is Regulator

    def test_ml(self) -> None:
        assert isinstance(create_regulator("ml", seed=42), MLRegulator)

    def test_enhanced(self) -> None:
        assert isinstance(create_regulator("enhanced", seed=42), EnhancedRegulator)

    @pytest.mark.parametrize("config", ["none", "disabled", "NONE"])
    def test_none_means_no_regulator(self, config: str) -> None:
        assert create_regulator(config, seed=42) is None

    def test_case_insensitive(self) -> None:
        assert isinstance(create_regulator("ML", seed=42), MLRegulator)

    def test_every_listed_config_is_accepted(self) -> None:
        for config in REGULATOR_CONFIGS:
            create_regulator(config, seed=0)

    def test_invalid_config(self) -> None:
        with pytest.raises(ValueError, match="Unknown regulator config"):
            create_regulator("invalid_config", seed=42)


class TestCalculateWelfareMetrics:
    """Test welfare metrics calculation functionality."""

    ZERO = {
        "consumer_surplus": 0.0,
        "producer_surplus": 0.0,
        "total_welfare": 0.0,
        "deadweight_loss": 0.0,
    }

    def test_empty_data(self) -> None:
        env = CartelEnv(n_firms=2, seed=42)
        assert calculate_welfare_metrics({}, env) == self.ZERO
        assert (
            calculate_welfare_metrics(
                {"episode_prices": [], "episode_profits": []}, env
            )
            == self.ZERO
        )

    def test_single_step(self) -> None:
        """CS uses the linear demand curve at the average market price."""
        env = CartelEnv(n_firms=2, seed=42)  # a=100, b=-1, mc=10
        data = {"episode_prices": [[20.0, 30.0]], "episode_profits": [[100.0, 120.0]]}

        metrics = calculate_welfare_metrics(data, env)

        # p = 25, q per firm = (100 - 25) / 2 = 37.5, CS = 0.5 * 75 * 37.5
        assert metrics["consumer_surplus"] == pytest.approx(1406.25)
        assert metrics["producer_surplus"] == pytest.approx(220.0)
        assert metrics["total_welfare"] == pytest.approx(1626.25)

    def test_deadweight_loss_positive_for_high_prices(self) -> None:
        env = CartelEnv(n_firms=2, seed=42)
        data = {"episode_prices": [[90.0, 90.0]], "episode_profits": [[10.0, 10.0]]}

        assert calculate_welfare_metrics(data, env)["deadweight_loss"] > 0

    def test_quantities_never_negative(self) -> None:
        env = CartelEnv(n_firms=2, seed=42)
        data = {"episode_prices": [[150.0, 150.0]], "episode_profits": [[0.0, 0.0]]}

        assert calculate_welfare_metrics(data, env)["consumer_surplus"] == 0.0


class TestPrintExperimentSummary:
    """Test experiment summary printing functionality."""

    WELFARE = {
        "consumer_surplus": 1000.0,
        "producer_surplus": 220.0,
        "total_welfare": 1220.0,
        "deadweight_loss": 50.0,
    }

    def test_basic(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = {
            "episode_id": "test_episode",
            "log_file": "/path/to/log.jsonl",
            "episode_summary": {
                "agent_types": ["random", "titfortat"],
                "avg_prices": [20.0, 25.0],
                "total_profits": [100.0, 120.0],
            },
        }
        episode_data = {
            "total_steps": 1,
            "total_fines": 12.5,
            "violations": {"parallel": 1, "structural_break": 0},
        }

        print_experiment_summary(results, episode_data, self.WELFARE)

        out = capsys.readouterr().out
        assert "test_episode" in out
        assert "random, titfortat" in out
        assert "Total Fines Applied: 12.50" in out
        assert "/path/to/log.jsonl" in out

    def test_empty_data(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_experiment_summary({"episode_id": "empty"}, {}, self.WELFARE)

        assert "empty" in capsys.readouterr().out


class TestRunExperiment:
    """Run small real experiments end to end."""

    def _run(self, tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
        params: dict[str, Any] = {
            "firms": ["random", "random"],
            "steps": 5,
            "regulator_config": "rule_based",
            "seed": 42,
            "log_dir": str(tmp_path),
            "episode_id": "test_experiment",
        }
        params.update(kwargs)
        return run_experiment(**params)

    def test_basic(self, tmp_path: Path) -> None:
        results = self._run(tmp_path)

        for key in (
            "episode_id",
            "episode_data",
            "experiment_params",
            "welfare_metrics",
            "log_file",
        ):
            assert key in results
        assert results["episode_id"] == "test_experiment"
        assert results["episode_data"]["total_steps"] == 5
        assert Path(results["log_file"]).exists()
        assert results["experiment_params"] == {
            "firms": ["random", "random"],
            "steps": 5,
            "regulator_config": "rule_based",
            "seed": 42,
            "env_params": results["experiment_params"]["env_params"],
            "chat_monitoring": False,
        }

    def test_env_params_are_applied(self, tmp_path: Path) -> None:
        results = self._run(
            tmp_path,
            env_params={"marginal_cost": 15.0, "demand_intercept": 120.0},
        )

        env_params = results["experiment_params"]["env_params"]
        assert env_params["marginal_cost"] == 15.0
        assert env_params["demand_intercept"] == 120.0
        assert results["episode_summary"]["environment_params"]["marginal_cost"] == 15.0

    def test_auto_episode_id(self, tmp_path: Path) -> None:
        results = self._run(tmp_path, episode_id=None)

        assert results["episode_id"].startswith("experiment_")

    def test_results_are_json_native(self, tmp_path: Path) -> None:
        results = self._run(tmp_path)
        results.pop("logger")

        json.dumps(results)

    def test_three_firms(self, tmp_path: Path) -> None:
        results = self._run(tmp_path, firms=["random", "bestresponse", "titfortat"])

        assert results["experiment_params"]["env_params"]["n_firms"] == 3

    def test_no_regulator_means_no_fines(self, tmp_path: Path) -> None:
        results = self._run(
            tmp_path, firms=["stealth", "stealth"], steps=20, regulator_config="none"
        )

        assert results["episode_data"]["total_fines"] == 0.0
        assert results["episode_data"]["violations"] == {
            "parallel": 0,
            "structural_break": 0,
        }

    @pytest.mark.parametrize("config", ["rule_based", "enhanced"])
    def test_regulators_fine_colluders(self, tmp_path: Path, config: str) -> None:
        results = self._run(
            tmp_path, firms=["stealth", "stealth"], steps=30, regulator_config=config
        )

        assert results["episode_data"]["total_fines"] > 0

    def test_ml_regulator_runs(self, tmp_path: Path) -> None:
        results = self._run(tmp_path, steps=15, regulator_config="ml")

        assert results["experiment_params"]["regulator_config"] == "ml"

    def test_logs_progress(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO", logger="regulator.experiments.experiment_runner"):
            self._run(tmp_path, episode_id="test_print")

        assert "test_print" in caplog.text
        assert "random" in caplog.text
        assert "rule_based" in caplog.text


class TestEconomicValidation:
    """run_experiment checks the logged episode with EconomicValidator."""

    def test_valid_episode_passes(self, tmp_path: Path) -> None:
        results = run_experiment(
            firms=["random", "titfortat"], steps=10, seed=1, log_dir=str(tmp_path)
        )

        assert results["economic_validation"] == {
            "valid": True,
            "n_issues": 0,
            "issues": [],
        }

    def test_tampered_log_is_flagged(self, tmp_path: Path) -> None:
        results = run_experiment(
            firms=["random", "random"], steps=5, seed=1, log_dir=str(tmp_path)
        )
        log_file = Path(results["log_file"])
        lines = log_file.read_text().splitlines()
        tampered = []
        for line in lines:
            record = json.loads(line)
            if record.get("type") == "step":
                record["market_price"] += 10.0  # no longer the mean price
            tampered.append(json.dumps(record))
        log_file.write_text("\n".join(tampered) + "\n")

        validation = validate_episode(str(log_file), CartelEnv(n_firms=2))

        assert validation["valid"] is False
        assert validation["n_issues"] >= 5
        assert "Market price" in validation["issues"][0]

    def test_summary_reports_validation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_experiment(firms=["random"], steps=5, seed=1, log_dir=str(tmp_path))

        assert "ECONOMIC VALIDATION: passed" in capsys.readouterr().out


class TestChatMonitoring:
    """Chat firms talk each step; the chat regulator fines collusive senders."""

    def test_chat_agent_types(self) -> None:
        from regulator.agents.chat_firm import CollusiveChatAgent, CompetitiveChatAgent

        assert isinstance(create_agent("chat_colluder", 0, seed=0), CollusiveChatAgent)
        assert isinstance(
            create_agent("chat-competitor", 1, seed=0), CompetitiveChatAgent
        )

    def test_collusive_senders_are_fined(self, tmp_path: Path) -> None:
        results = run_experiment(
            firms=["chatcolluder", "chatcompetitor"],
            steps=40,
            regulator_config="none",
            seed=3,
            log_dir=str(tmp_path),
            chat_monitoring=True,
        )

        data = results["episode_data"]
        assert data["messages_sent"] > 0
        assert data["message_violations"] > 0
        assert data["chat_fines"] == data["message_violations"] * 25.0
        assert data["total_fines"] == data["chat_fines"]

        steps = [
            json.loads(line)
            for line in Path(results["log_file"]).read_text().splitlines()
            if '"type": "step"' in line
        ]
        fined = [s for s in steps if any(s["regulator_flags"]["chat_fines"])]
        assert fined
        for step in fined:
            senders = {m["sender_id"] for m in step["additional_info"]["messages"]}
            fined_firms = {
                i for i, f in enumerate(step["regulator_flags"]["chat_fines"]) if f
            }
            assert fined_firms <= senders
            # The competitor's templates are never collusive
            assert 1 not in fined_firms

    def test_messages_without_monitoring_are_not_fined(self, tmp_path: Path) -> None:
        results = run_experiment(
            firms=["chatcolluder", "chatcompetitor"],
            steps=20,
            regulator_config="none",
            seed=3,
            log_dir=str(tmp_path),
        )

        assert results["episode_data"]["messages_sent"] > 0
        assert results["episode_data"]["chat_fines"] == 0.0
