# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `scripts/benchmark.py`: seeded benchmark that reports accuracy, precision,
  recall, F1 and ROC AUC per scenario (baseline, noisy, mixed, adaptive) for
  the ML detector, and on template and hard messages for the LLM detector.
  `--llm-model` evaluates a real OpenAI model with latency and token usage.
- `enhanced` regulator config (`EnhancedRegulator`).
- `LLMDetector(model_name=...)`; honours `OPENAI_MODEL`,
  `OPENAI_TEMPERATURE`, `OPENAI_MAX_TOKENS`.
- Playwright browser tests for the dashboard (`pytest -m e2e`, `e2e` extra)
  and a CI job that runs them.
- `REGULATOR_LOG_DIR` sets the dashboard's log directory.
- `stealth` agent type in `create_agent`; `StealthCollusiveAgent` is exported
  from `regulator.agents`.
- `llm` and `dashboard` optional extras.
- CI: native pytest matrix (Python 3.10–3.12), coverage floor, notebook
  execution via nbmake, Dependabot.

### Fixed
- `create_regulator` ignored its config and always returned a default
  `Regulator`; `ml`, `enhanced` and `none` now do what they say.
- `EnhancedRegulator` never applied its graduated fines.
- `LLMDetector` read `OPENAI_KEY` instead of `OPENAI_API_KEY`, and its
  fallback to the stub crashed on API errors.
- Limited price visibility hid prices far more often than configured with
  more than two firms.
- ML-regulator fines were missing from `total_fines`.
- Experiment welfare metrics used a fixed 10% deadweight loss and read a key
  the logs don't contain.

### Changed
- `scripts/run_experiment.py` is a thin wrapper over
  `regulator.experiments.experiment_runner`.
- `__version__` / `regulator --version` come from package metadata.
- Chart.js is vendored, so the dashboard works offline.
- The dashboard binds to `127.0.0.1` with debug off by default
  (`DASHBOARD_HOST`, `DASHBOARD_PORT`, `FLASK_DEBUG` override).
- `openai` and `flask` are no longer core dependencies.
- Library diagnostics use `logging` instead of `print`.
- Ruff now enforces import sorting, bugbear, pyupgrade and simplify rules;
  `ruff format` replaces black.
- README results table now shows benchmark output instead of unverified
  figures.

### Removed
- Python 3.9 support.
- Unused `SimplifiedCartelEnv`.

## [0.1.0] - 2026-08-27

- Initial release.
