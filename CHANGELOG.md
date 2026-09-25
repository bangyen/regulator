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
- `QLearningAgent` and `regulator.experiments.q_learning` (tacit collusion,
  Calvano et al. 2020); a "tacit" benchmark scenario.
- `BaseAgent.observe_outcome(profit)`; runners pass each firm its profit
  after fines.
- Strategic-interaction detector features: normalized markup, lead-lag,
  rigidity, unprovoked cuts, rival response, punish-and-return.
- Chat monitoring in experiments: `chatcolluder` / `chatcompetitor` firms,
  `--chat` on the CLI and script, a dashboard checkbox.
- Batch experiments: `regulator batch`, `run_batch()`, `summarize()` with
  95% confidence intervals.
- Dashboard controls for firms, regulator, steps, seed and chat
  (`/api/options`; `/api/experiment/run` takes a validated JSON body).
- Economic validation runs after every experiment.
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
- `MLRegulator` trained its classifier on the rule-based detector's own
  verdicts and predicted on unscaled features. The classifier is now trained
  offline on simulations labeled by strategy; scaling is in pipelines.
- `CartelEnv` always applied learning-curve cost reductions and never reset
  them, driving marginal cost from 10 to about 2 within an episode. They are
  now opt-in (`use_learning_curves`) and reset each episode.
- `CollusiveAgent` (30) and the stealth agent (40) defaulted to prices at or
  below the market's Nash price; defaults are now 55 and 50.
- The detector's marginal-cost lookup read a header path that isn't
  written.
- `ChatRegulator` compared its threshold to a confidence score on a
  different scale and didn't attribute fines to senders.
- The concentration check flagged the cheapest firm winning most of the
  market.
- The dashboard showed 0 violations as a dash, and a double click could start
  two runs.
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
