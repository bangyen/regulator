# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Screening study** — the project's focus. `regulator.screens` (variance,
  rigidity, markup, parallel pricing, punish-and-return screens; static
  Nash/monopoly benchmarks; calibration on a competitive null) and
  `regulator.experiments.screening` / `scripts/screen_study.py` /
  `regulator screen`, which measure false-positive and detection rates on
  competitive, cartel and Q-learning markets.
- `scripts/benchmark.py`: seeded benchmark that reports accuracy, precision,
  recall, F1 and ROC AUC per scenario (baseline, noisy, mixed, adaptive) for
  the supervised ML detector.
- `enhanced` regulator config (`EnhancedRegulator`).
- Playwright browser tests for the dashboard (`pytest -m e2e`, `e2e` extra)
  and a CI job that runs them.
- `REGULATOR_LOG_DIR` sets the dashboard's log directory.
- `QLearningAgent` and `regulator.experiments.q_learning` (tacit collusion,
  Calvano et al. 2020); a "tacit" benchmark scenario.
- `BaseAgent.observe_outcome(profit)`; runners pass each firm its profit
  after fines.
- Strategic-interaction detector features: normalized markup, lead-lag,
  rigidity, unprovoked cuts, rival response, punish-and-return.
- Batch experiments: `regulator batch`, `run_batch()`, `summarize()` with
  95% confidence intervals.
- Dashboard controls for firms, regulator, steps and seed
  (`/api/options`; `/api/experiment/run` takes a validated JSON body).
- Economic validation runs after every experiment.
- `stealth` agent type in `create_agent`; `StealthCollusiveAgent` is exported
  from `regulator.agents`.
- `dashboard` and `e2e` optional extras.
- CI: native pytest matrix (Python 3.10–3.12), coverage floor, notebook
  execution via nbmake, Dependabot.

### Fixed
- `create_regulator` ignored its config and always returned a default
  `Regulator`; `ml`, `enhanced` and `none` now do what they say.
- `EnhancedRegulator` never applied its graduated fines.
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
- `flask` is no longer a core dependency.
- Library diagnostics use `logging` instead of `print`.
- Ruff now enforces import sorting, bugbear, pyupgrade and simplify rules;
  `ruff format` replaces black.
- README results table now shows benchmark output instead of unverified
  figures.

### Removed
- Python 3.9 support.
- Unused `SimplifiedCartelEnv`.
- The LLM chat layer (chat firms, `LLMDetector`, `ChatRegulator`,
  `EpisodeLogger`, the `llm` extra and OpenAI settings): detection ran on
  template strings the simulation generated itself.
- The leniency program and whistleblower agents (not used by any
  experiment).
- The matplotlib `regulator.monitoring` module (unused).

## [0.1.0] - 2026-08-27

- Initial release.
