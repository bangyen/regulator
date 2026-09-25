# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `scripts/benchmark.py`: seeded benchmark that reports accuracy, precision,
  recall, F1 and ROC AUC for the ML and LLM detectors.
- `stealth` agent type in `create_agent`; `StealthCollusiveAgent` is exported
  from `regulator.agents`.
- `llm` and `dashboard` optional extras.
- CI: native pytest matrix (Python 3.10–3.12), coverage floor, notebook
  execution via nbmake, Dependabot.

### Changed
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

## [0.1.0] - 2026-08-27

- Initial release.
