# Regulator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bangyen/regulator/blob/main/regulator_demo.ipynb)
[![CI](https://github.com/bangyen/regulator/actions/workflows/ci.yml/badge.svg)](https://github.com/bangyen/regulator/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/bangyen/regulator)](LICENSE)

**Market collusion simulation and detection: ML episode classifier, LLM chat monitoring, real-time dashboard, and economic validation**

<p align="center">
  <img src="docs/price_trajectories.png" alt="Price trajectories demo" width="600">
</p>

## Quickstart

Clone the repo and run the demo:

```bash
git clone https://github.com/bangyen/regulator.git
cd regulator
pip install -e ".[dev,ml,llm,dashboard]"   # ".[ml]" alone is enough to just run the demo
pytest   # optional: run tests
python scripts/run_experiment.py --firms "random,titfortat" --steps 100
```

Or open in Colab: [Colab Notebook](https://colab.research.google.com/github/bangyen/regulator/blob/main/regulator_demo.ipynb).

For real-time monitoring, run the dashboard:

```bash
python dashboard/main.py
# Or use the CLI: regulator dashboard
# Visit http://localhost:5000
```

The dashboard binds to `127.0.0.1` with debug off. Override with
`DASHBOARD_HOST`, `DASHBOARD_PORT` and `FLASK_DEBUG=1`. Never enable debug on a
network-reachable host, because the Werkzeug debugger allows arbitrary code
execution.

## Development Workflow

This project uses `uv` for dependency management and `just` as a task runner.

- **Initialize environment**: `just init`
- **Format code**: `just fmt`
- **Lint**: `just lint`
- **Type check**: `just type`
- **Run tests**: `just test`
- **Run all checks**: `just all`
- **Benchmark detectors**: `python scripts/benchmark.py`


## Results

Reproduce with `python scripts/benchmark.py` (seed 42, ~5 s):

| Detector | Accuracy | Precision | Recall | F1 | ROC AUC | Test size |
|----------|----------|-----------|--------|----|---------|-----------|
| ML (logistic) | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 60 |
| LLM (stubbed) | 86.2% | 81.1% | 94.5% | 87.3% | 0.875 | 400 |

Ground truth comes from the firms' strategies (collusive/stealth agents vs
random/best-response/tit-for-tat) for the ML detector, and from the chat
agents' message templates for the LLM detector. The simulated classes are
easy to separate, so the ML score shows the pipeline works end to end. It is
not an estimate of real-world performance. The LLM row uses the keyword-based
stub; set `OPENAI_API_KEY` and construct `LLMDetector` with any other `model_type` to call OpenAI.

## Features

- **Real-Time Dashboard** — Professional monitoring interface with live metrics, charts, and violation tracking.
- **ML+LLM Detection** — Episode-level ML classifier (logistic / LightGBM) plus message-level LLM analysis.
- **Economic Validation** — Built-in consistency checks and market dynamics validation.
- **Chat Integration** — Natural language communication monitoring with OpenAI-powered analysis.
- **Enhanced Regulator** — Graduated penalties, continuous risk scores, and market-aware monitoring.
- **Leniency Programs** — Strategic whistleblower dynamics and evidence-based reporting.

## Repo Structure

```plaintext
regulator/
├── regulator_demo.ipynb  # Colab notebook
├── dashboard/            # Flask dashboard (run with dashboard/main.py)
├── scripts/              # Example run scripts
├── tests/                # Unit/integration tests
├── docs/                 # Images for README
└── src/                  # Core implementation
    └── regulator/        # Main package
        ├── agents/       # Market agents, regulators, chat/adaptive/stealth firms
        ├── cartel/       # Market environment
        ├── detectors/    # ML + LLM collusion detectors
        ├── episode_logging/ # Logger & episode runner
        ├── experiments/  # Experiment logic
        ├── monitoring/   # Enhanced monitoring dashboard
        └── cli.py        # CLI entry point
```

## Validation

- ✅ 540 tests, ~91% coverage, enforced floor of 88% (`pytest --cov=src/regulator`)
- ✅ Reproducible seeds for experiments
- ✅ Seeded benchmark: `python scripts/benchmark.py`
- ✅ Demo notebook executed in CI

## Roadmap

- [Harder detection benchmark](https://github.com/bangyen/regulator/issues/1) (tacit collusion, noisy colluders, mixed line-ups)
- [Evaluate the LLM detector against a real model](https://github.com/bangyen/regulator/issues/2)
- [Fix `create_regulator` ignoring its config](https://github.com/bangyen/regulator/issues/3)
- [Remove duplicated experiment and environment code](https://github.com/bangyen/regulator/issues/4)
- [Publish to PyPI on tagged releases](https://github.com/bangyen/regulator/issues/5)
- [Browser tests for the dashboard](https://github.com/bangyen/regulator/issues/6)

## References

- [Algorithms, Machine Learning, and Collusion](https://academic.oup.com/jcle/article-abstract/14/4/568/5514023) - Comprehensive analysis of self-learning algorithms and collusive outcomes
- [Deep learning for detecting bid rigging](https://arxiv.org/abs/2104.11142) - CNN-based approach for flagging cartel participants using pairwise bidding interactions
- [Algorithmic Collusion: A Critical Review](https://arxiv.org/abs/2110.04740) - Critical assessment of pricing algorithms and collusion potential

## License

This project is licensed under the [MIT License](LICENSE).
