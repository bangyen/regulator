# Regulator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bangyen/regulator/blob/main/regulator_demo.ipynb)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![License](https://img.shields.io/github/license/bangyen/regulator)](LICENSE)

**Advanced Market Collusion Detection: 93% accuracy with ML+LLM detection, real-time monitoring, and economic validation**

<p align="center">
  <img src="docs/price_trajectories.png" alt="Price trajectories demo" width="600">
</p>

## Quickstart

Clone the repo and run the demo:

```bash
git clone https://github.com/bangyen/regulator.git
cd regulator
pip install -e .
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

## Development Workflow

This project uses `uv` for dependency management and `just` as a task runner.

- **Initialize environment**: `just init`
- **Format code**: `just fmt`
- **Lint**: `just lint`
- **Type check**: `just type`
- **Run tests**: `just test`
- **Run all checks**: `just all`


## Results

| Detection Method | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------------------|----------|-----------|--------|----------|---------|
| ML Detector | **93.0%** | **85.0%** | **97.1%** | **90.7%** | **98.0%** |
| LLM Detector | 69.0% | 49.2% | 96.7% | 65.2% | 91.8% |

## Features

- **Real-Time Dashboard** — Professional monitoring interface with live metrics, charts, and violation tracking.
- **ML+LLM Detection** — 93% accuracy with sub-millisecond processing for real-time monitoring.
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
        ├── agents/       # Market agents (firm_agents.py, leniency.py)
        ├── cartel/       # Market environment
        ├── detectors/    # Detection systems (under development)
        ├── episode_logging/ # Logger & episode runner
        ├── experiments/  # Experiment logic
        ├── monitoring/   # Placeholder for future metrics
        └── cli.py        # CLI entry point
```

## Validation

- ✅ Overall test coverage of 92% (`pytest`)
- ✅ Reproducible seeds for experiments
- ✅ Benchmark scripts included

## References

- [Algorithms, Machine Learning, and Collusion](https://academic.oup.com/jcle/article-abstract/14/4/568/5514023) - Comprehensive analysis of self-learning algorithms and collusive outcomes
- [Deep learning for detecting bid rigging](https://arxiv.org/abs/2104.11142) - CNN-based approach for flagging cartel participants using pairwise bidding interactions
- [Algorithmic Collusion: A Critical Review](https://arxiv.org/abs/2110.04740) - Critical assessment of pricing algorithms and collusion potential

## License

This project is licensed under the [MIT License](LICENSE).
