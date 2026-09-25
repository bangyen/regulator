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

Compare line-ups and regulators across seeds (mean ± 95% CI per cell):

```bash
regulator batch --lineups "bestresponse,bestresponse;stealth,stealth" \
    --regulators none,rule_based,enhanced --seeds 10 --csv runs.csv
```

Monitor chat between firms (keyword stub by default, `--llm-model` for OpenAI):

```bash
regulator experiment --firms chatcolluder,chatcompetitor --chat
```

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
- **Browser tests** (dashboard front end): `pip install -e ".[e2e]" && playwright install chromium && pytest -m e2e`

## Results

Reproduce with `python scripts/benchmark.py` (seed 42, ~45 s on 4 cores):

| Detector | Accuracy | Precision | Recall | F1 | ROC AUC | Test size |
|----------|----------|-----------|--------|----|---------|-----------|
| ML (logistic) — baseline | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 60 |
| ML (logistic) — noisy | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 60 |
| ML (logistic) — mixed | 98.3% | 100.0% | 96.7% | 98.3% | 0.999 | 60 |
| ML (logistic) — adaptive | 31.7% | 35.1% | 43.3% | 38.8% | 0.371 | 60 |
| ML (logistic) — tacit (Q-learning) | 98.0% | 100.0% | 96.0% | 98.0% | 0.998 | 100 |
| LLM (stub) — templates | 87.0% | 80.6% | 97.5% | 88.2% | 0.878 | 400 |
| LLM (stub) — hard | 55.0% | 57.1% | 40.0% | 47.1% | 0.580 | 20 |

Labels come from the firms' strategies, not from a heuristic on prices:

- **baseline** — fixed-price colluders vs random / best-response / tit-for-tat
  firms. The classes differ mainly in price variance, so this is easy.
- **noisy** — both sides have comparable noise; colluders price 0–6 above the
  best-response level (~40), with the markup drawn per episode.
- **mixed** — colluding pairs vs a colluder paired with a noisy competitor.
- **adaptive** — `AdaptiveAgent` pairs that differ only in
  `collusion_tendency` (0.9 vs 0.1). The detector is at chance: over 50 steps
  that setting doesn't produce distinguishable prices.
- **tacit** — Q-learning pairs trained for 150k periods, patient (discount
  0.95) vs myopic (discount 0), after Calvano et al. (2020). Nobody tells them
  to collude, yet patient pairs settle at a mean price of 47.0 (collusion
  index 0.47; Nash 40, monopoly 55) vs 44.3 for myopic pairs. Train and test
  use different trained pairs.

Over 5 seeds, the strategic-interaction features (lead-lag, punishment and
return, rigidity) raised mean accuracy from 95.7% to 99.3% on noisy and from
93.0% to 97.0% on mixed; adaptive stayed at chance and tacit was unchanged
within noise (96.0% vs 94.2%).

The LLM rows use the keyword stub. "templates" are the chat agents' own
messages; "hard" is a hand-written set of paraphrased and indirect messages
where the stub is near chance. To evaluate a real model (paid API calls):

```bash
OPENAI_API_KEY=... python scripts/benchmark.py --llm-model gpt-4o-mini
```

This adds rows for that model and reports fallbacks, mean latency and token
usage.

## Features

- **Market** — Oligopoly with linear demand and logit market shares; with
  default parameters the one-shot Nash price is 40 and the joint-profit
  maximum is 55.
- **Firms** — Random, best-response, tit-for-tat, explicit and stealth
  colluders, chat firms, and Q-learning firms that can learn to collude
  tacitly (Calvano et al. 2020).
- **Regulators** — Rule-based (parallel pricing, structural breaks),
  enhanced (graduated penalties), ML (rules plus an anomaly detector and a
  classifier trained offline on labeled simulations) and chat monitoring.
- **Detection** — Episode-level classifier over price-level and
  strategic-interaction features (lead-lag, punishment and return), plus
  message-level LLM analysis.
- **Batch experiments** — Seeds × line-ups × regulators with confidence
  intervals for prices, welfare and fines.
- **Economic validation** — Every experiment's log is checked for
  accounting consistency.
- **Dashboard** — Live metrics and charts; choose firms, regulator, steps,
  seed and chat monitoring for new runs.
- **Leniency programs** — Whistleblower dynamics and evidence-based reporting.

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

- ✅ 585 unit/integration tests (~93% coverage, 88% floor) + 12 browser tests
- ✅ Reproducible seeds for experiments
- ✅ Seeded benchmark: `python scripts/benchmark.py`
- ✅ Demo notebook executed in CI

## References

- [Artificial Intelligence, Algorithmic Pricing, and Collusion](https://www.aeaweb.org/articles?id=10.1257/aer.20190623) - Calvano et al. (AER 2020); the Q-learning setup used here
- [Algorithms, Machine Learning, and Collusion](https://academic.oup.com/jcle/article-abstract/14/4/568/5514023) - Comprehensive analysis of self-learning algorithms and collusive outcomes
- [Deep learning for detecting bid rigging](https://arxiv.org/abs/2104.11142) - CNN-based approach for flagging cartel participants using pairwise bidding interactions
- [Algorithmic Collusion: A Critical Review](https://arxiv.org/abs/2110.04740) - Critical assessment of pricing algorithms and collusion potential

## License

This project is licensed under the [MIT License](LICENSE).
