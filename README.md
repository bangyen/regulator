# Regulator: Market Competition & Collusion Detection

A Python package for simulating market competition and detecting collusive behavior using machine learning.

## Features

- **Market Simulation**: CartelEnv for oligopolistic price competition
- **Agent Framework**: Multiple agent types (random, tit-for-tat, best response)
- **Regulator System**: Real-time monitoring and penalty application
- **ML Collusion Detection**: Machine learning-based collusion detection
- **Comprehensive Logging**: Structured episode logging and analysis
- **Modern tooling**: Black, Ruff, MyPy, Pytest

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd your-package-name
   make init
   ```

2. **Development workflow**:
   ```bash
   make fmt    # Format code
   make lint   # Lint code
   make type   # Type check
   make test   # Run tests
   make all    # Run all checks
   ```

## ML Collusion Detector

Train and use machine learning models to detect collusive behavior:

```bash
# Train ML detector with demo episodes
python scripts/train_ml_detector.py --n-episodes 50

# Train with existing log files
python scripts/train_ml_detector.py --existing-logs logs/

# Use LightGBM model
python scripts/train_ml_detector.py --model-type lightgbm --n-episodes 50
```

The ML detector extracts 26 features from episode logs and achieves ≥0.8 AUROC on synthetic datasets.

## Project Structure

```
├── src/
│   ├── agents/          # Market agents (firms, regulator)
│   ├── cartel/          # Market environment
│   ├── detectors/       # ML collusion detection
│   └── episode_logging/ # Structured logging
├── tests/               # Test files
├── scripts/             # Utility scripts and demos
├── logs/                # Episode logs (gitignored)
├── ml_detector_output/  # ML training outputs (gitignored)
├── Makefile            # Development commands
└── pyproject.toml      # Package configuration
```

## Customization

1. **Update package info** in `pyproject.toml`:
   - Change `name`, `description`, `authors`
   - Update repository URLs

2. **Add your code** to `src/`

3. **Write tests** in `tests/`

## License

MIT License - see LICENSE file for details.
