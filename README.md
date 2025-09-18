# Regulator: Market Competition & Collusion Detection

A Python package for simulating market competition and detecting collusive behavior using machine learning.

## Features

- **Market Simulation**: CartelEnv for oligopolistic price competition
- **Agent Framework**: Multiple agent types (random, tit-for-tat, best response)
- **Regulator System**: Real-time monitoring and penalty application
- **ML Collusion Detection**: Machine learning-based collusion detection
- **LLM Detection**: OpenAI-powered natural language collusion detection
- **Comprehensive Logging**: Structured episode logging and analysis
- **Interactive Dashboard**: Streamlit-based visualization and analysis
- **CLI Interface**: Command-line tools for experiments and training
- **Modern tooling**: Black, Ruff, MyPy, Pytest

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone https://github.com/bangyen/regulator.git
   cd regulator
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

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Use the CLI**:
   ```bash
   # Run an experiment
   regulator experiment --n-episodes 10 --n-firms 3
   
   # Train ML detector
   regulator train --n-episodes 50 --model-type lightgbm
   
   # Run single episode
   regulator episode --firms "random,tit_for_tat" --steps 50
   
   # Launch dashboard
   regulator dashboard
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

## LLM Collusion Detector

Use OpenAI's GPT models to detect collusive behavior in natural language communications:

```bash
# Use LLM detector in experiments
python scripts/chat_demo.py --llm-model gpt-4o-mini

# Test LLM detection
python -c "
from src.detectors.llm_detector import LLMDetector
detector = LLMDetector(model_type='llm')
result = detector.classify_message('Let\\'s coordinate our pricing', 0, 1, 1)
print(result)
"
```

The LLM detector analyzes messages for collusive intent and provides confidence scores and reasoning.

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
