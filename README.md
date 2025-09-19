# Regulator: Market Competition & Collusion Detection

A Python package for simulating market competition and detecting collusive behavior using LLM-based chat analysis and rule-based monitoring.

## Features

- **Market Simulation**: CartelEnv for oligopolistic price competition
- **Agent Framework**: Multiple agent types (random, tit-for-tat, best response)
- **Regulator System**: Real-time monitoring and penalty application
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
   regulator experiment --steps 100 --firms "random,tit_for_tat"
   
   # Launch dashboard
   regulator dashboard
   ```

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
│   ├── detectors/       # LLM collusion detection
│   └── episode_logging/ # Structured logging
├── tests/               # Test files
├── scripts/             # Utility scripts and demos
├── logs/                # Episode logs (gitignored)
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
