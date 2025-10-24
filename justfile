# Task runner for the Regulator project

# Auto-detect uv - falls back to plain python if not available
PYTHON := `command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python"`

# install tooling
init:
    #!/usr/bin/env bash
    if command -v uv >/dev/null 2>&1; then
        echo "Using uv..."
        uv sync --extra dev
        uv run pre-commit install
    else
        echo "Using pip..."
        python -m pip install -U pip
        pip install -e ".[dev]"
        pre-commit install
    fi

# format code
fmt:
    {{PYTHON}} -m black .

# lint code
lint:
    {{PYTHON}} -m ruff check .

# type-check
type:
    {{PYTHON}} -m mypy src/

# run tests
test:
    {{PYTHON}} -m pytest

# run all checks (fmt, lint, type, test)
all: fmt lint type test
    echo "All checks completed!"

# start Regulator dashboard
dashboard:
    echo "Starting Regulator Dashboard..."
    echo "Dashboard will be available at http://localhost:5000"
    {{PYTHON}} dashboard/main.py

# run smoke tests in Docker
docker-test:
    cd docker && docker compose --profile test up smoke-test

# clean up Docker containers and images
docker-clean:
    cd docker && docker compose down --volumes --remove-orphans
    docker system prune -f

# run one-command containerized demo
demo:
    #!/usr/bin/env bash
    echo "Starting Regulator Containerized Demo..."
    echo "=========================================="
    echo "Creating .env file from template..."
    if [ ! -f .env ]; then
        cp env.template .env
        echo "Please edit .env file and add your OpenAI API key for full functionality"
        echo "   For demo purposes, the app will work with stubbed LLM detection"
    fi
    echo "Building Docker images..."
    cd docker && docker compose build
    echo "Running smoke tests..."
    cd docker && docker compose --profile test up smoke-test
    echo "Starting Regulator Dashboard..."
    echo "   Dashboard will be available at: http://localhost:5000"
    echo "   Press Ctrl+C to stop the demo"
    echo ""
    cd docker && docker compose up dashboard

