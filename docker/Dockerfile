# Multi-stage Dockerfile for Regulator application
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY ../pyproject.toml ./
COPY ../README.md ./
COPY ../src/ ./src/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[dev,ml]"

# Copy remaining source code
COPY .. .

# Create logs directory
RUN mkdir -p logs

# Development stage
FROM base as development
# Install additional dev tools
RUN pip install jupyter ipython
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production
# Remove dev dependencies for smaller image
RUN pip uninstall -y pytest pytest-cov black ruff mypy pre-commit jupyter ipython
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# CLI stage for running experiments
FROM base as cli
# Remove Streamlit and dashboard dependencies for CLI-only usage
RUN pip uninstall -y streamlit
CMD ["python", "-m", "regulator_cli", "--help"]
