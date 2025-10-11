# Multi-stage Dockerfile for Regulator application
FROM python:3.11-slim AS base

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
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[dev,ml]"

# Copy remaining source code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Development stage
FROM base AS development
# Install additional dev tools
RUN pip install jupyter ipython
EXPOSE 5000
CMD ["python", "dashboard/main.py"]

# Production stage
FROM base AS production
# Remove dev dependencies for smaller image
RUN pip uninstall -y pytest pytest-cov black ruff mypy pre-commit jupyter ipython
EXPOSE 5000
CMD ["python", "dashboard/main.py"]

# Test stage for running tests (keeps all dependencies)
FROM base AS test
# Keep all dependencies for testing
CMD ["python", "-m", "pytest", "tests/", "-v"]

# CLI stage for running experiments
FROM base AS cli
# Remove dashboard dependencies for CLI-only usage
RUN pip uninstall -y flask || true
CMD ["python", "-m", "regulator_cli", "--help"]
