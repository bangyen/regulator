.PHONY: init fmt lint type test all dashboard install demo

init: ## install tooling
	python -m pip install -U pip
	pip install -e ".[dev]"
	pre-commit install

install: ## install package
	python -m pip install -U pip
	pip install -e .

fmt:  ## format code
	black .

lint: ## lint code
	ruff check .

type: ## type-check
	mypy src/

test: ## run tests
	python -m pytest

dashboard: ## run the dashboard
	python dashboard/main.py

all: fmt lint type test

# Docker commands
docker-test: ## run smoke tests in Docker
	cd docker && docker compose --profile test up smoke-test

docker-clean: ## clean up Docker containers and images
	cd docker && docker compose down --volumes --remove-orphans
	docker system prune -f

# One-command demo
demo: ## run one-command containerized demo
	@echo "Starting Regulator Containerized Demo..."
	@echo "=========================================="
	@echo "Creating .env file from template..."
	@if [ ! -f .env ]; then \
		cp env.template .env; \
		echo "Please edit .env file and add your OpenAI API key for full functionality"; \
		echo "   For demo purposes, the app will work with stubbed LLM detection"; \
	fi
	@echo "Building Docker images..."
	cd docker && docker compose build
	@echo "Running smoke tests..."
	cd docker && docker compose --profile test up smoke-test
	@echo "Starting Regulator Dashboard..."
	@echo "   Dashboard will be available at: http://localhost:8503"
	@echo "   Press Ctrl+C to stop the demo"
	@echo ""
	cd docker && docker compose up dashboard

help: ## show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
