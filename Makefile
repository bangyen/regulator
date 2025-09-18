.PHONY: init fmt lint type test all
init: ## install tooling
	python -m pip install -U pip
	pip install -e ".[dev]"
	pre-commit install

fmt:  ## format code
	black .

lint: ## lint code
	ruff check .

type: ## type-check
	mypy --ignore-missing-imports src/agents/firm_agents.py src/cartel/cartel_env.py scripts/agent_demo.py

test: ## run tests
	python -m pytest

all: fmt lint type test
