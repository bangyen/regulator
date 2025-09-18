.PHONY: init fmt lint type test all dashboard install
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
	streamlit run dashboard/app.py

all: fmt lint type test

help: ## show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
