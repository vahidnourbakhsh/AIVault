# Makefile for AIVault

.PHONY: help install install-dev test test-cov lint format clean docs

help:
	@echo "Available commands:"
	@echo "  install     - Install package in production mode"
	@echo "  install-dev - Install package in development mode"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  clean       - Clean temporary files"
	@echo "  docs        - Generate documentation"

install:
	pip install .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=aivault --cov-report=html --cov-report=term

lint:
	flake8 aivault tests examples
	mypy aivault
	black --check aivault tests examples
	isort --check-only aivault tests examples

format:
	black aivault tests examples
	isort aivault tests examples

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/

docs:
	@echo "Documentation generation would go here"
	@echo "Consider using Sphinx or MkDocs"

# Environment setup
env-create:
	@echo "Creating conda environment for your platform..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "Detected macOS - using environment-macos.yml"; \
		conda env create -f environment-macos.yml; \
	else \
		echo "Detected Linux - you can choose:"; \
		echo "  make env-create-gpu  (for NVIDIA GPU support)"; \
		echo "  make env-create-cpu  (for CPU-only)"; \
		conda env create -f environment.yml; \
	fi

env-create-macos:
	conda env create -f environment-macos.yml

env-create-gpu:
	conda env create -f environment-gpu.yml

env-create-cpu:
	conda env create -f environment.yml

env-update:
	@if [ "$$(uname)" = "Darwin" ]; then \
		conda env update -f environment-macos.yml; \
	else \
		conda env update -f environment.yml; \
	fi

env-remove:
	conda env remove -n aivault
