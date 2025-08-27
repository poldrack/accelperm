.PHONY: help install install-dev test test-cov lint format type-check clean build docs serve-docs benchmark

help:
	@echo "Available commands:"
	@echo "  make install      Install the package"
	@echo "  make install-dev  Install the package with dev dependencies"
	@echo "  make test         Run tests"
	@echo "  make test-cov     Run tests with coverage"
	@echo "  make lint         Run linting checks"
	@echo "  make format       Format code"
	@echo "  make type-check   Run type checking"
	@echo "  make clean        Clean build artifacts"
	@echo "  make build        Build distribution packages"
	@echo "  make docs         Build documentation"
	@echo "  make serve-docs   Serve documentation locally"
	@echo "  make benchmark    Run benchmarks"

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/

test-cov:
	pytest --cov=accelperm --cov-report=term-missing --cov-report=html tests/

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	isort src/ tests/

type-check:
	mypy src/ --strict

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

docs:
	cd docs && make html

serve-docs:
	cd docs && python -m http.server --directory _build/html

benchmark:
	pytest benchmarks/ --benchmark-only