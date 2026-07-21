.PHONY: setup lint test run-pipeline docker-build docker-run help

setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	pre-commit install

lint:
	black --check src tests
	ruff check src tests
	@which mypy > /dev/null 2>&1 && mypy src tests || echo "mypy not present; skipping static type checking."

format:
	black src tests
	ruff check --fix --select I src tests
	ruff check --fix src tests

test:
	pytest tests/

run-pipeline:
	dvc repro

docker-build:
	docker build -t fraud-detection-api:latest -f docker/Dockerfile .

docker-run:
	docker run -p 8000:8000 fraud-detection-api:latest

help:
	@echo "Available commands:"
	@echo "  setup         - Install dependencies and pre-commit hooks"
	@echo "  lint          - Run formatting and quality checks"
	@echo "  format        - Format codebase automatically"
	@echo "  test          - Run unit tests with pytest"
	@echo "  run-pipeline  - Execute DVC pipeline"
	@echo "  docker-build  - Build API Docker image"
	@echo "  docker-run    - Run API web service locally"
