.PHONY: install test lint run clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	python -m black --check src/ tests/
	python -m isort --check src/ tests/
	python -m flake8 src/ tests/
	python -m mypy src/

format:
	python -m black src/ tests/
	python -m isort src/ tests/

run:
	python -m src.main

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
