.PHONY: install test lint clean

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

test:
	pytest -v

test-fast:
	pytest -v -m "not slow and not neural"

lint:
	ruff check koopsim/ tests/
	ruff format --check koopsim/ tests/

format:
	ruff format koopsim/ tests/
	ruff check --fix koopsim/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
