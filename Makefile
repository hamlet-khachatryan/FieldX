.PHONY: sync lint format test check

sync:
	uv sync --locked --extra cpu --group dev

lint:
	uv run --frozen --no-sync ruff check src tests

format:
	uv run --frozen --no-sync ruff format src tests

test:
	uv run --frozen --no-sync pytest -m "not gpu and not cluster"

check:
	uv run --frozen --no-sync ruff format --check src tests
	uv run --frozen --no-sync ruff check src tests
	uv run --frozen --no-sync pytest -m "not gpu and not cluster"
