.PHONY: sync lint format test check shellcheck

sync:
	uv sync --locked --extra cpu --group dev

lint:
	uv run --frozen --no-sync ruff check src tests

format:
	uv run --frozen --no-sync ruff format src tests

test:
	uv run --frozen --no-sync pytest -m "not gpu and not cluster"

shellcheck:
	@for f in scripts/*.sh slurm/*.sh slurm/dls/*.sh slurm/*.sbatch; do bash -n "$$f" || exit 1; done
	@echo "shell syntax ok"

check: shellcheck
	uv run --frozen --no-sync ruff format --check src tests
	uv run --frozen --no-sync ruff check src tests
	uv run --frozen --no-sync pytest -m "not gpu and not cluster"
