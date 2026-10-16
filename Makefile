.PHONY: sync hooks fmt lint type test guard check run

sync:
	uv sync

hooks:
	uv run pre-commit install

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

type:
	uv run mypy

test:
	uv run pytest

guard:
	./scripts/check_tracked_files.sh

check: guard lint type test

run:
	uv run research-copilot
