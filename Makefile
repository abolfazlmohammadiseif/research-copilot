.PHONY: sync test check run

sync:
	uv sync

test:
	uv run pytest

check:
	./scripts/check_tracked_files.sh

run:
	uv run research-copilot
