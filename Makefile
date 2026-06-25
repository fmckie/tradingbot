# Developer task runner for the AI Trading Bot.
# One-word targets wrap the project's lint/format/typecheck/test tools.
# Requires GNU Make; recipes use TAB indentation.

.PHONY: lint format format-check typecheck test verify install

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	mypy .

test:
	pytest tests/ -q

verify: lint format-check typecheck test

install:
	pip install -r requirements-dev.txt
