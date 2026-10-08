# Provides short commands for local setup, development, and verification.
# Keeps Docker Compose and package-specific tools as the underlying sources of truth.

.DEFAULT_GOAL := help

.PHONY: help setup up down logs renderer evidence-data evidence-figures test lint typecheck check

help:
	@printf '%s\n' \
		'make setup      Create backend/.env without overwriting it' \
		'make up         Prepare renderers and start the application' \
		'make down       Stop the local application' \
		'make logs       Follow frontend and backend logs' \
		'make renderer   Rebuild the narration renderer' \
		'make evidence-data     Rebuild and verify the pinned experiment dataset' \
		'make evidence-figures  Regenerate figures once numeric summaries are released' \
		'make test       Run backend, frontend, and training tests' \
		'make lint       Run Python lint checks' \
		'make typecheck  Run Python and frontend type checks' \
		'make check      Run all tests, lint checks, and type checks'

setup:
	@if [ -f backend/.env ]; then \
		printf '%s\n' 'backend/.env already exists; leaving it unchanged.'; \
	else \
		cp backend/.env.example backend/.env; \
		printf '%s\n' 'Created backend/.env. Add the credentials you want to use.'; \
	fi

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs --follow backend frontend

renderer:
	docker compose build manim-voiceover

evidence-data:
	uv run --frozen --project training --python 3.12 --with pyarrow==21.0.0 python -m dynamic_lora.evidence dataset

evidence-figures:
	uv run --frozen --project training --python 3.12 python -m dynamic_lora.evidence fetch
	uv run --frozen --project training --python 3.12 python -m dynamic_lora.evidence figures

test:
	cd backend && uv run pytest -q
	cd frontend && npm test
	cd training && uv run pytest -q

lint:
	cd backend && uv run ruff check src tests
	cd training && uv run ruff check src tests

typecheck:
	cd backend && uv run mypy src
	cd frontend && npm run typecheck
	cd training && uv run mypy src

check: test lint typecheck
