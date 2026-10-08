# Provides short commands for local setup, development, and verification.
# Keeps Docker Compose and package-specific tools as the underlying sources of truth.

.DEFAULT_GOAL := help

.PHONY: help setup up down logs renderer evidence-data evidence-figures evaluation-freeze evaluation-generate evaluation-download evaluation-render test lint typecheck check

MODAL_PROFILE ?= splion-360
EVALUATION_RUN ?= paired-pilot-v2
EVALUATION_DIR := training/artifacts/$(EVALUATION_RUN)

help:
	@printf '%s\n' \
		'make setup      Create backend/.env without overwriting it' \
		'make up         Prepare renderers and start the application' \
		'make down       Stop the local application' \
		'make logs       Follow frontend and backend logs' \
		'make renderer   Rebuild the narration renderer' \
		'make evidence-data     Rebuild and verify the pinned experiment dataset' \
		'make evidence-figures  Regenerate figures once numeric summaries are released' \
		'make evaluation-freeze  Freeze prompts, token allowance, and renderer contract on CPU' \
		'make evaluation-generate  Run the frozen paired pilot on Modal (uses GPU credits)' \
		'make evaluation-download  Download the completed paired run' \
		'make evaluation-render  Render raw first attempts and prepare human review' \
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

evaluation-freeze:
	uv run --frozen --project training --python 3.12 --with transformers==4.57.6 --with jinja2==3.1.6 --with-editable ./backend python -m dynamic_lora.paired_generation

evaluation-generate:
	uvx modal run --profile $(MODAL_PROFILE) deployments/modal/paired_evaluation.py --run-id $(EVALUATION_RUN)

evaluation-download:
	mkdir -p $(EVALUATION_DIR)
	uvx modal volume get --profile $(MODAL_PROFILE) math-tutor-paired-evaluation $(EVALUATION_RUN)/frozen_plan.json $(EVALUATION_DIR)/frozen_plan.json --force
	uvx modal volume get --profile $(MODAL_PROFILE) math-tutor-paired-evaluation $(EVALUATION_RUN)/runtime.json $(EVALUATION_DIR)/runtime.json --force
	uvx modal volume get --profile $(MODAL_PROFILE) math-tutor-paired-evaluation $(EVALUATION_RUN)/generations.jsonl $(EVALUATION_DIR)/generations.jsonl --force

evaluation-render:
	uv run --frozen --project backend python -m math_tutor.paired_evaluation --run $(EVALUATION_DIR)

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
