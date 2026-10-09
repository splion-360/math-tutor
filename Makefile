# Provides short commands for local setup, development, and verification.
# Keeps Docker Compose and package-specific tools as the underlying sources of truth.

.DEFAULT_GOAL := help

.PHONY: help setup up down logs renderer evidence-data evidence-figures gradient-label-analysis evaluation-freeze evaluation-generate evaluation-download evaluation-render evaluation-review test lint validate-staged-python lint-staged format-check format-staged typecheck secrets pre-commit-check install-gitleaks setup-hooks check

MODAL_PROFILE ?=
MODAL_PROFILE_FLAG := $(if $(MODAL_PROFILE),--profile $(MODAL_PROFILE),)
EVALUATION_RUN ?= paired-pilot-v2-greedy
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
		'make gradient-label-analysis  Test projected gradients against subject and difficulty labels' \
		'make evaluation-freeze  Freeze prompts, token allowance, and renderer contract on CPU' \
		'make evaluation-generate  Run the frozen paired pilot on Modal (uses GPU credits)' \
		'make evaluation-download  Download the completed paired run' \
		'make evaluation-render  Render raw first attempts and prepare human review' \
		'make evaluation-review  Import completed human video judgments' \
		'make test       Run backend, frontend, and training tests' \
		'make lint       Run Python lint checks' \
		'make format-check  Check Python formatting' \
		'make typecheck  Run Python and frontend type checks' \
		'make secrets    Scan staged changes for leaked secrets' \
		'make setup-hooks  Install Gitleaks and enable repository hooks' \
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

gradient-label-analysis:
	uv run --frozen --project training --python 3.12 python -m dynamic_lora.evidence fetch-all
	uv run --frozen --project training --python 3.12 python training/scripts/analyze_gradient_label_alignment.py \
		--signatures training/artifacts/evidence/full_corpus_signatures.jsonl \
		--records training/artifacts/evidence/bespoke_manim_train.jsonl \
		--exact-summary training/artifacts/evidence/lora_cosines.json \
		--output-directory training/artifacts/gradient-label-alignment

evaluation-freeze:
	uv run --frozen --project training --python 3.12 --with transformers==4.57.6 --with jinja2==3.1.6 --with-editable ./backend python -m dynamic_lora.paired_generation

evaluation-generate:
	uvx modal run $(MODAL_PROFILE_FLAG) deployments/modal/paired_evaluation.py --run-id $(EVALUATION_RUN)

evaluation-download:
	mkdir -p $(EVALUATION_DIR)
	uvx modal volume get $(MODAL_PROFILE_FLAG) math-tutor-paired-evaluation $(EVALUATION_RUN)/frozen_plan.json $(EVALUATION_DIR)/frozen_plan.json --force
	uvx modal volume get $(MODAL_PROFILE_FLAG) math-tutor-paired-evaluation $(EVALUATION_RUN)/runtime.json $(EVALUATION_DIR)/runtime.json --force
	uvx modal volume get $(MODAL_PROFILE_FLAG) math-tutor-paired-evaluation $(EVALUATION_RUN)/generations.jsonl $(EVALUATION_DIR)/generations.jsonl --force

evaluation-render:
	uv run --frozen --project backend python -m math_tutor.evaluation.paired --run $(EVALUATION_DIR)
	uv run --frozen --project backend python -m math_tutor.evaluation.review --evaluation $(EVALUATION_DIR)/evaluation --prepare

evaluation-review:
	uv run --frozen --project backend python -m math_tutor.evaluation.review --evaluation $(EVALUATION_DIR)/evaluation --reviews $(EVALUATION_DIR)/evaluation/human_review_bound.csv

test:
	cd backend && uv run pytest -q
	cd frontend && npm test
	cd training && uv run pytest -q

lint:
	cd backend && uv run ruff check src tests
	cd training && uv run ruff check src tests

validate-staged-python:
	python3 .githooks/check_staged_python.py validate

lint-staged:
	python3 .githooks/check_staged_python.py lint

format-check:
	cd backend && uv run ruff format --check src tests
	cd training && uv run ruff format --check src tests

format-staged:
	python3 .githooks/check_staged_python.py format

typecheck:
	cd backend && uv run mypy src
	cd frontend && npm run typecheck
	cd training && uv run mypy src

secrets:
	@command -v gitleaks >/dev/null 2>&1 || { \
		printf '%s\n' 'Gitleaks is required. Run make install-gitleaks.' >&2; \
		exit 1; \
	}
	gitleaks git --pre-commit --staged --redact

pre-commit-check:
	python3 .githooks/check_staged_python.py all
	@$(MAKE) --no-print-directory secrets

install-gitleaks:
	@if command -v gitleaks >/dev/null 2>&1; then \
		printf '%s\n' "Gitleaks $$(gitleaks version) is already installed."; \
	elif command -v brew >/dev/null 2>&1; then \
		brew install gitleaks; \
	else \
		printf '%s\n' 'Homebrew is required to install Gitleaks automatically.' >&2; \
		printf '%s\n' 'Install it from https://github.com/gitleaks/gitleaks#installing and retry.' >&2; \
		exit 1; \
	fi

setup-hooks: install-gitleaks
	chmod +x .githooks/pre-commit
	git config core.hooksPath .githooks
	@printf '%s\n' 'Configured core.hooksPath=.githooks for this repository.'

check: test lint typecheck
