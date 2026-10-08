# syntax=docker/dockerfile:1

FROM docker:29.2.1-cli AS docker-cli

FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8000

FROM base AS development
RUN uv sync --frozen
CMD ["uvicorn", "math_tutor.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "src/math_tutor"]

FROM base AS production
CMD ["uvicorn", "math_tutor.main:app", "--host", "0.0.0.0", "--port", "8000"]
