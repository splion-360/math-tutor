"""Run the frozen single-attempt Manim evaluation and aggregate metrics.
Evaluation artifacts preserve the original baseline contract without repair."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from math_tutor.generated_lesson import GeneratedLessonError, GeneratedLessonPipeline
from math_tutor.generation import GenerationConfig, NebiusTokenFactoryClient, ProviderError
from math_tutor.jobs import is_safe_job_id
from math_tutor.renderer import (
    DEFAULT_MANIM_IMAGE,
    DockerManimRenderer,
    RenderFailed,
    RenderTimedOut,
)
from math_tutor.settings import get_settings

Difficulty = Literal["foundational", "intermediate", "advanced"]
DIFFICULTIES: tuple[Difficulty, ...] = ("foundational", "intermediate", "advanced")


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationExample:
    id: str
    difficulty: Difficulty
    topic: str
    prompt: str
    split: str
    source_name: str
    source_reference: str


@dataclass(frozen=True)
class AttemptRecord:
    example_id: str
    difficulty: str
    topic: str
    prompt: str
    model: str
    extraction_success: bool
    parse_success: bool
    render_success: bool
    timed_out: bool
    generation_latency_seconds: float
    render_latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    first_attempt: bool
    failure_stage: str | None
    artifact_directory: str
    renderer: str


def load_evaluation_slice(path: Path) -> list[EvaluationExample]:
    examples: list[EvaluationExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item: Any = json.loads(line)
            source = item["source"]
            difficulty = str(item["difficulty"])
            if difficulty not in DIFFICULTIES:
                raise DatasetValidationError(
                    f"line {line_number} has unknown difficulty '{difficulty}'"
                )
            examples.append(
                EvaluationExample(
                    id=str(item["id"]),
                    difficulty=difficulty,
                    topic=str(item["topic"]),
                    prompt=str(item["prompt"]),
                    split=str(item["split"]),
                    source_name=str(source["name"]),
                    source_reference=str(source["reference"]),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise DatasetValidationError(
                f"invalid evaluation example on line {line_number}"
            ) from error
    validate_evaluation_slice(examples, training_ids=set())
    return examples


def validate_evaluation_slice(
    examples: Sequence[EvaluationExample],
    *,
    training_ids: set[str],
) -> None:
    ids = [example.id for example in examples]
    if len(ids) != len(set(ids)):
        raise DatasetValidationError("evaluation identifiers must be unique")
    if any(not is_safe_job_id(identifier) for identifier in ids):
        raise DatasetValidationError("evaluation identifiers must be safe artifact identifiers")
    overlap = set(ids) & training_ids
    if overlap:
        raise DatasetValidationError(
            f"evaluation and training identifiers overlap: {', '.join(sorted(overlap))}"
        )
    counts = Counter(example.difficulty for example in examples)
    missing = [difficulty for difficulty in DIFFICULTIES if counts[difficulty] < 5]
    if missing:
        raise DatasetValidationError(
            "evaluation slice requires at least five examples per difficulty: " + ", ".join(missing)
        )
    if any(example.split != "evaluation" for example in examples):
        raise DatasetValidationError("every held-out example must use split='evaluation'")
    if any(
        not example.id
        or not example.topic
        or not example.prompt
        or not example.source_name
        or not example.source_reference
        for example in examples
    ):
        raise DatasetValidationError("evaluation examples require stable IDs and source metadata")


def aggregate_metrics(attempts: Sequence[AttemptRecord]) -> dict[str, object]:
    first_attempts = [attempt for attempt in attempts if attempt.first_attempt]
    if not first_attempts:
        raise ValueError("at least one first attempt is required")
    return {
        "first_attempt": _summarize(first_attempts),
        "repair_attempts": sum(not attempt.first_attempt for attempt in attempts),
        "by_difficulty": {
            difficulty: _summarize(
                [attempt for attempt in first_attempts if attempt.difficulty == difficulty]
            )
            for difficulty in DIFFICULTIES
        },
    }


def _summarize(attempts: Sequence[AttemptRecord]) -> dict[str, int | float | None]:
    count = len(attempts)
    if count == 0:
        return {
            "attempts": 0,
            "extraction_success_rate": None,
            "parse_success_rate": None,
            "render_pass_at_1": None,
            "timeout_rate": None,
            "mean_generation_latency_seconds": None,
            "mean_render_latency_seconds": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    return {
        "attempts": count,
        "extraction_success_rate": sum(a.extraction_success for a in attempts) / count,
        "parse_success_rate": sum(a.parse_success for a in attempts) / count,
        "render_pass_at_1": sum(a.render_success for a in attempts) / count,
        "timeout_rate": sum(a.timed_out for a in attempts) / count,
        "mean_generation_latency_seconds": fmean(a.generation_latency_seconds for a in attempts),
        "mean_render_latency_seconds": fmean(a.render_latency_seconds for a in attempts),
        "prompt_tokens": sum(a.prompt_tokens for a in attempts),
        "completion_tokens": sum(a.completion_tokens for a in attempts),
        "total_tokens": sum(a.total_tokens for a in attempts),
    }


def run_evaluation(
    *,
    dataset_path: Path,
    artifact_root: Path,
    run_id: str,
    api_key: str,
    base_url: str,
) -> dict[str, object]:
    examples = load_evaluation_slice(dataset_path)
    config = GenerationConfig()
    client = NebiusTokenFactoryClient(
        api_key=api_key,
        config=config,
        base_url=base_url,
    )
    health = client.health()
    if not health.reachable or not health.model_available:
        client.close()
        raise RuntimeError(
            f"frozen model {config.model} is not available from the configured Nebius endpoint"
        )

    run_dir = artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    attempts_dir = run_dir / "attempts"
    renderer = DockerManimRenderer(
        artifact_root=attempts_dir,
        scene_path=Path(__file__).parent / "scenes" / "pythagorean_theorem.py",
    )
    manifest = {
        "run_id": run_id,
        "model": config.model,
        "decoding": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
        },
        "renderer": f"docker:{DEFAULT_MANIM_IMAGE}",
        "python": sys.version.split()[0],
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path.read_bytes()).hexdigest(),
        "example_count": len(examples),
    }
    _write_json(run_dir / "run.json", manifest)

    attempts: list[AttemptRecord] = []
    try:
        for example in examples:
            pipeline = GeneratedLessonPipeline(
                artifact_root=attempts_dir,
                prompt=example.prompt,
                generator=client,
                renderer=renderer,
                generation_provider="nebius_token_factory",
                max_repair_attempts=0,
            )
            failure_stage: str | None = None
            timed_out = False
            render_success = False
            try:
                pipeline.render(example.id)
                render_success = True
            except RenderTimedOut:
                failure_stage = "render"
                timed_out = True
            except RenderFailed:
                failure_stage = "render"
            except GeneratedLessonError as error:
                failure_stage = str(error.diagnostics.get("failure_stage", "validation"))
            except ProviderError:
                failure_stage = "provider"
            except Exception:
                failure_stage = "error"

            job_dir = attempts_dir / example.id / "attempts" / "0"
            generation = _read_json_if_present(job_dir / "generation.json")
            render = _read_json_if_present(
                attempts_dir / f"{example.id}-attempt-0" / "render.json"
            )
            extraction_success = failure_stage not in {"provider", "extraction"}
            parse_success = extraction_success and failure_stage != "parse"
            attempts.append(
                AttemptRecord(
                    example_id=example.id,
                    difficulty=example.difficulty,
                    topic=example.topic,
                    prompt=example.prompt,
                    model=str(generation.get("model", config.model)),
                    extraction_success=extraction_success,
                    parse_success=parse_success,
                    render_success=render_success,
                    timed_out=timed_out,
                    generation_latency_seconds=_float_field(generation, "elapsed_seconds"),
                    render_latency_seconds=_float_field(render, "elapsed_seconds"),
                    prompt_tokens=_int_field(generation, "prompt_tokens"),
                    completion_tokens=_int_field(generation, "completion_tokens"),
                    total_tokens=_int_field(generation, "total_tokens"),
                    first_attempt=True,
                    failure_stage=failure_stage,
                    artifact_directory=str(Path("attempts") / example.id),
                    renderer=str(render.get("image", DEFAULT_MANIM_IMAGE)),
                )
            )
            _write_attempts(run_dir / "attempts.jsonl", attempts)
    finally:
        client.close()

    metrics = aggregate_metrics(attempts)
    _write_json(run_dir / "metrics.json", metrics)
    return metrics


def _write_attempts(path: Path, attempts: Iterable[AttemptRecord]) -> None:
    path.write_text(
        "".join(json.dumps(asdict(attempt), sort_keys=True) + "\n" for attempt in attempts),
        encoding="utf-8",
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _read_json_if_present(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _float_field(value: dict[str, object], key: str) -> float:
    field = value.get(key, 0)
    return float(field) if isinstance(field, (int, float)) else 0.0


def _int_field(value: dict[str, object], key: str) -> int:
    field = value.get(key, 0)
    return int(field) if isinstance(field, (int, float)) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen zero-shot Manim evaluation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/manim_eval_v1.jsonl"),
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/evaluations"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    settings = get_settings()
    if settings.nebius_api_key is None:
        raise SystemExit("NEBIUS_API_KEY is required")
    metrics = run_evaluation(
        dataset_path=args.dataset,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        api_key=settings.nebius_api_key.get_secret_value(),
        base_url=settings.nebius_base_url,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
