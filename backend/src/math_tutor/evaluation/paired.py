"""Evaluate a complete matrix of raw paired generations with the Docker renderer.
Preserve first-attempt failures and keep human video judgments explicitly pending."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any, TypedDict, cast

from math_tutor.generation.errors import ExtractionError, SceneValidationError
from math_tutor.generation.pipeline import SourceRenderer
from math_tutor.generation.source import extract_and_validate_raw_scene
from math_tutor.rendering.manim import (
    DEFAULT_MANIM_IMAGE,
    DockerManimRenderer,
    RenderFailed,
    RenderTimedOut,
)

CONDITIONS = ("base", "shared_adapter")


class GenerationView(TypedDict):
    """Required fields of a serialized first-attempt generation."""

    id: str
    condition: str
    difficulty: str
    topic: str
    prompt: str
    response: str
    first_attempt: bool
    completion_token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    generation_latency_seconds: float
    finish_reason: str
    output_capped: bool


class PlanView(TypedDict):
    """Frozen evaluation inputs consumed by the local renderer."""

    examples: list[dict[str, Any]]
    decoding: dict[str, Any]
    renderer: dict[str, Any]
    scope: str


def renderer_contract() -> dict[str, Any]:
    """Identify the actual pinned renderer and executable evaluation sources.

    Returns:
        Image, timeout, and SHA-256 digests frozen before model generation.
    """
    package_root = Path(__file__).parents[1]
    sources = (
        "evaluation/paired.py",
        "generation/source.py",
        "rendering/manim.py",
        "rendering/scenes/render_known.py",
    )
    return {
        "image": DEFAULT_MANIM_IMAGE,
        "timeout_seconds": 90,
        "source_hashes": {
            name: sha256((package_root / name).read_bytes()).hexdigest()
            for name in sources
        },
    }


def validate_pairs(rows: list[GenerationView], plan: PlanView) -> None:
    """Require one unchanged first-attempt generation per prompt and condition.

    Args:
        rows: Raw GPU generation records.
        plan: Frozen prompts and generation settings.

    Raises:
        ValueError: If pairing, prompt identity, or measured generation fields are invalid.
    """
    examples = {row["id"]: row for row in plan["examples"]}
    expected = {(identifier, condition) for identifier in examples for condition in CONDITIONS}
    keys = [(row["id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("generations must exactly match the frozen paired matrix")
    for row in rows:
        example = examples[row["id"]]
        if any(row.get(key) != example.get(key) for key in ("prompt", "topic", "difficulty")):
            raise ValueError("generation prompt identity differs from the frozen plan")
        if row.get("first_attempt") is not True or not isinstance(row.get("response"), str):
            raise ValueError("only raw first-attempt responses are accepted")
        for key in ("prompt_tokens", "completion_tokens"):
            if type(row.get(key)) is not int or cast(int, row.get(key)) < 0:
                raise ValueError("generation token usage must be measured nonnegative integers")
        elapsed = row.get("generation_latency_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, int | float)
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("generation latency must be finite and nonnegative")
        if row.get("finish_reason") not in {"eos", "length"}:
            raise ValueError("unknown finish reason")
        limit = plan["decoding"]["max_new_tokens"]
        if row["completion_tokens"] > limit or (
            row["finish_reason"] == "length" and row["completion_tokens"] != limit
        ):
            raise ValueError("completion termination is inconsistent with the frozen cap")
        tokens = row.get("completion_token_ids")
        if (
            not isinstance(tokens, list)
            or len(tokens) != row["completion_tokens"]
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ValueError("completion token IDs disagree with measured token count")
        capped = row["completion_tokens"] == limit
        if type(row.get("output_capped")) is not bool or row["output_capped"] != (
            row["finish_reason"] == "length" and capped
        ):
            raise ValueError("output cap status is inconsistent with measured completion")


def wilson_interval(successes: int, total: int) -> list[float]:
    """Return a marginal 95 percent Wilson interval for a binomial count.

    Args:
        successes: Success count between zero and total.
        total: Positive number of evaluated prompts.

    Returns:
        Lower and upper bounds; this does not estimate the paired difference.

    Raises:
        ValueError: If the counts are invalid.
    """
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    midpoint = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, midpoint - radius), min(1.0, midpoint + radius)]


def classify_render_failure(logs: str) -> str:
    """Classify renderer logs using the failure categories recorded by issue 28.

    Args:
        logs: Renderer diagnostics or exception message.

    Returns:
        Missing import/name, invalid API, syntax, or unclassified render failure.
    """
    if "NameError" in logs or "ModuleNotFoundError" in logs or "ImportError" in logs:
        return "missing_import_or_name"
    if "AttributeError" in logs or "TypeError" in logs:
        return "invalid_api"
    if "SyntaxError" in logs:
        return "syntax"
    return "render_other"


def evaluate_pairs(
    *,
    rows: list[GenerationView],
    plan: PlanView,
    output: Path,
    renderer: SourceRenderer,
) -> dict[str, Any]:
    """Validate and render every first attempt, writing reviewable evidence.

    Args:
        rows: Complete matched GPU response matrix.
        plan: Frozen settings, prompts, and review checks.
        output: New evaluation directory, containing renderer artifacts.
        renderer: Existing isolated renderer with matched settings for both conditions.

    Returns:
        Automatic counts, per-prompt evidence, and pending human-review records.

    Raises:
        ValueError: If the generation matrix does not match the plan.
        OSError: If evidence cannot be written.
    """
    validate_pairs(rows, plan)
    output.mkdir(parents=True, exist_ok=False)
    attempts: list[dict[str, Any]] = []
    for row in rows:
        key = f"{row['condition']}-{row['id']}"
        response_path = output / "responses" / f"{key}.txt"
        response_path.parent.mkdir(exist_ok=True)
        response_path.write_text(row["response"])
        result: dict[str, Any] = {
            key: row.get(key)
            for key in (
                "id",
                "condition",
                "difficulty",
                "topic",
                "prompt",
                "prompt_tokens",
                "completion_tokens",
                "generation_latency_seconds",
                "finish_reason",
                "output_capped",
            )
        }
        result.update(
            {
                "extraction_success": False,
                "parse_success": False,
                "validation_success": False,
                "render_success": False,
                "render_latency_seconds": None,
                "failure_stage": None,
                "failure_category": None,
                "diagnostics": None,
                "video_path": None,
                "response_path": str(response_path.relative_to(output)),
                "response_sha256": sha256(row["response"].encode()).hexdigest(),
                "mathematical_correctness": None,
                "prompt_adherence": None,
                "duration_adherence": None,
                "human_review_status": "pending",
            }
        )
        try:
            scene = extract_and_validate_raw_scene(row["response"])
            result.update(extraction_success=True, parse_success=True, validation_success=True)
        except ExtractionError as error:
            result.update(
                failure_stage="extraction",
                failure_category="extraction",
                diagnostics=str(error),
                human_review_status="unreviewable_no_video",
            )
        except SceneValidationError as error:
            stage = error.diagnostics.get("failure_stage", "validation")
            result.update(
                extraction_success=True,
                parse_success=stage != "parse",
                failure_stage=stage,
                failure_category="syntax" if stage == "parse" else "contract",
                diagnostics=str(error),
                human_review_status="unreviewable_no_video",
            )
        else:
            started = monotonic()
            try:
                rendered = renderer.render_source(key, scene.source, scene.scene_class)
                result.update(
                    render_success=True,
                    video_path=str(rendered.video_path.relative_to(output.resolve())),
                    render_latency_seconds=rendered.elapsed_seconds,
                )
            except (RenderFailed, RenderTimedOut) as error:
                category = (
                    "timeout"
                    if isinstance(error, RenderTimedOut)
                    else (
                        classify_render_failure(str(error.diagnostics.get("logs", "")) + str(error))
                    )
                )
                result.update(
                    failure_stage="render",
                    failure_category=category,
                    diagnostics=error.diagnostics,
                    render_latency_seconds=monotonic() - started,
                    human_review_status="unreviewable_no_video",
                )
        attempts.append(result)
        with (output / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print(
            f"{key}: render={result['render_success']} capped={result['output_capped']}", flush=True
        )
    summaries = {}
    for condition in CONDITIONS:
        subset = [row for row in attempts if row["condition"] == condition]
        successes = sum(row["render_success"] for row in subset)
        summaries[condition] = {
            "attempts": len(subset),
            "render_successes": successes,
            "render_pass_at_1": successes / len(subset),
            "render_marginal_wilson_95": wilson_interval(successes, len(subset)),
            "extraction_successes": sum(row["extraction_success"] for row in subset),
            "parse_successes": sum(row["parse_success"] for row in subset),
            "validation_successes": sum(row["validation_success"] for row in subset),
            "output_capped": sum(row["output_capped"] for row in subset),
            "failure_categories": dict(
                Counter(row["failure_category"] for row in subset if row["failure_category"])
            ),
            "human_reviewed": 0,
            "mathematical_correctness": None,
            "prompt_adherence": None,
        }
    by_id = {(row["id"], row["condition"]): row for row in attempts}
    pairs = Counter(
        f"base_{int(by_id[(example['id'], 'base')]['render_success'])}"
        f"_adapter_{int(by_id[(example['id'], 'shared_adapter')]['render_success'])}"
        for example in plan["examples"]
    )
    report = {
        "conditions": summaries,
        "paired_render_outcomes": dict(pairs),
        "attempts": attempts,
        "renderer": plan["renderer"]["image"],
        "render_timeout_seconds": plan["renderer"]["timeout_seconds"],
        "human_review_status": "pending",
        "scope": plan["scope"],
        "uncertainty_note": "Marginal Wilson intervals assume binomial sampling; these authored "
        "prompts are not a random population sample. Intervals do not test the paired difference.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output / "human_review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "id",
                "condition",
                "video_path",
                "review_checks",
                "reviewer",
                "reviewed_at_utc",
                "mathematical_correctness",
                "prompt_adherence",
                "duration_seconds",
                "notes",
            ],
        )
        writer.writeheader()
        examples = {row["id"]: row for row in plan["examples"]}
        for attempt in attempts:
            writer.writerow(
                {
                    key: value
                    for key, value in {
                        "id": attempt["id"],
                        "condition": attempt["condition"],
                        "video_path": attempt["video_path"],
                        "review_checks": " | ".join(examples[attempt["id"]]["review_checks"]),
                    }.items()
                }
            )
    return report


def main() -> None:
    """Render a downloaded paired run using the same pinned Docker image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    plan = cast(PlanView, json.loads((args.run / "frozen_plan.json").read_text()))
    if plan["renderer"] != renderer_contract():
        parser.error("renderer/evaluator differs from the contract frozen before generation")
    runtime = json.loads((args.run / "runtime.json").read_text())
    if (
        runtime["status"] != "completed"
        or sha256((args.run / "frozen_plan.json").read_bytes()).hexdigest()
        != runtime["plan_sha256"]
    ):
        parser.error("run must be complete and use its recorded frozen plan")
    rows = cast(
        list[GenerationView],
        [json.loads(line) for line in (args.run / "generations.jsonl").read_text().splitlines()],
    )
    output = args.run / "evaluation"
    renderer = DockerManimRenderer(
        artifact_root=output / "renders",
        image=plan["renderer"]["image"],
        timeout_seconds=plan["renderer"]["timeout_seconds"],
        scene_path=(
            Path(__file__).parents[1]
            / "rendering"
            / "scenes"
            / "pythagorean_theorem.py"
        ),
    )
    report = evaluate_pairs(rows=rows, plan=plan, output=output, renderer=renderer)
    print(json.dumps(report["conditions"], indent=2))


if __name__ == "__main__":
    main()
