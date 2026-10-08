"""Evaluate raw adapter outputs with the isolated Manim renderer.
The evaluator preserves per-prompt failures for paired placement comparisons."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from math_tutor.evaluation.baseline import DIFFICULTIES, load_evaluation_slice
from math_tutor.generation.errors import ExtractionError, SceneValidationError
from math_tutor.generation.pipeline import SourceRenderer
from math_tutor.generation.source import extract_and_validate_raw_scene
from math_tutor.rendering.manim import DockerManimRenderer, RenderFailed, RenderTimedOut


def evaluate_placement_generations(
    *, generations_path: Path, expected_ids: set[str], renderer: SourceRenderer
) -> dict[str, Any]:
    """Parse and render every generated script with no base-model fallback.

    Args:
        generations_path: JSONL of direct adapter generations for one arm.
        expected_ids: Exact IDs of the frozen prompt-only evaluation slice.
        renderer: Renderer that isolates untrusted Manim source.

    Returns:
        Per-prompt outcomes plus overall and difficulty-level success rates.

    Raises:
        ValueError: If any prompt is missing, repeated, or unexpected.
    """
    generations = [
        json.loads(line)
        for line in generations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [item["id"] for item in generations]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("generation IDs must exactly match the evaluation slice")
    attempts: list[dict[str, Any]] = []
    for item in generations:
        result: dict[str, Any] = {
            "id": item["id"],
            "difficulty": item["difficulty"],
            "parse_success": False,
            "render_success": False,
            "timed_out": False,
            "failure_stage": None,
        }
        try:
            scene = extract_and_validate_raw_scene(item["response"])
            result["parse_success"] = True
        except (ExtractionError, SceneValidationError) as error:
            result["failure_stage"] = error.diagnostics.get("failure_stage", "validation")
            attempts.append(result)
            continue
        try:
            outcome = renderer.render_source(item["id"], scene.source, scene.scene_class)
            result["render_success"] = True
            result["render_latency_seconds"] = outcome.elapsed_seconds
            result["video_path"] = str(outcome.video_path)
        except RenderTimedOut:
            result["timed_out"] = True
            result["failure_stage"] = "render_timeout"
        except RenderFailed:
            result["failure_stage"] = "render"
        attempts.append(result)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
        """Summarize parse and render outcomes for one result group."""
        if not rows:
            return {"attempts": 0, "parse_success_rate": None, "render_pass_at_1": None}
        return {
            "attempts": len(rows),
            "parse_success_rate": sum(row["parse_success"] for row in rows) / len(rows),
            "render_pass_at_1": sum(row["render_success"] for row in rows) / len(rows),
        }

    return {
        "first_attempt": summarize(attempts),
        "by_difficulty": {
            difficulty: summarize([row for row in attempts if row["difficulty"] == difficulty])
            for difficulty in DIFFICULTIES
        },
        "failure_stages": dict(Counter(row["failure_stage"] for row in attempts)),
        "attempts": attempts,
    }


def main() -> None:
    """Evaluate downloaded Modal generation files using local Docker."""
    parser = argparse.ArgumentParser(description="Render direct placement-adapter generations")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--evaluation-slice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    docker = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
    if docker.returncode != 0:
        raise RuntimeError("Docker is required to evaluate Manim rendering")
    expected_ids = {item.id for item in load_evaluation_slice(args.evaluation_slice)}
    renderer = DockerManimRenderer(
        artifact_root=args.output / "renders",
        scene_path=(
            Path(__file__).parents[1]
            / "rendering"
            / "scenes"
            / "pythagorean_theorem.py"
        ),
    )
    result = evaluate_placement_generations(
        generations_path=args.generations,
        expected_ids=expected_ids,
        renderer=renderer,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["first_attempt"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
