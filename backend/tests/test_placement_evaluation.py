"""Test paired raw-code rendering metrics for placement arms.
These checks keep extraction failure separate from renderer failure."""

from __future__ import annotations

import json
from pathlib import Path

from math_tutor.jobs import RenderOutcome
from math_tutor.placement_evaluation import evaluate_placement_generations


def test_evaluation_counts_parse_and_render_failures_separately(tmp_path: Path) -> None:
    generation_path = tmp_path / "generations.jsonl"
    items = [
        {
            "id": "foundational-1",
            "difficulty": "foundational",
            "response": (
                "from manim import *\nclass Example(Scene):\n"
                "    def construct(self):\n        pass"
            ),
        },
        {"id": "advanced-1", "difficulty": "advanced", "response": "not Python %"},
    ]
    generation_path.write_text(
        "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8"
    )

    class Renderer:
        def render_source(self, job_id: str, source: str, scene_class: str) -> RenderOutcome:
            assert job_id == "foundational-1"
            assert scene_class == "Example"
            assert source.startswith("from manim import *")
            return RenderOutcome(
                video_path=tmp_path / "example.mp4",
                renderer="docker:test",
                elapsed_seconds=1.0,
                logs="ok",
            )

    result = evaluate_placement_generations(
        generations_path=generation_path,
        expected_ids={"foundational-1", "advanced-1"},
        renderer=Renderer(),
    )

    assert result["first_attempt"]["parse_success_rate"] == 0.5
    assert result["first_attempt"]["render_pass_at_1"] == 0.5
    assert result["by_difficulty"]["advanced"]["parse_success_rate"] == 0.0
    assert result["attempts"][1]["failure_stage"] == "parse"
