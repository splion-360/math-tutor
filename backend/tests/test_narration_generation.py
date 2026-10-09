"""Verify model-generated narration plans for rendered Manim lessons.
The tests cover prompt construction, strict JSON parsing, and bounded segments."""

from __future__ import annotations

import json

import pytest

from math_tutor.generation.narration import ModelNarrationPlanner, NarrationPlanningError
from math_tutor.generation.provider import GenerationConfig, GenerationResult, TokenUsage


class RecordingGenerator:
    """Return one configured response and record the narration request."""

    def __init__(self, content: str) -> None:
        self.config = GenerationConfig(model="base-model")
        self.content = content
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> GenerationResult:
        self.prompts.append(prompt)
        return GenerationResult(
            content=self.content,
            model=self.config.model,
            request_id="narration-request",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=30, completion_tokens=20, total_tokens=50),
            elapsed_seconds=0.2,
            provider_response=json.dumps({"id": "narration-request"}),
        )


def test_model_narration_planner_uses_prompt_source_and_video_duration() -> None:
    generator = RecordingGenerator(
        '```json\n{"segments":[{"text":"Start at the curve."},'
        '{"text":"Follow the slope downhill."}]}\n```'
    )

    result = ModelNarrationPlanner(generator).create_plan(
        lesson_id="lesson-1",
        prompt="Explain gradient descent.",
        source='class GradientScene(Scene):\n    """Show the loss curve."""',
        target_duration_seconds=12.4,
    )

    request = json.loads(generator.prompts[0])
    assert request["lesson_prompt"] == "Explain gradient descent."
    assert request["manim_source"].startswith("class GradientScene")
    assert request["target_duration_seconds"] == 12.4
    assert request["target_word_count"] == 25
    assert [segment.text for segment in result.plan.segments] == [
        "Start at the curve.",
        "Follow the slope downhill.",
    ]
    assert result.model == "base-model"
    assert result.raw_response.startswith("```json")


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"segments":[]}',
        '{"segments":[{"text":""}]}',
        '{"segments":[{"text":"one"},{"text":"two"},{"text":"three"},'
        '{"text":"four"},{"text":"five"}]}',
    ],
)
def test_model_narration_planner_rejects_malformed_or_unbounded_plans(
    content: str,
) -> None:
    with pytest.raises(NarrationPlanningError):
        ModelNarrationPlanner(RecordingGenerator(content)).create_plan(
            lesson_id="lesson-1",
            prompt="Explain limits.",
            source="class LimitScene(Scene): pass",
            target_duration_seconds=10,
        )
