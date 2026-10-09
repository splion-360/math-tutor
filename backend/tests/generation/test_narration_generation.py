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


class SequenceGenerator(RecordingGenerator):
    """Return configured responses in order and retain every request."""

    def __init__(self, contents: list[str]) -> None:
        super().__init__(contents[0])
        self._contents = iter(contents)

    def generate(self, prompt: str) -> GenerationResult:
        self.content = next(self._contents)
        return super().generate(prompt)


def test_model_narration_planner_uses_prompt_source_and_video_duration(tmp_path) -> None:
    generator = RecordingGenerator(
        '```json\n{"segments":[{"text":"Start at the curve."},'
        '{"text":"Follow the slope downhill."}]}\n```'
    )

    result = ModelNarrationPlanner(generator).create_plan(
        lesson_id="lesson-1",
        prompt="Explain gradient descent.",
        source='class GradientScene(Scene):\n    """Show the loss curve."""',
        target_duration_seconds=12.4,
        artifact_dir=tmp_path,
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
    assert result.generation_attempts == 1
    assert (tmp_path / "narration-response-attempt-0.txt").read_text() == result.raw_response


def test_model_narration_planner_retries_invalid_json_once_and_preserves_evidence(
    tmp_path,
) -> None:
    generator = SequenceGenerator(
        [
            "This is not JSON.",
            '{"segments":[{"text":"Start at the curve."},{"text":"Move downhill."}]}',
        ]
    )

    result = ModelNarrationPlanner(generator, max_retries=1).create_plan(
        lesson_id="lesson-1",
        prompt="Explain gradient descent.",
        source="class GradientScene(Scene): pass",
        target_duration_seconds=12.4,
        artifact_dir=tmp_path,
    )

    assert result.generation_attempts == 2
    assert len(generator.prompts) == 2
    retry_request = json.loads(generator.prompts[1])
    assert retry_request["previous_response"] == "This is not JSON."
    assert (tmp_path / "narration-response-attempt-0.txt").read_text() == "This is not JSON."
    assert "Start at the curve" in (tmp_path / "narration-response-attempt-1.txt").read_text()


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
    tmp_path,
) -> None:
    with pytest.raises(NarrationPlanningError) as caught:
        ModelNarrationPlanner(RecordingGenerator(content)).create_plan(
            lesson_id="lesson-1",
            prompt="Explain limits.",
            source="class LimitScene(Scene): pass",
            target_duration_seconds=10,
            artifact_dir=tmp_path,
        )

    assert caught.value.diagnostics["narration_plan_attempt_count"] == 2
    assert (tmp_path / "narration-response-attempt-0.txt").is_file()
    assert (tmp_path / "narration-response-attempt-1.txt").is_file()
