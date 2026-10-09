"""Verify that spatial findings drive one bounded lesson regeneration.
The integration test uses real pipeline, media, spatial, and artifact boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from math_tutor.generation.pipeline import GeneratedLessonPipeline
from math_tutor.generation.provider import GenerationConfig, GenerationResult, TokenUsage
from math_tutor.jobs import RenderOutcome
from math_tutor.validation.media import MediaInspection, MediaValidator
from math_tutor.validation.spatial import SpatialValidator
from math_tutor.validation.suite import ValidatorSuite

_VALID_SCENE = """from manim import *

class GeneratedLesson(Scene):
    def construct(self):
        self.play(Write(MathTex(r"x^2")))
"""


class _SequenceGenerator:
    """Return valid source twice while retaining regeneration prompts."""

    config = GenerationConfig()

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> GenerationResult:
        """Return a fixed valid scene and record its input prompt."""
        self.prompts.append(prompt)
        return GenerationResult(
            content=f"```python\n{_VALID_SCENE}```",
            model="test-model",
            request_id=f"request-{len(self.prompts)}",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            elapsed_seconds=0.1,
            provider_response='{"choices": []}',
        )


class _SpatialSequenceRenderer:
    """Render an off-frame first trace followed by a valid trace."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self.calls = 0

    def render_source(
        self,
        job_id: str,
        source: str,
        scene_class: str,
    ) -> RenderOutcome:
        """Write one video and its attempt-specific geometry trace."""
        self.calls += 1
        self._root.mkdir(exist_ok=True)
        video = self._root / f"{job_id}.mp4"
        trace = self._root / f"{job_id}-trace.json"
        video.write_bytes(b"video")
        bounds = (
            {"left": 4.5, "bottom": -1, "right": 6, "top": 1}
            if self.calls == 1
            else {"left": -1, "bottom": -1, "right": 1, "top": 1}
        )
        trace.write_text(
            json.dumps(
                {
                    "schema_version": "manim-spatial-trace.v1",
                    "frame": {"width": 10, "height": 6},
                    "checkpoints": [
                        {
                            "index": 0,
                            "kind": "final",
                            "time_seconds": 1,
                            "objects": [
                                {
                                    "id": "equation",
                                    "type": "MathTex",
                                    "bounds": bounds,
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return RenderOutcome(
            video,
            "test",
            1,
            "rendered",
            spatial_trace_path=trace,
        )


class _PassingMediaInspector:
    """Return deterministic media metadata accepted by MediaValidator."""

    def inspect(self, video_path: Path) -> MediaInspection:
        """Report one decodable silent video with target duration."""
        return MediaInspection(
            video_stream_count=1,
            audio_stream_count=0,
            video_duration_seconds=35,
            audio_duration_seconds=None,
        )


def test_pipeline_repairs_spatial_failure_once_before_publication(tmp_path: Path) -> None:
    generator = _SequenceGenerator()
    renderer = _SpatialSequenceRenderer(tmp_path / "rendered")
    validator = ValidatorSuite(
        (
            MediaValidator(inspector=_PassingMediaInspector()),
            SpatialValidator(),
        )
    )
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain distance between two points.",
        generator=generator,
        renderer=renderer,
        validator=validator,
        max_repair_attempts=1,
    )

    outcome = pipeline.render("spatial-repair-123")

    assert renderer.calls == 2
    assert "object_off_frame" in generator.prompts[1]
    assert outcome.validation_diagnostics is not None
    assert outcome.validation_diagnostics["repair_count"] == 1
    attempts = tmp_path / "artifacts" / "spatial-repair-123" / "attempts"
    first_report = json.loads((attempts / "0" / "validation.json").read_text())
    second_report = json.loads((attempts / "1" / "validation.json").read_text())
    assert first_report["component_reports"][0]["status"] == "pass"
    assert first_report["component_reports"][1]["status"] == "fail"
    assert second_report["component_reports"][0]["status"] == "pass"
    assert second_report["component_reports"][1]["status"] == "pass"
    assert json.loads((attempts / "0" / "attempt.json").read_text())["render"]["spatial_trace"][
        "sha256"
    ]
