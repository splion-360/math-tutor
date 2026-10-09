"""Verify narration value contracts and immutable normalized metadata.
The tests cover plan ordering, identifiers, durations, and digests."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from math_tutor.jobs import RenderOutcome
from math_tutor.rendering.narration import (
    MediaBundle,
    NarratedSourceRenderer,
    NarrationPlan,
    NarrationSegment,
    NarrationStatus,
    SynthesizedNarration,
    SynthesizedSegment,
)


def test_narrated_source_renderer_keeps_provider_secret_outside_generated_code(
    tmp_path: Path,
) -> None:
    source = """from manim import *
import math
import numpy as np
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.elevenlabs import ElevenLabsService

class GeneratedLesson(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            ElevenLabsService(
                voice_id="voice",
                model="eleven_multilingual_v2",
                transcription_model=None,
            )
        )
        with self.voiceover(text="First explanation.") as tracker:
            self.wait(tracker.duration)
        with self.voiceover(text="Second explanation.") as tracker:
            self.play(FadeIn(Dot()), run_time=tracker.duration)
        with self.voiceover(text="Final explanation.") as tracker:
            self.wait(tracker.duration)
"""
    rendered_sources: list[str] = []
    observed_plans: list[NarrationPlan] = []

    class Renderer:
        def render_source(
            self,
            job_id: str,
            admitted_source: str,
            scene_class: str,
        ) -> RenderOutcome:
            assert job_id == "lesson-1"
            assert scene_class == "GeneratedLesson"
            rendered_sources.append(admitted_source)
            return RenderOutcome(
                video_path=tmp_path / "silent.mp4",
                renderer="fake",
                elapsed_seconds=1.0,
                logs="rendered",
            )

    class Provider:
        def synthesize(
            self,
            plan: NarrationPlan,
            output_dir: Path,
        ) -> SynthesizedNarration:
            observed_plans.append(plan)
            durations = (1.25, 2.5, 0.75)
            segments = tuple(
                SynthesizedSegment(
                    id=segment.id,
                    cue=segment.cue,
                    text=segment.text,
                    audio_path=output_dir / f"{segment.id}.mp3",
                    duration_seconds=duration,
                    sha256=str(index) * 64,
                )
                for index, (segment, duration) in enumerate(
                    zip(plan.segments, durations, strict=True),
                    start=1,
                )
            )
            return SynthesizedNarration(
                lesson_id=plan.lesson_id,
                provider="fake",
                model_id="fake-model",
                segments=segments,
                plan=plan,
            )

    class Assembler:
        def assemble(
            self,
            *,
            silent_video: Path,
            narration: SynthesizedNarration,
            output_dir: Path,
        ) -> MediaBundle:
            assert silent_video == tmp_path / "silent.mp4"
            assert output_dir == tmp_path / "lesson-1" / "narration"
            return MediaBundle(
                silent_video_path=silent_video,
                video_path=output_dir / "narrated.mp4",
                narration_status=NarrationStatus.READY,
                captions_path=output_dir / "captions.vtt",
                diagnostics=MappingProxyType({"narration_provider": narration.provider}),
            )

    outcome = NarratedSourceRenderer(
        renderer=Renderer(),
        provider=Provider(),
        assembler=Assembler(),
        artifact_root=tmp_path,
    ).render_source("lesson-1", source, "GeneratedLesson")

    assert [segment.text for segment in observed_plans[0].segments] == [
        "First explanation.",
        "Second explanation.",
        "Final explanation.",
    ]
    transformed = rendered_sources[0]
    assert "class GeneratedLesson(Scene)" in transformed
    assert "manim_voiceover" not in transformed
    assert "ElevenLabsService" not in transformed
    assert "self.voiceover" not in transformed
    assert "tracker.duration" not in transformed
    assert "run_time=2.5" in transformed
    assert outcome.video_path == tmp_path / "lesson-1" / "narration" / "narrated.mp4"
    assert outcome.narration_status is NarrationStatus.READY


def test_narration_plan_preserves_authoritative_segment_order() -> None:
    plan = NarrationPlan(
        lesson_id="pythagorean-theorem",
        segments=(
            NarrationSegment("introduce", "Consider a right triangle.", "triangle-visible"),
            NarrationSegment("equation", "Its sides satisfy a² + b² = c².", "equation-visible"),
        ),
    )

    assert plan.schema_version == "narration-plan.v1"
    assert [segment.id for segment in plan.segments] == ["introduce", "equation"]


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ((), "at least one segment"),
        (
            (
                NarrationSegment("same", "First.", "first-visible"),
                NarrationSegment("same", "Second.", "second-visible"),
            ),
            "unique",
        ),
    ],
)
def test_narration_plan_rejects_invalid_segment_collections(
    segments: tuple[NarrationSegment, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        NarrationPlan(lesson_id="lesson", segments=segments)


@pytest.mark.parametrize("field", ["id", "text", "cue"])
def test_narration_segment_rejects_blank_required_values(field: str) -> None:
    values = {"id": "segment", "text": "Explain it.", "cue": "visual-visible"}
    values[field] = "  "

    with pytest.raises(ValueError, match=field):
        NarrationSegment(**values)


def test_synthesized_narration_must_match_plan_order() -> None:
    plan = NarrationPlan(
        lesson_id="lesson",
        segments=(NarrationSegment("one", "One.", "one-visible"),),
    )
    wrong = SynthesizedSegment(
        id="two",
        cue="two-visible",
        text="Two.",
        audio_path=Path("audio/two.mp3"),
        duration_seconds=1.5,
        sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="plan order"):
        SynthesizedNarration(
            lesson_id="lesson",
            provider="fake",
            model_id="fake-model",
            segments=(wrong,),
            plan=plan,
        )


def test_narration_status_values_are_stable_api_contracts() -> None:
    assert [status.value for status in NarrationStatus] == [
        "not_requested",
        "pending",
        "ready",
        "unavailable",
    ]
