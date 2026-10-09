"""Verify narration value contracts and immutable normalized metadata.
The tests cover plan ordering, identifiers, durations, and digests."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from math_tutor.jobs import JobExecutionError, RenderOutcome
from math_tutor.rendering.media import MediaAssemblyError
from math_tutor.rendering.narration import (
    MediaBundle,
    NarratedSourceRenderer,
    NarrationOutcomeProcessor,
    NarrationPlan,
    NarrationSegment,
    NarrationStatus,
    PlannedNarration,
    SynthesizedNarration,
    SynthesizedSegment,
)


def test_outcome_processor_attaches_narration_and_persists_model_evidence(
    tmp_path: Path,
) -> None:
    plan = NarrationPlan(
        lesson_id="lesson-1",
        segments=(
            NarrationSegment("segment-01", "Start at the curve.", "visual-sequence-01"),
            NarrationSegment("segment-02", "Move downhill.", "visual-sequence-02"),
        ),
    )
    planned = PlannedNarration(
        plan=plan,
        model="base-model",
        raw_response='{"segments":[]}',
        provider_response='{"id":"request-1"}',
    )
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"silent")
    observed: dict[str, object] = {}

    class Planner:
        def create_plan(self, **kwargs: object) -> PlannedNarration:
            observed["plan_request"] = kwargs
            return planned

    class Provider:
        def synthesize(
            self,
            received_plan: NarrationPlan,
            output_dir: Path,
        ) -> SynthesizedNarration:
            observed["synthesis"] = (received_plan, output_dir)
            return SynthesizedNarration(
                lesson_id=received_plan.lesson_id,
                provider="elevenlabs",
                model_id="speech-model",
                segments=tuple(
                    SynthesizedSegment(
                        id=segment.id,
                        cue=segment.cue,
                        text=segment.text,
                        audio_path=output_dir / f"{segment.id}.mp3",
                        duration_seconds=2.0,
                        sha256=str(index) * 64,
                    )
                    for index, segment in enumerate(received_plan.segments, start=1)
                ),
                plan=received_plan,
            )

    class Assembler:
        def assemble(self, **kwargs: object) -> MediaBundle:
            observed["assembly"] = kwargs
            output_dir = kwargs["output_dir"]
            assert isinstance(output_dir, Path)
            return MediaBundle(
                silent_video_path=video,
                video_path=output_dir / "narrated.mp4",
                captions_path=output_dir / "captions.vtt",
                narration_status=NarrationStatus.READY,
                diagnostics=MappingProxyType({"narration_provider": "elevenlabs"}),
            )

    attempt_dir = tmp_path / "attempt"
    outcome = NarrationOutcomeProcessor(
        planner=Planner(),
        provider=Provider(),
        assembler=Assembler(),
        duration_probe=lambda _path: 12.5,
    ).process(
        job_id="lesson-1",
        prompt="Explain gradient descent.",
        source="class GradientScene(Scene): pass",
        outcome=RenderOutcome(video, "docker", 1.0, "rendered"),
        artifact_dir=attempt_dir,
    )

    assert observed["plan_request"] == {
        "lesson_id": "lesson-1",
        "prompt": "Explain gradient descent.",
        "source": "class GradientScene(Scene): pass",
        "target_duration_seconds": 12.5,
        "artifact_dir": attempt_dir,
    }
    assert outcome.narration_status is NarrationStatus.READY
    assert outcome.silent_video_path == video
    assert outcome.narration_diagnostics == {
        "narration_provider": "elevenlabs",
        "narration_plan_model": "base-model",
        "narration_plan_attempt_count": 1,
        "narration_target_duration_seconds": 12.5,
    }
    saved_plan = json.loads((attempt_dir / "narration-plan.json").read_text())
    assert saved_plan["model"] == "base-model"
    assert [segment["text"] for segment in saved_plan["segments"]] == [
        "Start at the curve.",
        "Move downhill.",
    ]


def test_outcome_processor_marks_narration_unavailable_on_failure(tmp_path: Path) -> None:
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"silent")

    class FailingPlanner:
        def create_plan(self, **_kwargs: object) -> PlannedNarration:
            raise RuntimeError("private provider failure")

    class UnusedProvider:
        def synthesize(
            self,
            _plan: NarrationPlan,
            _output_dir: Path,
        ) -> SynthesizedNarration:
            raise AssertionError("synthesis must not run")

    class UnusedAssembler:
        def assemble(self, **_kwargs: object) -> MediaBundle:
            raise AssertionError("assembly must not run")

    with pytest.raises(JobExecutionError) as caught:
        NarrationOutcomeProcessor(
            planner=FailingPlanner(),
            provider=UnusedProvider(),
            assembler=UnusedAssembler(),
            duration_probe=lambda _path: 5.0,
        ).process(
            job_id="lesson-1",
            prompt="Explain limits.",
            source="class GeneratedLesson(Scene): pass",
            outcome=RenderOutcome(video, "docker", 1.0, "rendered"),
            artifact_dir=tmp_path / "attempt",
        )

    assert caught.value.diagnostics == {
        "failure_stage": "narration",
        "failure_kind": "operational",
        "narration_status": "unavailable",
    }
    assert "private provider failure" not in str(caught.value)


def test_outcome_processor_reports_media_assembly_failure(tmp_path: Path) -> None:
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"silent")
    plan = NarrationPlan(
        lesson_id="lesson-1",
        segments=(NarrationSegment("segment-01", "Explain it.", "visual-01"),),
    )
    planned = PlannedNarration(plan, "base-model", "{}", "{}")

    class Planner:
        def create_plan(self, **_kwargs: object) -> PlannedNarration:
            return planned

    class Provider:
        def synthesize(
            self,
            received_plan: NarrationPlan,
            output_dir: Path,
        ) -> SynthesizedNarration:
            return SynthesizedNarration(
                lesson_id=received_plan.lesson_id,
                provider="elevenlabs",
                model_id="speech-model",
                segments=(
                    SynthesizedSegment(
                        "segment-01",
                        "visual-01",
                        "Explain it.",
                        output_dir / "segment.mp3",
                        1.0,
                        "a" * 64,
                    ),
                ),
                plan=received_plan,
            )

    class FailingAssembler:
        def assemble(self, **_kwargs: object) -> MediaBundle:
            raise MediaAssemblyError("video muxing failed")

    with pytest.raises(JobExecutionError) as caught:
        NarrationOutcomeProcessor(
            planner=Planner(),
            provider=Provider(),
            assembler=FailingAssembler(),
            duration_probe=lambda _path: 5.0,
        ).process(
            job_id="lesson-1",
            prompt="Explain limits.",
            source="class GeneratedLesson(Scene): pass",
            outcome=RenderOutcome(video, "docker", 1.0, "rendered"),
            artifact_dir=tmp_path / "attempt",
        )

    assert caught.value.diagnostics == {
        "failure_stage": "narration",
        "failure_kind": "operational",
        "narration_error_code": "narration_media_assembly_failed",
        "narration_error": "video muxing failed",
        "narration_status": "unavailable",
    }


def test_outcome_processor_classifies_artifact_directory_failure(tmp_path: Path) -> None:
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"silent")
    artifact_file = tmp_path / "attempt"
    artifact_file.write_text("not a directory")

    class UnusedPlanner:
        def create_plan(self, **_kwargs: object) -> PlannedNarration:
            raise AssertionError("planner must not run")

    class UnusedProvider:
        def synthesize(
            self,
            _plan: NarrationPlan,
            _output_dir: Path,
        ) -> SynthesizedNarration:
            raise AssertionError("provider must not run")

    class UnusedAssembler:
        def assemble(self, **_kwargs: object) -> MediaBundle:
            raise AssertionError("assembler must not run")

    with pytest.raises(JobExecutionError) as caught:
        NarrationOutcomeProcessor(
            planner=UnusedPlanner(),
            provider=UnusedProvider(),
            assembler=UnusedAssembler(),
            duration_probe=lambda _path: 5.0,
        ).process(
            job_id="lesson-1",
            prompt="Explain limits.",
            source="class GeneratedLesson(Scene): pass",
            outcome=RenderOutcome(video, "docker", 1.0, "rendered"),
            artifact_dir=artifact_file,
        )

    assert caught.value.diagnostics["failure_stage"] == "narration"
    assert caught.value.diagnostics["failure_kind"] == "operational"
    assert caught.value.diagnostics["narration_status"] == "unavailable"


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
