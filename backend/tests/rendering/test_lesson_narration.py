"""Verify optional narration orchestration and silent-media fallback behavior.
The tests keep synthesized artifacts and provider failures deterministic."""

from __future__ import annotations

from pathlib import Path

from math_tutor.jobs import RenderOutcome
from math_tutor.rendering.narration import (
    MediaBundle,
    NarratingRenderer,
    NarrationPlan,
    NarrationSegment,
    NarrationStatus,
    SynthesizedNarration,
    SynthesizedSegment,
)


class SilentRenderer:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path

    def render(self, job_id: str) -> RenderOutcome:
        return RenderOutcome(self.video_path, "fake-manim", 1.0, "rendered")


class FakeProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def synthesize(self, plan: NarrationPlan, output_dir: Path) -> SynthesizedNarration:
        if self.fail:
            raise RuntimeError("secret-key must never escape")
        output_dir.mkdir(parents=True)
        audio = output_dir / "000-intro.mp3"
        audio.write_bytes(b"audio")
        segment = plan.segments[0]
        return SynthesizedNarration(
            lesson_id=plan.lesson_id,
            provider="fake",
            model_id="fake-model",
            plan=plan,
            segments=(
                SynthesizedSegment(
                    segment.id,
                    segment.cue,
                    segment.text,
                    audio,
                    1.0,
                    "a" * 64,
                ),
            ),
        )


class FakeAssembler:
    def __init__(self, narrated_video: Path, captions: Path) -> None:
        self.narrated_video = narrated_video
        self.captions = captions

    def assemble(
        self,
        *,
        silent_video: Path,
        narration: SynthesizedNarration,
        output_dir: Path,
    ) -> MediaBundle:
        return MediaBundle(
            silent_video_path=silent_video,
            video_path=self.narrated_video,
            narration_status=NarrationStatus.READY,
            captions_path=self.captions,
            diagnostics={"narration_provider": narration.provider},
        )


def plan() -> NarrationPlan:
    return NarrationPlan(
        lesson_id="pythagorean-theorem",
        segments=(NarrationSegment("intro", "Start.", "triangle-visible"),),
    )


def test_successful_narration_selects_narrated_video(tmp_path: Path) -> None:
    silent = tmp_path / "silent.mp4"
    narrated = tmp_path / "narrated.mp4"
    captions = tmp_path / "captions.vtt"
    renderer = NarratingRenderer(
        renderer=SilentRenderer(silent),
        provider=FakeProvider(),
        assembler=FakeAssembler(narrated, captions),
        plan_factory=plan,
        artifact_root=tmp_path / "artifacts",
    )

    result = renderer.render("job-1")

    assert isinstance(result, RenderOutcome)
    assert result.video_path == narrated
    assert result.silent_video_path == silent
    assert result.captions_path == captions
    assert result.narration_status is NarrationStatus.READY
    assert result.narration_diagnostics == {"narration_provider": "fake"}


def test_narration_failure_preserves_successful_silent_video(tmp_path: Path) -> None:
    silent = tmp_path / "silent.mp4"
    renderer = NarratingRenderer(
        renderer=SilentRenderer(silent),
        provider=FakeProvider(fail=True),
        assembler=FakeAssembler(tmp_path / "narrated.mp4", tmp_path / "captions.vtt"),
        plan_factory=plan,
        artifact_root=tmp_path / "artifacts",
    )

    result = renderer.render("job-1")

    assert isinstance(result, RenderOutcome)
    assert result.video_path == silent
    assert result.silent_video_path == silent
    assert result.narration_status is NarrationStatus.UNAVAILABLE
    assert result.narration_diagnostics == {
        "narration_error": "RuntimeError",
        "narration_error_cause": None,
    }
    assert "secret-key" not in str(result.narration_diagnostics)
