"""Verify deterministic validation of rendered lesson media.
Tests cover normalized probes, delivery requirements, and bounded failures."""

from __future__ import annotations

import subprocess
from pathlib import Path

from math_tutor.jobs import RenderOutcome
from math_tutor.validation.media import (
    FfprobeMediaInspector,
    MediaInspection,
    MediaInspectionError,
    MediaValidationPolicy,
    MediaValidator,
)
from math_tutor.validation.models import RenderedAttempt, ValidationStatus


class FixedInspector:
    def __init__(self, inspection: MediaInspection | Exception) -> None:
        self._inspection = inspection

    def inspect(self, video_path: Path) -> MediaInspection:
        if isinstance(self._inspection, Exception):
            raise self._inspection
        return self._inspection


def _attempt(
    tmp_path: Path,
    inspection: MediaInspection,
    *,
    narration_required: bool = False,
    captions_required: bool = False,
    captions: str | None = None,
) -> tuple[RenderedAttempt, MediaValidator]:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"media")
    captions_path = tmp_path / "captions.vtt" if captions is not None else None
    if captions_path is not None:
        captions_path.write_text(captions, encoding="utf-8")
    attempt = RenderedAttempt(
        number=0,
        artifact_dir=tmp_path,
        prompt="Explain a circle.",
        source="from manim import *",
        scene_class="GeneratedLesson",
        outcome=RenderOutcome(
            video_path=video,
            renderer="test",
            elapsed_seconds=1,
            logs="rendered",
            captions_path=captions_path,
        ),
        narration_required=narration_required,
        captions_required=captions_required,
    )
    validator = MediaValidator(
        inspector=FixedInspector(inspection),
        policy=MediaValidationPolicy(
            target_min_duration_seconds=30,
            target_max_duration_seconds=45,
        ),
    )
    return attempt, validator


def test_media_validator_passes_complete_narrated_media(tmp_path: Path) -> None:
    attempt, validator = _attempt(
        tmp_path,
        MediaInspection(
            video_stream_count=1,
            audio_stream_count=1,
            video_duration_seconds=36,
            audio_duration_seconds=35.5,
        ),
        narration_required=True,
        captions_required=True,
        captions="WEBVTT\n\n00:00.000 --> 00:02.000\nDraw a circle.\n",
    )

    report = validator.validate(attempt)

    assert report.status is ValidationStatus.PASS
    assert report.findings == ()
    assert report.advisories == ()


def test_media_validator_reports_duration_advisory_with_blocking_audio_finding(
    tmp_path: Path,
) -> None:
    attempt, validator = _attempt(
        tmp_path,
        MediaInspection(
            video_stream_count=1,
            audio_stream_count=0,
            video_duration_seconds=7,
            audio_duration_seconds=None,
        ),
        narration_required=True,
    )

    report = validator.validate(attempt)

    assert report.status is ValidationStatus.FAIL
    assert [finding.code for finding in report.findings] == ["missing_required_audio"]
    assert report.findings[0].repair_instruction is not None
    assert [advisory.code for advisory in report.advisories] == [
        "duration_outside_target"
    ]
    assert report.advisories[0].repair_instruction is None
    assert report.advisories[0].evidence == {
        "actual_seconds": 7,
        "target_minimum_seconds": 30,
        "target_maximum_seconds": 45,
    }


def test_media_validator_accepts_short_video_with_duration_advisory(
    tmp_path: Path,
) -> None:
    attempt, validator = _attempt(
        tmp_path,
        MediaInspection(
            video_stream_count=1,
            audio_stream_count=0,
            video_duration_seconds=11,
            audio_duration_seconds=None,
        ),
    )

    report = validator.validate(attempt)

    assert report.status is ValidationStatus.PASS
    assert report.findings == ()
    assert [advisory.code for advisory in report.advisories] == [
        "duration_outside_target"
    ]


def test_media_validator_rejects_missing_required_captions_without_model_repair(
    tmp_path: Path,
) -> None:
    attempt, validator = _attempt(
        tmp_path,
        MediaInspection(
            video_stream_count=1,
            audio_stream_count=1,
            video_duration_seconds=36,
            audio_duration_seconds=36,
        ),
        narration_required=True,
        captions_required=True,
    )

    report = validator.validate(attempt)

    assert report.status is ValidationStatus.FAIL
    assert report.findings[0].code == "missing_required_captions"
    assert report.findings[0].repair_instruction is None


def test_media_validator_reports_material_audio_video_duration_mismatch(
    tmp_path: Path,
) -> None:
    attempt, validator = _attempt(
        tmp_path,
        MediaInspection(
            video_stream_count=1,
            audio_stream_count=1,
            video_duration_seconds=36,
            audio_duration_seconds=32,
        ),
        narration_required=True,
    )

    report = validator.validate(attempt)

    assert report.status is ValidationStatus.FAIL
    assert report.findings[0].code == "audio_video_duration_mismatch"
    assert report.findings[0].evidence["difference_seconds"] == 4


def test_media_validator_reports_inspection_failure_as_validator_error(
    tmp_path: Path,
) -> None:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"media")
    attempt = RenderedAttempt(
        number=0,
        artifact_dir=tmp_path,
        prompt="Explain a circle.",
        source="from manim import *",
        scene_class="GeneratedLesson",
        outcome=RenderOutcome(video, "test", 1, "rendered"),
        narration_required=False,
        captions_required=False,
    )
    validator = MediaValidator(
        inspector=FixedInspector(MediaInspectionError("ffprobe missing"))
    )

    report = validator.validate(attempt)

    assert report.status is ValidationStatus.ERROR
    assert report.findings[0].code == "media_inspection_failed"
    assert "ffprobe missing" not in report.findings[0].message


def test_media_validator_rejects_empty_artifact_without_calling_ffprobe(
    tmp_path: Path,
) -> None:
    video = tmp_path / "empty.mp4"
    video.touch()
    attempt = RenderedAttempt(
        number=0,
        artifact_dir=tmp_path,
        prompt="Explain a circle.",
        source="from manim import *",
        scene_class="GeneratedLesson",
        outcome=RenderOutcome(video, "test", 1, "rendered"),
        narration_required=False,
        captions_required=False,
    )

    report = MediaValidator().validate(attempt)

    assert report.status is ValidationStatus.FAIL
    assert report.findings[0].code == "invalid_media_artifact"


def test_ffprobe_inspector_normalizes_stream_durations(tmp_path: Path) -> None:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"media")

    def run_command(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=(
                '{"streams": [{"codec_type": "video", "duration": "36.0"}, '
                '{"codec_type": "audio", "duration": "35.5"}], '
                '"format": {"duration": "36.0"}}'
            ),
            stderr="",
        )

    inspection = FfprobeMediaInspector(command_runner=run_command).inspect(video)

    assert inspection == MediaInspection(
        video_stream_count=1,
        audio_stream_count=1,
        video_duration_seconds=36,
        audio_duration_seconds=35.5,
    )


def test_ffprobe_inspector_uses_container_duration_for_na_stream_duration(
    tmp_path: Path,
) -> None:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"media")

    def run_command(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=(
                '{"streams": [{"codec_type": "video", "duration": "N/A"}], '
                '"format": {"duration": "36.0"}}'
            ),
            stderr="",
        )

    inspection = FfprobeMediaInspector(command_runner=run_command).inspect(video)

    assert inspection.video_duration_seconds == 36
