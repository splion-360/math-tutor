"""Inspect rendered media for deterministic delivery requirements.
The validator checks streams, duration, synchronization, and caption artifacts."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from math_tutor.validation.models import (
    RenderedAttempt,
    ValidationFinding,
    ValidationReport,
    ValidationStatus,
)

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class MediaInspectionError(RuntimeError):
    """Raised when media metadata cannot be inspected reliably."""


class InvalidMediaArtifactError(RuntimeError):
    """Raised when the rendered file is deterministically missing or invalid."""


class MediaInspector(Protocol):
    """Inspect one rendered media file without changing it."""

    def inspect(self, video_path: Path) -> MediaInspection:
        """Return normalized stream metadata for a rendered media file."""
        ...


@dataclass(frozen=True)
class MediaInspection:
    """Normalized audio and video stream measurements from one media file.

    Args:
        video_stream_count: Number of detected video streams.
        audio_stream_count: Number of detected audio streams.
        video_duration_seconds: Video or container duration when available.
        audio_duration_seconds: Audio or container duration when available.
    """

    video_stream_count: int
    audio_stream_count: int
    video_duration_seconds: float | None
    audio_duration_seconds: float | None


@dataclass(frozen=True)
class MediaValidationPolicy:
    """Thresholds for deterministic media checks.

    Args:
        target_min_duration_seconds: Advisory lower target for lesson duration.
        target_max_duration_seconds: Advisory upper target for lesson duration.
        sync_tolerance_seconds: Maximum allowed audio/video duration difference.
    """

    target_min_duration_seconds: float = 30
    target_max_duration_seconds: float = 45
    sync_tolerance_seconds: float = 1

    def __post_init__(self) -> None:
        if self.target_min_duration_seconds <= 0:
            raise ValueError("minimum duration target must be positive")
        if self.target_max_duration_seconds < self.target_min_duration_seconds:
            raise ValueError("maximum duration target must not be below minimum target")
        if self.sync_tolerance_seconds < 0:
            raise ValueError("sync tolerance must not be negative")


class FfprobeMediaInspector:
    """Read media stream metadata through the installed ffprobe binary."""

    def __init__(
        self,
        *,
        command_runner: CommandRunner = subprocess.run,
        timeout_seconds: float = 15,
    ) -> None:
        """Configure the ffprobe command boundary.

        Args:
            command_runner: Subprocess-compatible command executor.
            timeout_seconds: Maximum duration for one probe.

        Raises:
            ValueError: If the timeout is not positive.
        """
        if timeout_seconds <= 0:
            raise ValueError("media inspection timeout must be positive")
        self._run_command = command_runner
        self._timeout_seconds = timeout_seconds

    def inspect(self, video_path: Path) -> MediaInspection:
        """Inspect one media file.

        Args:
            video_path: Rendered media path to inspect.

        Returns:
            Normalized counts and durations for audio and video streams.

        Raises:
            MediaInspectionError: If the file is empty, missing, or unreadable.
        """
        try:
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise InvalidMediaArtifactError("rendered media is missing or empty")
            result = self._run_command(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration:stream=codec_type,duration",
                    "-of",
                    "json",
                    str(video_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except InvalidMediaArtifactError:
            raise
        except MediaInspectionError:
            raise
        except (OSError, subprocess.SubprocessError) as error:
            raise MediaInspectionError("media inspection command failed") from error
        if result.returncode != 0:
            raise InvalidMediaArtifactError("ffprobe rejected the rendered media")
        try:
            payload = json.loads(result.stdout)
            streams = payload.get("streams", [])
            format_duration = _positive_float(payload.get("format", {}).get("duration"))
            video_durations = _stream_durations(streams, "video")
            audio_durations = _stream_durations(streams, "audio")
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise MediaInspectionError("ffprobe returned invalid metadata") from error
        return MediaInspection(
            video_stream_count=len(video_durations),
            audio_stream_count=len(audio_durations),
            video_duration_seconds=_first_duration(video_durations, format_duration),
            audio_duration_seconds=_first_duration(audio_durations, format_duration),
        )


class MediaValidator:
    """Validate rendered media using deterministic stream and file checks."""

    name = "media"
    expected_checks: tuple[str, ...] = (
        "decodable_video_stream",
        "lesson_duration",
        "required_audio_stream",
        "audio_video_duration_match",
        "required_captions",
    )

    def __init__(
        self,
        *,
        inspector: MediaInspector | None = None,
        policy: MediaValidationPolicy | None = None,
    ) -> None:
        """Configure media inspection and validation thresholds.

        Args:
            inspector: Media metadata provider, defaulting to ffprobe.
            policy: Duration and synchronization thresholds.
        """
        self._inspector = inspector or FfprobeMediaInspector()
        self._policy = policy or MediaValidationPolicy()

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        """Validate media attached to an attempt.

        Args:
            attempt: Rendered attempt with expected narration and caption requirements.

        Returns:
            A structured pass, fail, or validator-error report.
        """
        try:
            inspection = self._inspector.inspect(attempt.outcome.video_path)
        except InvalidMediaArtifactError:
            return ValidationReport(
                validator=self.name,
                status=ValidationStatus.FAIL,
                findings=(
                    ValidationFinding(
                        code="invalid_media_artifact",
                        message="Rendered media is missing, empty, or corrupt.",
                    ),
                ),
            )
        except MediaInspectionError:
            return ValidationReport(
                validator=self.name,
                status=ValidationStatus.ERROR,
                findings=(
                    ValidationFinding(
                        code="media_inspection_failed",
                        message="Rendered media could not be inspected.",
                    ),
                ),
            )

        findings: list[ValidationFinding] = []
        advisories: list[ValidationFinding] = []
        if inspection.video_stream_count != 1:
            findings.append(
                ValidationFinding(
                    code="invalid_video_stream_count",
                    message="Rendered media must contain exactly one video stream.",
                    evidence={"video_stream_count": inspection.video_stream_count},
                )
            )
        duration = inspection.video_duration_seconds
        if duration is None:
            findings.append(
                ValidationFinding(
                    code="missing_video_duration",
                    message="Rendered media does not expose a valid video duration.",
                )
            )
        elif not (
            self._policy.target_min_duration_seconds
            <= duration
            <= self._policy.target_max_duration_seconds
        ):
            advisories.append(
                ValidationFinding(
                    code="duration_outside_target",
                    message="Lesson duration is outside the target range.",
                    evidence={
                        "actual_seconds": duration,
                        "target_minimum_seconds": (
                            self._policy.target_min_duration_seconds
                        ),
                        "target_maximum_seconds": (
                            self._policy.target_max_duration_seconds
                        ),
                    },
                )
            )
        if attempt.narration_required and inspection.audio_stream_count < 1:
            findings.append(
                ValidationFinding(
                    code="missing_required_audio",
                    message="Narration was required but no audio stream was found.",
                    evidence={"audio_stream_count": inspection.audio_stream_count},
                    repair_instruction=(
                        "Include the required narrated voiceover blocks and synchronize each "
                        "animation to its narration tracker."
                    ),
                )
            )
        if (
            inspection.audio_stream_count > 0
            and duration is not None
            and inspection.audio_duration_seconds is not None
        ):
            delta = abs(duration - inspection.audio_duration_seconds)
            if delta > self._policy.sync_tolerance_seconds:
                findings.append(
                    ValidationFinding(
                        code="audio_video_duration_mismatch",
                        message="Audio and video durations differ beyond the configured tolerance.",
                        evidence={
                            "video_seconds": duration,
                            "audio_seconds": inspection.audio_duration_seconds,
                            "difference_seconds": delta,
                            "tolerance_seconds": self._policy.sync_tolerance_seconds,
                        },
                        repair_instruction=(
                            "Align animation timing with narration so the audio and video finish "
                            "within the configured tolerance."
                        ),
                    )
                )
        if attempt.captions_required and not _has_captions(attempt.outcome.captions_path):
            findings.append(
                ValidationFinding(
                    code="missing_required_captions",
                    message="Captions were required but no non-empty WebVTT artifact was found.",
                )
            )
        return ValidationReport(
            validator=self.name,
            status=ValidationStatus.FAIL if findings else ValidationStatus.PASS,
            findings=tuple(findings),
            advisories=tuple(advisories),
        )


def _positive_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError("duration must be numeric")
    try:
        number = float(value)
    except ValueError:
        return None
    return number if number > 0 else None


def _stream_durations(streams: object, codec_type: str) -> list[float | None]:
    if not isinstance(streams, list):
        raise TypeError("streams must be a list")
    return [
        _positive_float(stream.get("duration"))
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == codec_type
    ]


def _first_duration(durations: list[float | None], fallback: float | None) -> float | None:
    return durations[0] if durations and durations[0] is not None else fallback


def _has_captions(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_file() and path.stat().st_size > 0 and path.read_text(
            encoding="utf-8"
        ).lstrip().startswith("WEBVTT")
    except (OSError, UnicodeError):
        return False
