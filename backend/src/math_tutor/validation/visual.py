"""Validate objective visual evidence in deterministic samples from lesson videos.
Sampling, model transport, and schema translation remain separate testable boundaries."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from math_tutor.validation.models import (
    RenderedAttempt,
    ValidationFinding,
    ValidationReport,
    ValidationStatus,
)

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class FrameSamplingError(RuntimeError):
    """Raised when stable frame evidence cannot be extracted from a video."""


class VisualModelError(RuntimeError):
    """Base error for credential-safe visual-model failures."""


class VisualModelUnavailable(VisualModelError):
    """Raised when the configured visual-model service is unavailable."""


class VisualModelTimedOut(VisualModelError):
    """Raised when the configured visual-model service exceeds its deadline."""


class MalformedVisualModelOutput(ValueError):
    """Raised when visual-model output does not match the constrained schema."""


class UnsupportedVisualClaim(ValueError):
    """Raised when a schema-shaped finding asks for an unsupported judgment."""


@dataclass(frozen=True)
class FrameSample:
    """One retained video frame with timestamp and content digest.

    Args:
        id: Stable identifier used in model findings.
        timestamp_seconds: Position of the frame in the source video.
        path: Retained PNG artifact path.
        sha256: SHA-256 digest of the retained PNG bytes.
    """

    id: str
    timestamp_seconds: float
    path: Path
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe frame provenance."""
        return {
            "id": self.id,
            "timestamp_seconds": self.timestamp_seconds,
            "path": str(self.path.resolve()),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class VisualModelResult:
    """Normalized visual-model response with exact execution provenance.

    Args:
        content: Constrained JSON content returned by the visual model.
        model: Model repository identifier.
        revision: Exact immutable model revision.
        request_id: Provider request identifier, when supplied.
        provider_response: Raw provider response retained only as evidence.
    """

    content: str
    model: str
    revision: str
    request_id: str
    provider_response: str


class FrameSampler(Protocol):
    """Produce retained visual evidence from one rendered attempt."""

    def sample(self, video_path: Path, artifact_dir: Path) -> tuple[FrameSample, ...]:
        """Return retained frame samples in timestamp order."""
        ...


class VisualModel(Protocol):
    """Inspect sampled lesson frames through a separately hosted model."""

    model: str
    revision: str

    def inspect(
        self,
        *,
        prompt: str,
        frames: tuple[FrameSample, ...],
    ) -> VisualModelResult:
        """Return constrained JSON evidence for the supplied frames."""
        ...


class FfmpegFrameSampler:
    """Retain equally spaced PNG frames through ffprobe and ffmpeg."""

    def __init__(
        self,
        *,
        command_runner: CommandRunner = subprocess.run,
        sample_count: int = 4,
        timeout_seconds: float = 15,
    ) -> None:
        """Configure deterministic frame extraction.

        Args:
            command_runner: Subprocess-compatible command executor.
            sample_count: Number of equally spaced frames, from three through five.
            timeout_seconds: Maximum duration for each external command.

        Raises:
            ValueError: If sample count or timeout is outside the supported range.
        """
        if not 3 <= sample_count <= 5:
            raise ValueError("visual validation must sample three to five frames")
        if timeout_seconds <= 0:
            raise ValueError("frame sampling timeout must be positive")
        self._run_command = command_runner
        self._sample_count = sample_count
        self._timeout_seconds = timeout_seconds

    def sample(self, video_path: Path, artifact_dir: Path) -> tuple[FrameSample, ...]:
        """Extract and retain stable timestamped frames for one rendered attempt.

        Args:
            video_path: Rendered lesson video.
            artifact_dir: Immutable attempt directory that will own the samples.

        Returns:
            Ordered retained frame samples.

        Raises:
            FrameSamplingError: If duration inspection or extraction fails.
        """
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise FrameSamplingError("rendered video is missing or empty")
        duration = self._duration(video_path)
        visual_dir = artifact_dir / "visual_validation"
        frames_dir = visual_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=False)
        timestamps = tuple(
            round(duration * index / (self._sample_count + 1), 3)
            for index in range(1, self._sample_count + 1)
        )
        samples: list[FrameSample] = []
        for index, timestamp in enumerate(timestamps, start=1):
            sample_id = f"frame-{index:02d}"
            frame_path = frames_dir / f"{sample_id}.png"
            self._extract(video_path, timestamp, frame_path)
            payload = frame_path.read_bytes()
            if not payload:
                raise FrameSamplingError("ffmpeg produced an empty frame")
            samples.append(
                FrameSample(
                    id=sample_id,
                    timestamp_seconds=timestamp,
                    path=frame_path,
                    sha256=sha256(payload).hexdigest(),
                )
            )
        manifest = {
            "schema_version": "visual-frame-samples.v1",
            "source_video_path": str(video_path.resolve()),
            "source_video_sha256": sha256(video_path.read_bytes()).hexdigest(),
            "sampling": {
                "method": "equal_interval_excluding_endpoints",
                "sample_count": self._sample_count,
            },
            "frames": [sample.to_dict() for sample in samples],
        }
        (visual_dir / "frame_samples.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return tuple(samples)

    def _duration(self, video_path: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = self._run(command)
        try:
            duration = float(result.stdout.strip())
        except ValueError as error:
            raise FrameSamplingError("ffprobe returned an invalid video duration") from error
        if duration <= 0:
            raise FrameSamplingError("video duration must be positive")
        return duration

    def _extract(self, video_path: Path, timestamp: float, frame_path: Path) -> None:
        command = [
            "ffmpeg",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-map_metadata",
            "-1",
            "-y",
            str(frame_path),
        ]
        self._run(command)

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            result = self._run_command(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise FrameSamplingError("frame sampling command failed") from error
        if result.returncode != 0:
            raise FrameSamplingError("frame sampling command rejected the video")
        return result


_RULES: dict[str, tuple[str, str]] = {
    "visible_cropping_or_truncation": (
        "Visible content is cropped or truncated in the sampled frame.",
        "Reposition or resize the affected visual so it is fully visible within the frame.",
    ),
    "element_outside_frame": (
        "A visible element extends outside the video frame.",
        "Move the affected element inside the camera frame with a clear margin.",
    ),
    "missing_requested_visual_element": (
        "An explicitly requested visual element is absent from the sampled evidence.",
        "Add the explicitly requested visual element and keep it visible in the lesson.",
    ),
    "obvious_rendering_corruption": (
        "The sampled frame contains obvious rendering corruption.",
        "Replace or simplify the corrupted visual so every rendered element is intact.",
    ),
    "caption_visible_content_mismatch": (
        "The caption does not correspond to the visible sampled content.",
        "Align the caption text and timing with the visual content shown at that timestamp.",
    ),
    "severe_readability_clutter": (
        "Severe visible clutter prevents individual elements from being read.",
        "Separate or reduce the affected elements until each one can be read independently.",
    ),
}
_REGIONS = {
    "full_frame",
    "top",
    "bottom",
    "left",
    "right",
    "center",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
}


class VisualEvidenceValidator:
    """Translate constrained visual-model evidence into controlled findings."""

    name = "visual_evidence"
    expected_checks: tuple[str, ...] = tuple(_RULES)

    def __init__(self, *, sampler: FrameSampler, model: VisualModel) -> None:
        """Configure independent sampling and visual-model boundaries.

        Args:
            sampler: Deterministic retained-frame sampler.
            model: Separately hosted visual evidence model.
        """
        self._sampler = sampler
        self._model = model

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        """Inspect objective evidence in sampled frames.

        Args:
            attempt: Rendered lesson attempt to inspect.

        Returns:
            Pass, fail, uncertain, or validator-error report. Only locally defined
            repair instructions are included in repairable findings.
        """
        try:
            frames = self._sampler.sample(
                attempt.outcome.video_path,
                attempt.artifact_dir,
            )
        except FrameSamplingError:
            report = self._error_report(
                "visual_frame_sampling_failed",
                "Visual frame evidence could not be sampled.",
            )
            self._write_validation(attempt, report, result=None)
            return report

        try:
            result = self._model.inspect(
                prompt=self._inspection_prompt(attempt),
                frames=frames,
            )
        except VisualModelTimedOut:
            report = self._error_report(
                "visual_model_timed_out",
                "Visual evidence validation exceeded its configured deadline.",
            )
            self._write_validation(attempt, report, result=None)
            return report
        except VisualModelUnavailable:
            report = self._error_report(
                "visual_model_unavailable",
                "Visual evidence validation is currently unavailable.",
            )
            self._write_validation(attempt, report, result=None)
            return report
        except VisualModelError:
            report = self._error_report(
                "visual_model_failed",
                "Visual evidence validation could not be completed.",
            )
            self._write_validation(attempt, report, result=None)
            return report

        self._retain_model_response(attempt, result)
        try:
            report = self._parse_report(result.content, frames)
        except UnsupportedVisualClaim:
            report = ValidationReport(
                validator=self.name,
                status=ValidationStatus.UNCERTAIN,
                findings=(
                    ValidationFinding(
                        code="unsupported_visual_claim",
                        message=(
                            "The visual model requested a judgment outside the supported rules."
                        ),
                    ),
                ),
            )
        except MalformedVisualModelOutput:
            report = self._error_report(
                "malformed_visual_model_output",
                "Visual evidence validation returned an invalid structured response.",
            )
        self._write_validation(attempt, report, result=result)
        return report

    @staticmethod
    def _inspection_prompt(attempt: RenderedAttempt) -> str:
        original_prompt = attempt.prompt
        if attempt.original_prompt_path is not None:
            with suppress(OSError):
                original_prompt = attempt.original_prompt_path.read_text(encoding="utf-8")
        captions = ""
        if attempt.outcome.captions_path is not None:
            with suppress(OSError):
                captions = attempt.outcome.captions_path.read_text(encoding="utf-8")[:12_000]
        return (
            "Inspect only the supplied timestamped frames. Return the required JSON schema. "
            "Use only the permitted rule vocabulary. Do not judge pedagogy, mathematical "
            "correctness, or overall lesson coherence. Return uncertain when the samples do "
            "not support a conclusion.\n\n"
            f"Original request:\n{original_prompt}\n\nCaptions:\n{captions or '[none]'}"
        )

    def _parse_report(
        self,
        content: str,
        frames: tuple[FrameSample, ...],
    ) -> ValidationReport:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise MalformedVisualModelOutput from error
        if not isinstance(payload, dict) or set(payload) != {"status", "findings"}:
            raise MalformedVisualModelOutput
        status = payload["status"]
        raw_findings = payload["findings"]
        if status not in {"pass", "fail", "uncertain"} or not isinstance(raw_findings, list):
            raise MalformedVisualModelOutput
        if len(raw_findings) > 10:
            raise MalformedVisualModelOutput
        if status == "pass":
            if raw_findings:
                raise MalformedVisualModelOutput
            return ValidationReport(self.name, ValidationStatus.PASS)
        if status == "uncertain":
            if raw_findings:
                raise MalformedVisualModelOutput
            return ValidationReport(
                validator=self.name,
                status=ValidationStatus.UNCERTAIN,
                findings=(
                    ValidationFinding(
                        code="visual_evidence_uncertain",
                        message="Sampled frames do not support a conclusive visual judgment.",
                    ),
                ),
            )
        if not raw_findings:
            raise MalformedVisualModelOutput
        by_id = {frame.id: frame for frame in frames}
        findings = tuple(self._parse_finding(value, by_id) for value in raw_findings)
        return ValidationReport(
            validator=self.name,
            status=ValidationStatus.FAIL,
            findings=findings,
        )

    @staticmethod
    def _parse_finding(
        value: object,
        frames: dict[str, FrameSample],
    ) -> ValidationFinding:
        if not isinstance(value, dict) or set(value) != {
            "rule",
            "frame_ids",
            "regions",
        }:
            raise MalformedVisualModelOutput
        rule = value["rule"]
        if not isinstance(rule, str):
            raise MalformedVisualModelOutput
        if rule not in _RULES:
            raise UnsupportedVisualClaim
        frame_ids = value["frame_ids"]
        regions = value["regions"]
        if (
            not isinstance(frame_ids, list)
            or not frame_ids
            or len(frame_ids) > 5
            or any(not isinstance(item, str) or item not in frames for item in frame_ids)
            or len(set(frame_ids)) != len(frame_ids)
        ):
            raise MalformedVisualModelOutput
        if (
            not isinstance(regions, list)
            or not regions
            or len(regions) > 4
            or any(not isinstance(item, str) or item not in _REGIONS for item in regions)
            or len(set(regions)) != len(regions)
        ):
            raise MalformedVisualModelOutput
        message, repair_instruction = _RULES[rule]
        return ValidationFinding(
            code=rule,
            message=message,
            evidence={
                "frames": [
                    {
                        "frame_id": frame_id,
                        "timestamp_seconds": frames[frame_id].timestamp_seconds,
                        "sha256": frames[frame_id].sha256,
                    }
                    for frame_id in frame_ids
                ],
                "regions": regions,
            },
            repair_instruction=repair_instruction,
        )

    def _write_validation(
        self,
        attempt: RenderedAttempt,
        report: ValidationReport,
        *,
        result: VisualModelResult | None,
    ) -> None:
        visual_dir = attempt.artifact_dir / "visual_validation"
        visual_dir.mkdir(parents=True, exist_ok=True)
        model_response = visual_dir / "model_response.txt"
        provider_response = visual_dir / "provider_response.json"
        payload: dict[str, object] = {
            "schema_version": "visual-validation.v1",
            "report": report.to_dict(),
            "publication_allowed": report.status is ValidationStatus.PASS,
            "provenance": {
                "model": result.model if result is not None else self._model.model,
                "model_revision": (result.revision if result is not None else self._model.revision),
                "request_id": result.request_id if result is not None else None,
            },
            "artifacts": {
                "frame_samples": self._optional_digest(visual_dir / "frame_samples.json"),
                "model_response_sha256": (
                    sha256(model_response.read_bytes()).hexdigest()
                    if model_response.is_file()
                    else None
                ),
                "provider_response_sha256": (
                    sha256(provider_response.read_bytes()).hexdigest()
                    if provider_response.is_file()
                    else None
                ),
            },
        }
        (visual_dir / "validation.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _retain_model_response(
        attempt: RenderedAttempt,
        result: VisualModelResult,
    ) -> None:
        visual_dir = attempt.artifact_dir / "visual_validation"
        visual_dir.mkdir(parents=True, exist_ok=True)
        (visual_dir / "model_response.txt").write_text(
            result.content,
            encoding="utf-8",
        )
        (visual_dir / "provider_response.json").write_text(
            result.provider_response,
            encoding="utf-8",
        )

    @staticmethod
    def _optional_digest(path: Path) -> str | None:
        return sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    def _error_report(self, code: str, message: str) -> ValidationReport:
        return ValidationReport(
            validator=self.name,
            status=ValidationStatus.VALIDATOR_ERROR,
            findings=(ValidationFinding(code=code, message=message),),
        )
