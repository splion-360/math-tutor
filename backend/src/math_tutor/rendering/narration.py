"""Define narration plans and coordinate narration with rendered lesson media.
Provider and assembler protocols keep orchestration independent of external services."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from math_tutor.domain import NarrationStatus
from math_tutor.jobs import JobRenderer, PartialOutcome, RenderOutcome


@dataclass(frozen=True)
class NarrationSegment:
    """One narration line and the visual cue it explains."""

    id: str
    text: str
    cue: str

    def __post_init__(self) -> None:
        for name, value in (("id", self.id), ("text", self.text), ("cue", self.cue)):
            if not value.strip():
                raise ValueError(f"narration segment {name} must not be blank")


@dataclass(frozen=True)
class NarrationPlan:
    """Ordered narration segments for one lesson."""

    lesson_id: str
    segments: tuple[NarrationSegment, ...]
    schema_version: str = field(default="narration-plan.v1", init=False)

    def __post_init__(self) -> None:
        if not self.lesson_id.strip():
            raise ValueError("lesson_id must not be blank")
        if not self.segments:
            raise ValueError("narration plan needs at least one segment")
        segment_ids = [segment.id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("narration segment ids must be unique")


@dataclass(frozen=True)
class SynthesizedSegment:
    """Audio artifact and provenance for one narration segment."""

    id: str
    cue: str
    text: str
    audio_path: Path
    duration_seconds: float
    sha256: str

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if len(self.sha256) != 64:
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        try:
            int(self.sha256, 16)
        except ValueError as error:
            raise ValueError("sha256 must be a 64-character hexadecimal digest") from error


@dataclass(frozen=True)
class SynthesizedNarration:
    """Synthesized narration that preserves the source plan order."""

    lesson_id: str
    provider: str
    model_id: str
    segments: tuple[SynthesizedSegment, ...]
    plan: NarrationPlan = field(repr=False, compare=False)
    schema_version: str = field(default="audio-timeline.v1", init=False)

    def __post_init__(self) -> None:
        if self.lesson_id != self.plan.lesson_id:
            raise ValueError("synthesized narration lesson must match its plan")
        expected = [segment.id for segment in self.plan.segments]
        actual = [segment.id for segment in self.segments]
        if actual != expected:
            raise ValueError("synthesized segments must match plan order")


@dataclass(frozen=True)
class MediaBundle:
    """Assembled lesson media and optional narration artifacts."""

    silent_video_path: Path
    video_path: Path
    narration_status: NarrationStatus
    captions_path: Path | None = None
    timeline_path: Path | None = None
    diagnostics: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )


class NarrationProvider(Protocol):
    """Synthesize the segments in a narration plan."""

    def synthesize(
        self,
        plan: NarrationPlan,
        output_dir: Path,
    ) -> SynthesizedNarration:
        """Create audio artifacts for every segment in a plan."""
        ...


class MediaAssembler(Protocol):
    """Combine a silent render with synthesized narration artifacts."""

    def assemble(
        self,
        *,
        silent_video: Path,
        narration: SynthesizedNarration,
        output_dir: Path,
    ) -> MediaBundle:
        """Combine silent video and narration into a media bundle."""
        ...


class NarratingRenderer:
    """Add optional narration to one renderer with a fixed lesson plan."""

    def __init__(
        self,
        *,
        renderer: JobRenderer,
        provider: NarrationProvider,
        assembler: MediaAssembler,
        plan_factory: Callable[[], NarrationPlan],
        artifact_root: Path,
    ) -> None:
        self._renderer = renderer
        self._provider = provider
        self._assembler = assembler
        self._plan_factory = plan_factory
        self._artifact_root = artifact_root

    def render(self, job_id: str) -> RenderOutcome | PartialOutcome:
        """Render a lesson and attach narration when synthesis succeeds."""
        outcome = self._renderer.render(job_id)
        if isinstance(outcome, PartialOutcome):
            return outcome
        return _attach_narration(
            job_id=job_id,
            outcome=outcome,
            plan=self._plan_factory(),
            provider=self._provider,
            assembler=self._assembler,
            artifact_root=self._artifact_root,
        )


def _attach_narration(
    *,
    job_id: str,
    outcome: RenderOutcome,
    plan: NarrationPlan,
    provider: NarrationProvider,
    assembler: MediaAssembler,
    artifact_root: Path,
) -> RenderOutcome:
    try:
        job_dir = artifact_root / job_id
        narration = provider.synthesize(plan, job_dir / "audio")
        bundle = assembler.assemble(
            silent_video=outcome.video_path,
            narration=narration,
            output_dir=job_dir / "narration",
        )
    except Exception as error:
        return replace(
            outcome,
            silent_video_path=outcome.video_path,
            narration_status=NarrationStatus.UNAVAILABLE,
            narration_diagnostics={
                "narration_error": type(error).__name__,
                "narration_error_cause": (
                    type(error.__cause__).__name__ if error.__cause__ is not None else None
                ),
            },
        )
    return replace(
        outcome,
        video_path=bundle.video_path,
        silent_video_path=bundle.silent_video_path,
        captions_path=bundle.captions_path,
        narration_status=bundle.narration_status,
        narration_diagnostics=bundle.diagnostics,
    )
