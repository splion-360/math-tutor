"""Define stable lesson job and progress domain types.
These records are shared by orchestration, persistence, and the HTTP API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class LessonStatus(StrEnum):
    """Terminal and non-terminal states of a lesson job."""

    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"
    PARTIAL = "partial"
    FAILED = "failed"


class LessonStage(StrEnum):
    """User-visible processing stage for an active lesson job."""

    ACCEPTED = "accepted"
    GENERATING_CODE = "generating_code"
    VALIDATING_CODE = "validating_code"
    RENDERING = "rendering"
    VALIDATING_OUTPUT = "validating_output"
    REPAIRING = "repairing"
    READY = "ready"
    FAILED = "failed"


class NarrationStatus(StrEnum):
    """Availability of optional narration for a lesson job."""

    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    READY = "ready"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class LessonJob:
    """Immutable snapshot of lesson state, artifacts, and diagnostics."""

    id: str
    lesson: str
    status: LessonStatus
    stage: LessonStage
    created_at: datetime
    attempt: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    video_path: str | None = None
    silent_video_path: str | None = None
    captions_path: str | None = None
    narration_status: NarrationStatus = NarrationStatus.NOT_REQUESTED
    explanation: str | None = None
    generated_code: str | None = None
    diagnostics: Mapping[str, object] = MappingProxyType({})
    error: str | None = None
