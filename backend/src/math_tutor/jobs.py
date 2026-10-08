"""Coordinate queued lesson rendering and in-memory job state transitions.
Renderers return bounded outcomes that the service exposes through the API."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Protocol
from uuid import uuid4

from math_tutor.domain import (
    Difficulty,
    LessonJob,
    LessonStage,
    LessonStatus,
    NarrationStatus,
    utc_now,
)

_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_safe_job_id(value: str) -> bool:
    """Return whether a job identifier is safe for paths and container names."""
    return _SAFE_JOB_ID.fullmatch(value) is not None


@dataclass(frozen=True)
class RenderOutcome:
    """Successful rendered media and bounded execution diagnostics."""

    video_path: Path
    renderer: str
    elapsed_seconds: float
    logs: str
    silent_video_path: Path | None = None
    captions_path: Path | None = None
    narration_status: NarrationStatus = NarrationStatus.NOT_REQUESTED
    narration_diagnostics: Mapping[str, object] | None = None
    validation_diagnostics: Mapping[str, object] | None = None
    spatial_trace_path: Path | None = None


class JobExecutionError(RuntimeError):
    """Expected job failure with diagnostics safe for API exposure."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class PartialOutcome:
    """Useful incomplete renderer output that cannot become a ready lesson."""

    renderer: str
    elapsed_seconds: float
    logs: str
    error: str


class JobRenderer(Protocol):
    """Render a configured lesson by job identifier."""

    def render(self, job_id: str) -> RenderOutcome | PartialOutcome:
        """Render one configured lesson job."""
        ...


class PromptRenderer(Protocol):
    """Render one free-form lesson prompt."""

    def render(self, job_id: str, prompt: str) -> RenderOutcome | PartialOutcome:
        """Render a prompt-bound lesson job."""
        ...


class Renderer(Protocol):
    """Render a lesson selected by a service-level lesson value."""

    def render(self, job_id: str, lesson: str) -> RenderOutcome | PartialOutcome:
        """Render one selected lesson job."""
        ...


class DispatchingRenderer:
    """Dispatch known lessons and route free-form prompts to a fallback."""

    def __init__(
        self,
        renderers: Mapping[str, JobRenderer],
        fallback: PromptRenderer | None = None,
    ) -> None:
        self._renderers = dict(renderers)
        self._fallback = fallback

    def render(self, job_id: str, lesson: str) -> RenderOutcome | PartialOutcome:
        """Render through the named renderer or configured prompt fallback."""
        try:
            renderer = self._renderers[lesson]
        except KeyError as error:
            if self._fallback is not None:
                return self._fallback.render(job_id, lesson)
            raise JobExecutionError(f"no renderer configured for lesson '{lesson}'") from error
        return renderer.render(job_id)


class JobNotFoundError(KeyError):
    """Raised when an in-memory lesson job does not exist."""


class RenderQueueFullError(RuntimeError):
    """Raised when the bounded lesson execution capacity is exhausted."""


class JobStore:
    """Store immutable job snapshots and serialize state transitions."""

    def __init__(self) -> None:
        self._jobs: dict[str, LessonJob] = {}
        self._lock = Lock()

    def create(
        self,
        lesson: str,
        *,
        difficulty: Difficulty | None = None,
        narration_requested: bool = False,
        routing_policy: str = "default",
    ) -> LessonJob:
        """Create and retain one queued lesson job."""
        job = LessonJob(
            id=uuid4().hex,
            lesson=lesson,
            status=LessonStatus.QUEUED,
            stage=LessonStage.ROUTING,
            created_at=utc_now(),
            difficulty=difficulty,
            narration_status=(
                NarrationStatus.PENDING if narration_requested else NarrationStatus.NOT_REQUESTED
            ),
            diagnostics=({"routing_policy": routing_policy} if routing_policy != "default" else {}),
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> LessonJob:
        """Return the current job snapshot or raise when absent."""
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as error:
                raise JobNotFoundError(job_id) from error

    def mark_running(self, job_id: str) -> LessonJob:
        """Transition a queued job into running state."""
        return self._mutate(
            job_id,
            lambda job: replace(
                job,
                status=LessonStatus.RUNNING,
                started_at=utc_now(),
            ),
        )

    def mark_stage(self, job_id: str, stage: LessonStage) -> LessonJob:
        """Update the processing stage without changing terminal status."""
        return self._mutate(job_id, lambda job: replace(job, stage=stage))

    def mark_ready(self, job_id: str, outcome: RenderOutcome) -> LessonJob:
        """Publish a successful render as the ready job snapshot."""
        return self._mutate(
            job_id,
            lambda job: replace(
                job,
                status=LessonStatus.READY,
                stage=LessonStage.READY,
                completed_at=utc_now(),
                video_path=str(outcome.video_path),
                silent_video_path=str(outcome.silent_video_path or outcome.video_path),
                captions_path=(
                    str(outcome.captions_path) if outcome.captions_path is not None else None
                ),
                narration_status=outcome.narration_status,
                diagnostics={
                    **dict(job.diagnostics),
                    "renderer": outcome.renderer,
                    "elapsed_seconds": outcome.elapsed_seconds,
                    "logs": outcome.logs,
                    **dict(outcome.narration_diagnostics or {}),
                    **dict(outcome.validation_diagnostics or {}),
                },
            ),
        )

    def mark_failed(self, job_id: str, error: Exception) -> LessonJob:
        """Record a terminal failure and bounded diagnostics."""
        diagnostics = error.diagnostics if isinstance(error, JobExecutionError) else {}
        return self._mutate(
            job_id,
            lambda job: replace(
                job,
                status=LessonStatus.FAILED,
                stage=LessonStage.FAILED,
                completed_at=utc_now(),
                error=str(error),
                narration_status=self._failure_narration_status(
                    job.narration_status,
                    diagnostics,
                ),
                diagnostics={**dict(job.diagnostics), **diagnostics},
            ),
        )

    @staticmethod
    def _failure_narration_status(
        current: NarrationStatus,
        diagnostics: Mapping[str, object],
    ) -> NarrationStatus:
        """Read a completed attempt's bounded narration status on failure.

        Args:
            current: Narration state recorded before the terminal failure.
            diagnostics: Bounded failure diagnostics from the renderer pipeline.

        Returns:
            Recognized attempt narration status, otherwise the existing job status.
        """
        value = diagnostics.get("narration_status")
        if not isinstance(value, str):
            return current
        try:
            return NarrationStatus(value)
        except ValueError:
            return current

    def mark_partial(self, job_id: str, outcome: PartialOutcome) -> LessonJob:
        """Record a useful but incomplete terminal outcome."""
        return self._mutate(
            job_id,
            lambda job: replace(
                job,
                status=LessonStatus.PARTIAL,
                stage=LessonStage.FAILED,
                completed_at=utc_now(),
                error=outcome.error,
                diagnostics={
                    **dict(job.diagnostics),
                    "renderer": outcome.renderer,
                    "elapsed_seconds": outcome.elapsed_seconds,
                    "logs": outcome.logs,
                },
            ),
        )

    def _mutate(
        self,
        job_id: str,
        update: Callable[[LessonJob], LessonJob],
    ) -> LessonJob:
        with self._lock:
            try:
                updated = update(self._jobs[job_id])
            except KeyError as error:
                raise JobNotFoundError(job_id) from error
            self._jobs[job_id] = updated
            return updated


class LessonService:
    """Queue bounded background renders and expose job snapshots."""

    def __init__(
        self,
        renderer: Renderer,
        store: JobStore | None = None,
        executor: ThreadPoolExecutor | None = None,
        max_pending_jobs: int = 8,
        narration_requested: bool | Callable[[str], bool] = False,
        routed_renderers: Mapping[Difficulty, Renderer] | None = None,
    ) -> None:
        if max_pending_jobs <= 0:
            raise ValueError("max_pending_jobs must be positive")
        self._renderer = renderer
        self._store = store or JobStore()
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lesson-render",
        )
        self._capacity = BoundedSemaphore(max_pending_jobs)
        self._narration_requested = narration_requested
        self._routed_renderers = dict(routed_renderers or {})

    def submit(
        self,
        lesson: str,
        *,
        difficulty: Difficulty | None = None,
        routing_policy: str = "default",
    ) -> LessonJob:
        """Create and asynchronously schedule one lesson job."""
        if not self._capacity.acquire(blocking=False):
            raise RenderQueueFullError("render queue is full")
        narration_requested = (
            self._narration_requested(lesson)
            if callable(self._narration_requested)
            else self._narration_requested
        )
        job = self._store.create(
            lesson,
            difficulty=difficulty,
            narration_requested=narration_requested,
            routing_policy=routing_policy,
        )
        try:
            self._executor.submit(self._run, job.id)
        except RuntimeError:
            self._capacity.release()
            raise
        return job

    def get(self, job_id: str) -> LessonJob:
        """Return the current snapshot for one job."""
        return self._store.get(job_id)

    def close(self) -> None:
        """Drain and close the background executor."""
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _run(self, job_id: str) -> None:
        try:
            job = self._store.mark_running(job_id)
            try:
                renderer = (
                    self._renderer
                    if job.difficulty is None
                    else self._routed_renderers.get(job.difficulty, self._renderer)
                )
                outcome = renderer.render(job_id, job.lesson)
            except Exception as error:
                self._store.mark_failed(job_id, error)
            else:
                if isinstance(outcome, PartialOutcome):
                    self._store.mark_partial(job_id, outcome)
                else:
                    self._store.mark_ready(job_id, outcome)
        finally:
            self._capacity.release()
