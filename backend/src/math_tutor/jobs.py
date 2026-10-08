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

from math_tutor.domain import Difficulty, LessonJob, LessonStage, LessonStatus, utc_now
from math_tutor.narration import NarrationStatus

_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_safe_job_id(value: str) -> bool:
    return _SAFE_JOB_ID.fullmatch(value) is not None


@dataclass(frozen=True)
class RenderOutcome:
    video_path: Path
    renderer: str
    elapsed_seconds: float
    logs: str
    silent_video_path: Path | None = None
    captions_path: Path | None = None
    narration_status: NarrationStatus = NarrationStatus.NOT_REQUESTED
    narration_diagnostics: Mapping[str, object] | None = None
    validation_diagnostics: Mapping[str, object] | None = None


class JobExecutionError(RuntimeError):
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
    renderer: str
    elapsed_seconds: float
    logs: str
    error: str


class JobRenderer(Protocol):
    def render(self, job_id: str) -> RenderOutcome | PartialOutcome: ...


class PromptRenderer(Protocol):
    def render(self, job_id: str, prompt: str) -> RenderOutcome | PartialOutcome: ...


class Renderer(Protocol):
    def render(self, job_id: str, lesson: str) -> RenderOutcome | PartialOutcome: ...


class DispatchingRenderer:
    def __init__(
        self,
        renderers: Mapping[str, JobRenderer],
        fallback: PromptRenderer | None = None,
    ) -> None:
        self._renderers = dict(renderers)
        self._fallback = fallback

    def render(self, job_id: str, lesson: str) -> RenderOutcome | PartialOutcome:
        try:
            renderer = self._renderers[lesson]
        except KeyError as error:
            if self._fallback is not None:
                return self._fallback.render(job_id, lesson)
            raise JobExecutionError(f"no renderer configured for lesson '{lesson}'") from error
        return renderer.render(job_id)


class JobNotFoundError(KeyError):
    pass


class RenderQueueFullError(RuntimeError):
    pass


class JobStore:
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
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as error:
                raise JobNotFoundError(job_id) from error

    def mark_running(self, job_id: str) -> LessonJob:
        return self._mutate(
            job_id,
            lambda job: replace(
                job,
                status=LessonStatus.RUNNING,
                started_at=utc_now(),
            ),
        )

    def mark_stage(self, job_id: str, stage: LessonStage) -> LessonJob:
        return self._mutate(job_id, lambda job: replace(job, stage=stage))

    def mark_ready(self, job_id: str, outcome: RenderOutcome) -> LessonJob:
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
        diagnostics = error.diagnostics if isinstance(error, JobExecutionError) else {}
        return self._mutate(
            job_id,
            lambda job: replace(
                job,
                status=LessonStatus.FAILED,
                stage=LessonStage.FAILED,
                completed_at=utc_now(),
                error=str(error),
                diagnostics={**dict(job.diagnostics), **diagnostics},
            ),
        )

    def mark_partial(self, job_id: str, outcome: PartialOutcome) -> LessonJob:
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
        return self._store.get(job_id)

    def close(self) -> None:
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
