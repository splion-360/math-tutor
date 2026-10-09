"""Expose lesson creation, status, health, and media endpoints through FastAPI.
The HTTP layer translates service values and expected failures into API responses."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from math_tutor.domain import LessonJob, LessonStage, LessonStatus, NarrationStatus
from math_tutor.generation.provider import SHARED_ADAPTER_MODEL, ModelHealth
from math_tutor.jobs import JobNotFoundError, LessonService, RenderQueueFullError


class CreateLessonRequest(BaseModel):
    """Request containing one free-form math lesson prompt."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def normalize_prompt(self) -> CreateLessonRequest:
        """Strip surrounding whitespace and reject a blank prompt."""
        self.prompt = self.prompt.strip()
        if not self.prompt:
            raise ValueError("prompt must not be blank")
        return self


class ModelHealthResponse(BaseModel):
    """Public provider reachability response."""

    reachable: bool
    model: str
    model_available: bool
    error: str | None


class LessonResponse(BaseModel):
    """Public lesson state and available artifact URLs."""

    id: str
    lesson: str
    status: LessonStatus
    stage: LessonStage
    created_at: datetime
    attempt: int
    started_at: datetime | None
    completed_at: datetime | None
    video_url: str | None
    silent_video_url: str | None
    captions_url: str | None
    narration_status: NarrationStatus
    explanation: str | None
    generated_code: str | None
    diagnostics: dict[str, object]
    error: str | None


def to_response(job: LessonJob) -> LessonResponse:
    """Convert an internal job snapshot into its bounded API representation."""
    video_url = f"/lessons/{job.id}/video" if job.video_path else None
    silent_video_url = f"/lessons/{job.id}/video/silent" if job.silent_video_path else None
    captions_url = f"/lessons/{job.id}/captions" if job.captions_path else None
    public_error = job.error
    if job.status is LessonStatus.FAILED:
        public_error = "We couldn't generate this lesson. Please try again."
    elif job.status is LessonStatus.PARTIAL:
        public_error = "Some lesson assets could not be generated."
    return LessonResponse(
        id=job.id,
        lesson=job.lesson,
        status=job.status,
        stage=job.stage,
        created_at=job.created_at,
        attempt=job.attempt,
        started_at=job.started_at,
        completed_at=job.completed_at,
        video_url=video_url,
        silent_video_url=silent_video_url,
        captions_url=captions_url,
        narration_status=job.narration_status,
        explanation=job.explanation,
        generated_code=job.generated_code,
        diagnostics=dict(job.diagnostics),
        error=public_error,
    )


def create_app(
    service: LessonService,
    *,
    model_health: Callable[[], ModelHealth] | None = None,
    close_model: Callable[[], None] | None = None,
) -> FastAPI:
    """Create the HTTP application around a configured lesson service.

    Args:
        service: Lesson application service used by all endpoints.
        model_health: Optional provider health callback.
        close_model: Optional provider cleanup callback.

    Returns:
        Configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Close service and model resources during application shutdown."""
        yield
        await run_in_threadpool(service.close)
        if close_model is not None:
            await run_in_threadpool(close_model)

    app = FastAPI(title="Math Tutor API", version="0.1.0", lifespan=lifespan)

    @app.get("/model/health", response_model=ModelHealthResponse)
    def get_model_health() -> ModelHealthResponse:
        """Return reachability for the configured generation model."""
        health = (
            model_health()
            if model_health is not None
            else ModelHealth(
                reachable=False,
                model=SHARED_ADAPTER_MODEL,
                model_available=False,
                error="model provider is not configured",
            )
        )
        return ModelHealthResponse(
            reachable=health.reachable,
            model=health.model,
            model_available=health.model_available,
            error=health.error,
        )

    @app.post(
        "/lessons",
        response_model=LessonResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def submit_lesson(request: CreateLessonRequest) -> LessonResponse:
        """Queue a prompted lesson for asynchronous rendering."""
        try:
            return to_response(service.submit(request.prompt))
        except RenderQueueFullError as error:
            raise HTTPException(
                status_code=503,
                detail="render queue is full",
                headers={"Retry-After": "1"},
            ) from error

    @app.get("/lessons/{job_id}", response_model=LessonResponse)
    def get_lesson(job_id: str) -> LessonResponse:
        """Return the latest snapshot for one lesson job."""
        try:
            return to_response(service.get(job_id))
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="lesson job not found") from error

    @app.get("/lessons/{job_id}/video", response_class=FileResponse)
    def get_video(job_id: str) -> FileResponse:
        """Return the accepted lesson video when ready."""
        try:
            job = service.get(job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="lesson job not found") from error
        if job.status is not LessonStatus.READY or job.video_path is None:
            raise HTTPException(status_code=409, detail="lesson video is not ready")
        video_path = Path(job.video_path)
        if not video_path.is_file():
            raise HTTPException(status_code=410, detail="lesson video is unavailable")
        return FileResponse(video_path, media_type="video/mp4", filename="lesson.mp4")

    @app.get("/lessons/{job_id}/video/silent", response_class=FileResponse)
    def get_silent_video(job_id: str) -> FileResponse:
        """Return the silent lesson video when available."""
        job = _ready_job(service, job_id)
        if job.silent_video_path is None:
            raise HTTPException(status_code=409, detail="silent lesson video is not ready")
        video_path = Path(job.silent_video_path)
        if not video_path.is_file():
            raise HTTPException(status_code=410, detail="silent lesson video is unavailable")
        return FileResponse(video_path, media_type="video/mp4", filename="lesson-silent.mp4")

    @app.get("/lessons/{job_id}/captions", response_class=FileResponse)
    def get_captions(job_id: str) -> FileResponse:
        """Return WebVTT captions when available."""
        job = _ready_job(service, job_id)
        if job.captions_path is None:
            raise HTTPException(status_code=409, detail="lesson captions are not ready")
        captions_path = Path(job.captions_path)
        if not captions_path.is_file():
            raise HTTPException(status_code=410, detail="lesson captions are unavailable")
        return FileResponse(captions_path, media_type="text/vtt", filename="lesson.vtt")

    return app


def _ready_job(service: LessonService, job_id: str) -> LessonJob:
    try:
        job = service.get(job_id)
    except JobNotFoundError as error:
        raise HTTPException(status_code=404, detail="lesson job not found") from error
    if job.status is not LessonStatus.READY:
        raise HTTPException(status_code=409, detail="lesson is not ready")
    return job
