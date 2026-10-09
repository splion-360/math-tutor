"""Compose the shared-adapter generation, rendering, and validation workflow.
Runtime configuration enters through Settings and remains outside core logic."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from fastapi import FastAPI

from math_tutor.api import create_app
from math_tutor.domain import LessonStage
from math_tutor.generation.narration import (
    NARRATION_RESPONSE_FORMAT,
    NARRATION_SYSTEM_PROMPT,
    ModelNarrationPlanner,
)
from math_tutor.generation.pipeline import GeneratedLessonPipeline
from math_tutor.generation.provider import (
    BASE_MODEL,
    SHARED_ADAPTER_MODEL,
    GenerationConfig,
    ModalVllmClient,
    UnavailableModelClient,
)
from math_tutor.jobs import JobStore, LessonService, RenderOutcome
from math_tutor.rendering.elevenlabs import ElevenLabsNarrationProvider
from math_tutor.rendering.manim import DEFAULT_MANIM_IMAGE, DockerManimRenderer
from math_tutor.rendering.media import FfmpegMediaAssembler, probe_audio_duration
from math_tutor.rendering.narration import NarrationOutcomeProcessor
from math_tutor.settings import Settings, get_settings
from math_tutor.validation.media import MediaValidator
from math_tutor.validation.models import RenderedAttempt, ValidationReport
from math_tutor.validation.spatial import SpatialValidationPolicy, SpatialValidator
from math_tutor.validation.suite import ValidatorSuite
from math_tutor.validation.visual import FfmpegFrameSampler, VisualEvidenceValidator, VisualModel
from math_tutor.validation.visual_provider import (
    ModalVisualModelClient,
    UnavailableVisualModelClient,
)

GENERATED_DEMO_PROMPT = """Create a concise visual lesson explaining why the Taylor
series of e^x equals the function. Show the polynomial approximations building from
orders zero through five, label the equation, and keep all objects inside the frame."""
ModelClient: TypeAlias = ModalVllmClient | UnavailableModelClient


def build_app(settings: Settings | None = None) -> FastAPI:
    """Compose the configured shared adapter and validation workflow.

    Args:
        settings: Optional injected runtime configuration.

    Returns:
        Fully composed Math Tutor FastAPI application.
    """
    resolved = settings or get_settings()
    job_store = JobStore()
    visual_api_key = (
        resolved.modal_visual_model_api_key.get_secret_value()
        if resolved.modal_visual_model_api_key is not None
        else ""
    )
    configured_visual_client: ModalVisualModelClient | None = None
    visual_model: VisualModel
    if resolved.modal_visual_model_base_url:
        configured_visual_client = ModalVisualModelClient(
            base_url=resolved.modal_visual_model_base_url,
            api_key=visual_api_key,
            timeout_seconds=resolved.modal_visual_model_timeout_seconds,
        )
        visual_model = configured_visual_client
    else:
        visual_model = UnavailableVisualModelClient()

    def report_validation(attempt: RenderedAttempt, report: ValidationReport) -> None:
        """Record one completed validator axis for frontend polling."""
        if attempt.job_id is not None:
            job_store.mark_validation_axis(attempt.job_id, report.axis_summaries()[0])

    output_validator = ValidatorSuite(
        (
            MediaValidator(),
            SpatialValidator(
                policy=SpatialValidationPolicy(
                    unsafe_margin=resolved.spatial_unsafe_margin,
                    max_width_ratio=resolved.spatial_max_width_ratio,
                    max_height_ratio=resolved.spatial_max_height_ratio,
                    severe_overlap_ratio=resolved.spatial_severe_overlap_ratio,
                    persistent_checkpoints=resolved.spatial_persistent_checkpoints,
                )
            ),
            VisualEvidenceValidator(
                sampler=FfmpegFrameSampler(),
                model=visual_model,
            ),
        ),
        report_callback=report_validation,
    )

    def report_stage(job_id: str, stage: LessonStage) -> None:
        """Record the current workflow stage for frontend polling."""
        job_store.mark_stage(job_id, stage)

    def report_render(job_id: str, _attempt_number: int, outcome: RenderOutcome) -> None:
        """Expose the first successful render while validation continues."""
        job_store.mark_initial_render(job_id, outcome)

    elevenlabs_api_key = (
        resolved.elevenlabs_api_key.get_secret_value()
        if resolved.elevenlabs_api_key is not None
        else ""
    )
    narration_enabled = bool(elevenlabs_api_key)
    base_renderer = DockerManimRenderer(
        artifact_root=resolved.artifact_root,
        scene_path=(Path(__file__).parent / "rendering" / "scenes" / "pythagorean_theorem.py"),
        image=DEFAULT_MANIM_IMAGE,
        timeout_seconds=resolved.render_timeout_seconds,
    )
    modal_api_key = (
        resolved.modal_vllm_api_key.get_secret_value()
        if resolved.modal_vllm_api_key is not None
        else ""
    )
    generation_config = GenerationConfig(model=SHARED_ADAPTER_MODEL)
    model: ModelClient
    if resolved.modal_vllm_base_url:
        model = ModalVllmClient(
            api_key=modal_api_key,
            config=generation_config,
            base_url=resolved.modal_vllm_base_url,
            timeout_seconds=resolved.modal_vllm_timeout_seconds,
        )
        generation_provider = "modal_vllm"
    else:
        model = UnavailableModelClient(
            generation_config,
            "Modal inference endpoint is not configured",
        )
        generation_provider = "unavailable"

    narration_model: ModelClient | None = None
    outcome_processor: NarrationOutcomeProcessor | None = None
    if narration_enabled:
        narration_config = GenerationConfig(
            model=BASE_MODEL,
            max_tokens=512,
            system_prompt=NARRATION_SYSTEM_PROMPT,
            response_format=NARRATION_RESPONSE_FORMAT,
        )
        if resolved.modal_vllm_base_url:
            narration_model = ModalVllmClient(
                api_key=modal_api_key,
                config=narration_config,
                base_url=resolved.modal_vllm_base_url,
                timeout_seconds=resolved.modal_vllm_timeout_seconds,
            )
        else:
            narration_model = UnavailableModelClient(
                narration_config,
                "Modal inference endpoint is not configured",
            )
        outcome_processor = NarrationOutcomeProcessor(
            planner=ModelNarrationPlanner(
                narration_model,
                max_retries=resolved.narration_plan_max_retries,
            ),
            provider=ElevenLabsNarrationProvider(
                api_key=elevenlabs_api_key,
                voice_id=resolved.elevenlabs_voice_id,
                duration_probe=probe_audio_duration,
            ),
            assembler=FfmpegMediaAssembler(
                timeout_seconds=resolved.media_assembly_timeout_seconds,
            ),
            duration_probe=probe_audio_duration,
        )

    pipeline = GeneratedLessonPipeline(
        artifact_root=resolved.artifact_root,
        prompt=GENERATED_DEMO_PROMPT,
        generator=model,
        renderer=base_renderer,
        inference_path="lora_adapter",
        generation_provider=generation_provider,
        validator=output_validator,
        max_repair_attempts=resolved.validation_max_repair_attempts,
        stage_reporter=report_stage,
        render_reporter=report_render,
        outcome_processor=outcome_processor,
        narration_required=narration_enabled,
        captions_required=narration_enabled,
    )

    def close_models() -> None:
        """Close configured generation and visual-model clients."""
        model.close()
        if narration_model is not None:
            narration_model.close()
        if configured_visual_client is not None:
            configured_visual_client.close()

    return create_app(
        LessonService(
            renderer=pipeline,
            store=job_store,
            max_pending_jobs=resolved.max_pending_jobs,
            narration_requested=narration_enabled,
        ),
        model_health=model.health,
        close_model=close_models,
    )


app = build_app()
