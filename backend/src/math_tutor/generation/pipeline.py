"""Validate generated Manim scenes and coordinate lesson rendering.
Both product lessons and raw adapter outputs share the same safety checks."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Literal, Protocol

from math_tutor.domain import LessonStage, NarrationStatus
from math_tutor.generation.artifacts import AttemptArtifactStore
from math_tutor.generation.errors import (
    GeneratedLessonError,
    OutputValidationError,
)
from math_tutor.generation.provider import GenerationConfig, GenerationResult, ProviderError
from math_tutor.generation.repair import build_repair_prompt
from math_tutor.generation.source import (
    ExtractedScene,
    extract_and_validate_raw_scene,
    extract_and_validate_scene,
)
from math_tutor.jobs import RenderOutcome, is_safe_job_id
from math_tutor.rendering.manim import RenderError, RenderTimedOut
from math_tutor.validation.models import (
    AttemptValidator,
    RenderedAttempt,
    ValidationFinding,
    ValidationReport,
    ValidationStatus,
)


class Generator(Protocol):
    """Generate one model response for a lesson prompt."""

    config: GenerationConfig

    def generate(self, prompt: str) -> GenerationResult:
        """Return a normalized generation result for the prompt."""
        ...


class SourceRenderer(Protocol):
    """Render admitted source into a lesson video."""

    def render_source(self, job_id: str, source: str, scene_class: str) -> RenderOutcome:
        """Render source for one job and return its media outcome."""
        ...


class RenderedOutcomeProcessor(Protocol):
    """Transform a completed render before validation and publication."""

    def process(
        self,
        *,
        job_id: str,
        prompt: str,
        source: str,
        outcome: RenderOutcome,
        artifact_dir: Path,
    ) -> RenderOutcome:
        """Return the media outcome that validators should inspect."""
        ...


class PromptLessonRenderer(Protocol):
    """Render a lesson from a user prompt."""

    def render(self, job_id: str, prompt: str | None = None) -> RenderOutcome:
        """Render one prompt-bound lesson job."""
        ...


InferencePath = Literal["lora_adapter"]


class GeneratedLessonPipeline:
    """Generate, admit, render, validate, and optionally repair one lesson job.

    Args:
        artifact_root: Root directory for immutable job artifacts.
        prompt: Default lesson prompt used when render receives no override.
        generator: Model boundary used for initial generation and repair.
        renderer: Isolated Manim source renderer.
        voiceover: Whether generated source must use the VoiceoverScene contract.
        narration_required: Whether validated media must contain narration.
        captions_required: Whether rendered media must include WebVTT captions.
        outcome_processor: Optional post-render media processor run before validation.
        inference_path: Routing provenance recorded with each attempt.
        generation_provider: Provider provenance recorded with each attempt.
        validator: Independent rendered-attempt validator.
        max_repair_attempts: Maximum model regenerations after the initial attempt.
        stage_reporter: Optional job progress callback.
        render_reporter: Optional callback invoked after each successful render.
        narration_status_override: Optional status recorded for a successful silent render.
    """

    def __init__(
        self,
        *,
        artifact_root: Path,
        prompt: str,
        generator: Generator,
        renderer: SourceRenderer,
        voiceover: bool = False,
        narration_required: bool = False,
        captions_required: bool = False,
        outcome_processor: RenderedOutcomeProcessor | None = None,
        inference_path: InferencePath = "lora_adapter",
        generation_provider: str = "configured_generator",
        validator: AttemptValidator | None = None,
        max_repair_attempts: int = 1,
        stage_reporter: Callable[[str, LessonStage], None] | None = None,
        render_reporter: Callable[[str, int, RenderOutcome], None] | None = None,
        narration_status_override: NarrationStatus | None = None,
    ) -> None:
        if not 0 <= max_repair_attempts <= 1:
            raise ValueError("max_repair_attempts must be zero or one")
        self._artifact_root = artifact_root.resolve()
        self._prompt = prompt
        self._generator = generator
        self._renderer = renderer
        self._voiceover = voiceover
        self._narration_required = narration_required
        self._captions_required = captions_required
        self._outcome_processor = outcome_processor
        self._inference_path = inference_path
        self._generation_provider = generation_provider
        self._validator = validator
        self._max_repair_attempts = max_repair_attempts
        self._stage_reporter = stage_reporter
        self._render_reporter = render_reporter
        self._narration_status_override = narration_status_override

    def render(self, job_id: str, prompt: str | None = None) -> RenderOutcome:
        """Produce the first accepted rendered attempt for a lesson job.

        Args:
            job_id: Safe identifier used for immutable artifact paths.
            prompt: Optional request that overrides the configured demo prompt.

        Returns:
            Render outcome for the accepted attempt.

        Raises:
            GeneratedLessonError: If generation, admission, or output validation fails.
            RenderError: If isolated rendering fails operationally.
        """
        if not is_safe_job_id(job_id):
            raise GeneratedLessonError(
                "job id is not safe for an artifact path",
                diagnostics={"failure_stage": "validation"},
            )
        original_prompt = prompt or self._prompt
        artifacts = AttemptArtifactStore(
            artifact_root=self._artifact_root,
            job_id=job_id,
            original_prompt=original_prompt,
        )
        effective_prompt = original_prompt
        infrastructure_retry_count = 0

        for attempt_number in range(self._max_repair_attempts + 1):
            attempt_dir = artifacts.start_attempt(attempt_number, effective_prompt)
            result, metadata = self._generate_attempt(
                job_id=job_id,
                attempt_number=attempt_number,
                attempt_dir=attempt_dir,
                prompt=effective_prompt,
                artifacts=artifacts,
                infrastructure_retry_count=infrastructure_retry_count,
            )
            self._report_stage(job_id, LessonStage.VALIDATING_CODE)
            try:
                extracted = self._admit_source(
                    attempt_dir=attempt_dir,
                    result=result,
                    metadata=metadata,
                )
            except GeneratedLessonError as error:
                report = self._source_failure_report(error)
                self._write_validation(attempt_dir, report)
                manifest = artifacts.write_failed_attempt_manifest(
                    attempt_dir=attempt_dir,
                    number=attempt_number,
                    status=report.status.value,
                    generation_model=result.model,
                    generation_provider=self._generation_provider,
                    inference_path=self._inference_path,
                    expected_checks=self._expected_checks,
                    failure={
                        "failure_stage": error.diagnostics.get("failure_stage", "validation"),
                        "findings": [finding.to_dict() for finding in report.findings],
                    },
                )
                if attempt_number < self._max_repair_attempts:
                    self._report_stage(job_id, LessonStage.REPAIRING)
                    effective_prompt = build_repair_prompt(
                        original_prompt=original_prompt,
                        previous_output=result.content,
                        report=report,
                    )
                    continue
                error.diagnostics.update(
                    {
                        "attempt_count": attempt_number + 1,
                        "repair_count": attempt_number,
                        "validation_status": report.status.value,
                        "findings": [finding.to_dict() for finding in report.findings],
                        "validation_reports": [report.to_dict()],
                        "attempt_manifest": str(manifest.resolve()),
                        "infrastructure_retry_count": infrastructure_retry_count,
                    }
                )
                raise
            attempt = self._render_attempt(
                artifacts=artifacts,
                job_id=job_id,
                attempt_number=attempt_number,
                attempt_dir=attempt_dir,
                prompt=effective_prompt,
                narration_prompt=original_prompt,
                result=result,
                extracted=extracted,
                previous_infrastructure_retry_count=infrastructure_retry_count,
            )
            infrastructure_retry_count += attempt.infrastructure_retry_count
            attempt = replace(
                attempt,
                validation_input_path=artifacts.write_validation_input(
                    attempt,
                    expected_checks=self._expected_checks,
                ),
            )
            if self._validator is None:
                artifacts.write_attempt_manifest(
                    attempt,
                    expected_checks=self._expected_checks,
                    validation_status="not_run",
                )
                return artifacts.select_attempt(
                    attempt,
                    validation_status="not_run",
                    infrastructure_retry_count=infrastructure_retry_count,
                )

            self._report_stage(job_id, LessonStage.VALIDATING_OUTPUT)
            report = self._validator.validate(attempt)
            self._write_validation(attempt_dir, report)
            manifest = artifacts.write_attempt_manifest(
                attempt,
                expected_checks=self._expected_checks,
                validation_status=report.status.value,
                validation_report=report,
            )
            if report.status is ValidationStatus.PASS:
                return artifacts.select_attempt(
                    attempt,
                    validation_status=report.status.value,
                    infrastructure_retry_count=infrastructure_retry_count,
                    validation_advisories=[advisory.to_dict() for advisory in report.advisories],
                    validation_report=report,
                )
            if report.status is ValidationStatus.VALIDATOR_ERROR:
                raise OutputValidationError(
                    "rendered lesson validation could not be completed",
                    diagnostics=self._validation_diagnostics(
                        attempt=attempt,
                        report=report,
                        infrastructure_retry_count=infrastructure_retry_count,
                        manifest=manifest,
                    ),
                )
            if (
                report.status is ValidationStatus.FAIL
                and report.repairable_findings
                and attempt_number < self._max_repair_attempts
            ):
                self._report_stage(job_id, LessonStage.REPAIRING)
                effective_prompt = build_repair_prompt(
                    original_prompt=original_prompt,
                    previous_output=result.content,
                    report=report,
                )
                continue
            raise OutputValidationError(
                "rendered lesson did not pass output validation",
                diagnostics=self._validation_diagnostics(
                    attempt=attempt,
                    report=report,
                    infrastructure_retry_count=infrastructure_retry_count,
                    manifest=manifest,
                ),
            )

        raise AssertionError("generation attempt loop ended without a result")

    def _generate_attempt(
        self,
        *,
        job_id: str,
        attempt_number: int,
        attempt_dir: Path,
        prompt: str,
        artifacts: AttemptArtifactStore,
        infrastructure_retry_count: int,
    ) -> tuple[GenerationResult, dict[str, object]]:
        """Generate one response and persist provider evidence.

        Args:
            job_id: Parent job used for progress reporting.
            attempt_number: Zero-based generation attempt number.
            attempt_dir: Immutable evidence directory for this attempt.
            prompt: Prompt sent to the configured generator.
            artifacts: Artifact store used to record terminal failures.
            infrastructure_retry_count: Render retries completed by earlier attempts.

        Returns:
            Provider result and mutable generation metadata for source admission.

        Raises:
            GeneratedLessonError: If the provider cannot complete generation.
        """
        self._report_stage(job_id, LessonStage.GENERATING_CODE)
        started = monotonic()
        try:
            result = self._generator.generate(prompt)
        except ProviderError as error:
            metadata = {
                "status": "provider_failed",
                "failure_stage": "provider",
                "model": self._generator.config.model,
                "elapsed_seconds": monotonic() - started,
                "temperature": self._generator.config.temperature,
                "top_p": self._generator.config.top_p,
                "seed": self._generator.config.seed,
            }
            self._write_metadata(attempt_dir, metadata)
            manifest = artifacts.write_failed_attempt_manifest(
                attempt_dir=attempt_dir,
                number=attempt_number,
                status="provider_failed",
                generation_model=self._generator.config.model,
                generation_provider=self._generation_provider,
                inference_path=self._inference_path,
                expected_checks=self._expected_checks,
                failure={"failure_stage": "provider"},
            )
            raise GeneratedLessonError(
                str(error),
                diagnostics={
                    "failure_stage": "provider",
                    "failure_kind": "operational",
                    "attempt_count": attempt_number + 1,
                    "repair_count": attempt_number,
                    "infrastructure_retry_count": infrastructure_retry_count,
                    "attempt_manifest": str(manifest.resolve()),
                },
            ) from error
        (attempt_dir / "raw_response.txt").write_text(result.content, encoding="utf-8")
        (attempt_dir / "provider_response.json").write_text(
            result.provider_response,
            encoding="utf-8",
        )
        metadata = {
            "status": "generated",
            "model": result.model,
            "request_id": result.request_id,
            "finish_reason": result.finish_reason,
            "elapsed_seconds": result.elapsed_seconds,
            "temperature": self._generator.config.temperature,
            "top_p": self._generator.config.top_p,
            "seed": self._generator.config.seed,
            **asdict(result.usage),
        }
        self._write_metadata(attempt_dir, metadata)
        return result, metadata

    def _admit_source(
        self,
        *,
        attempt_dir: Path,
        result: GenerationResult,
        metadata: dict[str, object],
    ) -> ExtractedScene:
        """Run deterministic source admission and persist the admitted scene.

        Args:
            attempt_dir: Evidence directory for the attempt.
            result: Provider generation result to inspect.
            metadata: Generation metadata updated when admission fails.

        Returns:
            Admitted source and scene class.

        Raises:
            GeneratedLessonError: If extraction or source admission fails.
        """
        try:
            extracted = (
                extract_and_validate_scene(result.content, voiceover=True)
                if self._voiceover
                else extract_and_validate_raw_scene(result.content)
            )
        except GeneratedLessonError as error:
            metadata["status"] = f"{error.diagnostics.get('failure_stage', 'validation')}_failed"
            self._write_metadata(attempt_dir, metadata)
            raise
        (attempt_dir / "extracted_scene.py").write_text(
            extracted.source,
            encoding="utf-8",
        )
        return extracted

    def _render_attempt(
        self,
        *,
        artifacts: AttemptArtifactStore,
        job_id: str,
        attempt_number: int,
        attempt_dir: Path,
        prompt: str,
        narration_prompt: str,
        result: GenerationResult,
        extracted: ExtractedScene,
        previous_infrastructure_retry_count: int,
    ) -> RenderedAttempt:
        """Render admitted source and assemble its validation boundary object.

        Args:
            artifacts: Artifact store used to record render failures.
            job_id: Parent job used for progress and renderer isolation.
            attempt_number: Zero-based generation attempt number.
            attempt_dir: Evidence directory for the attempt.
            prompt: Prompt that produced the admitted source.
            narration_prompt: Original lesson request used to describe the rendered video.
            result: Provider result containing model provenance.
            extracted: Admitted source and scene class.
            previous_infrastructure_retry_count: Retries completed by earlier attempts.

        Returns:
            Immutable rendered-attempt description for output validation.

        Raises:
            RenderError: If isolated rendering fails operationally.
        """
        self._report_stage(job_id, LessonStage.RENDERING)
        infrastructure_retry_count = 0
        while True:
            try:
                outcome = self._renderer.render_source(
                    self._attempt_render_id(
                        job_id,
                        attempt_number,
                        infrastructure_retry_count,
                    ),
                    extracted.source,
                    extracted.scene_class,
                )
                break
            except RenderTimedOut as error:
                if infrastructure_retry_count == 0:
                    infrastructure_retry_count = 1
                    continue
                self._raise_render_failure(
                    error=error,
                    artifacts=artifacts,
                    attempt_dir=attempt_dir,
                    attempt_number=attempt_number,
                    result=result,
                    infrastructure_retry_count=infrastructure_retry_count,
                    previous_infrastructure_retry_count=(previous_infrastructure_retry_count),
                )
            except RenderError as error:
                self._raise_render_failure(
                    error=error,
                    artifacts=artifacts,
                    attempt_dir=attempt_dir,
                    attempt_number=attempt_number,
                    result=result,
                    infrastructure_retry_count=infrastructure_retry_count,
                    previous_infrastructure_retry_count=(previous_infrastructure_retry_count),
                )
        outcome = replace(
            outcome,
            narration_status=(
                NarrationStatus.READY
                if self._voiceover
                else self._narration_status_override or outcome.narration_status
            ),
            narration_diagnostics={
                **dict(outcome.narration_diagnostics or {}),
                "inference_path": self._inference_path,
                "inference_model": result.model,
            },
            generated_code=extracted.source,
        )
        if self._render_reporter is not None:
            self._render_reporter(job_id, attempt_number, outcome)
        if self._outcome_processor is not None:
            outcome = self._outcome_processor.process(
                job_id=job_id,
                prompt=narration_prompt,
                source=extracted.source,
                outcome=outcome,
                artifact_dir=attempt_dir,
            )
        (attempt_dir / "render.log").write_text(outcome.logs, encoding="utf-8")
        attempt = RenderedAttempt(
            number=attempt_number,
            artifact_dir=attempt_dir,
            prompt=prompt,
            source=extracted.source,
            scene_class=extracted.scene_class,
            outcome=outcome,
            narration_required=self._narration_required or self._voiceover,
            captions_required=self._captions_required,
            original_prompt_path=artifacts.original_prompt_path,
            generation_model=result.model,
            generation_provider=self._generation_provider,
            inference_path=self._inference_path,
            infrastructure_retry_count=infrastructure_retry_count,
            job_id=job_id,
        )
        return attempt

    def _raise_render_failure(
        self,
        *,
        error: RenderError,
        artifacts: AttemptArtifactStore,
        attempt_dir: Path,
        attempt_number: int,
        result: GenerationResult,
        infrastructure_retry_count: int,
        previous_infrastructure_retry_count: int,
    ) -> None:
        """Record an operational render failure without consuming model repair.

        Args:
            error: Terminal renderer error after any transient retry.
            artifacts: Artifact store used to preserve failure evidence.
            attempt_dir: Evidence directory for this generation attempt.
            attempt_number: Zero-based generation attempt number.
            result: Generation result containing model provenance.
            infrastructure_retry_count: Render retries already attempted.
            previous_infrastructure_retry_count: Retries completed by earlier attempts.

        Raises:
            RenderError: Always re-raises the supplied renderer error.
        """
        manifest = artifacts.write_failed_attempt_manifest(
            attempt_dir=attempt_dir,
            number=attempt_number,
            status="render_failed",
            generation_model=result.model,
            generation_provider=self._generation_provider,
            inference_path=self._inference_path,
            expected_checks=self._expected_checks,
            failure={
                "failure_stage": "render",
                "failure_kind": "operational",
                "infrastructure_retry_count": infrastructure_retry_count,
                "job_infrastructure_retry_count": (
                    previous_infrastructure_retry_count + infrastructure_retry_count
                ),
                **dict(error.diagnostics),
            },
        )
        error.diagnostics.update(
            {
                "failure_stage": "render",
                "failure_kind": "operational",
                "attempt_count": attempt_number + 1,
                "repair_count": attempt_number,
                "infrastructure_retry_count": (
                    previous_infrastructure_retry_count + infrastructure_retry_count
                ),
                "attempt_manifest": str(manifest.resolve()),
            }
        )
        raise error

    def _report_stage(self, job_id: str, stage: LessonStage) -> None:
        if self._stage_reporter is not None:
            self._stage_reporter(job_id, stage)

    def _validation_diagnostics(
        self,
        *,
        attempt: RenderedAttempt,
        report: ValidationReport,
        infrastructure_retry_count: int,
        manifest: Path,
    ) -> dict[str, object]:
        """Build public diagnostics for a rejected rendered attempt.

        Args:
            attempt: Rendered attempt that could not be published.
            report: Aggregate output-validation report.
            infrastructure_retry_count: Total render retries for the lesson job.
            manifest: Final evidence manifest for the rejected attempt.

        Returns:
            Bounded attempt, validation, provenance, and narration diagnostics.
        """
        return {
            "failure_stage": "output_validation",
            "validator": report.validator,
            "validation_status": report.status.value,
            "validation_axes": report.axis_summaries(),
            "validation_reports": [
                component.to_dict() for component in report.component_reports or (report,)
            ],
            "findings": [finding.to_dict() for finding in report.findings],
            "advisories": [advisory.to_dict() for advisory in report.advisories],
            "attempt_count": attempt.number + 1,
            "repair_count": attempt.number,
            "infrastructure_retry_count": infrastructure_retry_count,
            "attempt_manifest": str(manifest.resolve()),
            "generation_provenance": attempt.generation_provenance(),
            "narration_status": attempt.outcome.narration_status.value,
        }

    @staticmethod
    def _write_metadata(job_dir: Path, metadata: dict[str, object]) -> None:
        (job_dir / "generation.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _source_failure_report(error: GeneratedLessonError) -> ValidationReport:
        evidence = {
            key: value
            for key, value in error.diagnostics.items()
            if key in {"failure_stage", "line"}
        }
        return ValidationReport(
            validator="source",
            status=ValidationStatus.FAIL,
            findings=(
                ValidationFinding(
                    code="source_admission_failed",
                    message="Generated source did not pass deterministic admission checks.",
                    evidence=evidence,
                    repair_instruction=(
                        "Return only one complete Manim Python scene that satisfies this "
                        f"admission error: {error}"
                    ),
                ),
            ),
        )

    @staticmethod
    def _write_validation(attempt_dir: Path, report: ValidationReport) -> None:
        (attempt_dir / "validation.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @property
    def _expected_checks(self) -> tuple[str, ...]:
        return self._validator.expected_checks if self._validator is not None else ()

    @staticmethod
    def _attempt_render_id(
        job_id: str,
        attempt_number: int,
        infrastructure_retry_count: int = 0,
    ) -> str:
        """Return a safe, distinct renderer identifier for an attempt execution.

        Args:
            job_id: Parent lesson job identifier.
            attempt_number: Zero-based model attempt number.
            infrastructure_retry_count: Retry number for the same admitted source.

        Returns:
            Renderer identifier no longer than the job-id contract permits.
        """
        retry_suffix = f"-retry-{infrastructure_retry_count}" if infrastructure_retry_count else ""
        suffix = f"-attempt-{attempt_number}{retry_suffix}"
        if len(job_id) + len(suffix) <= 64:
            return f"{job_id}{suffix}"
        digest = sha256(job_id.encode()).hexdigest()[:8]
        prefix_length = 64 - len(suffix) - len(digest) - 1
        return f"{job_id[:prefix_length]}-{digest}{suffix}"
