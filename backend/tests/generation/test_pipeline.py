"""Verify generated-lesson orchestration, validation, and repair behavior.
The tests preserve evidence and provenance across pipeline outcomes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from math_tutor.domain import LessonStage
from math_tutor.generation.errors import (
    GeneratedLessonError,
    OutputValidationError,
    SceneValidationError,
)
from math_tutor.generation.pipeline import GeneratedLessonPipeline
from math_tutor.generation.provider import (
    SHARED_ADAPTER_MODEL,
    GenerationConfig,
    GenerationResult,
    ModelHealth,
    ProviderError,
    TokenUsage,
)
from math_tutor.jobs import JobExecutionError, JobStore, RenderOutcome
from math_tutor.rendering.manim import RenderFailed, RenderTimedOut
from math_tutor.rendering.narration import NarrationStatus
from math_tutor.validation.models import (
    RenderedAttempt,
    ValidationFinding,
    ValidationReport,
    ValidationStatus,
)
from math_tutor.validation.suite import ValidatorSuite

VALID_SCENE = """from manim import *

class GeneratedLesson(Scene):
    def construct(self):
        self.play(Write(MathTex(r"x^2")))
"""

VOICEOVER_SCENE = """from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.elevenlabs import ElevenLabsService

class GeneratedLesson(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            ElevenLabsService(
                voice_id="voice-id",
                model="eleven_multilingual_v2",
                transcription_model=None,
            )
        )
        circle = Circle()
        with self.voiceover(text="Draw the circle.") as tracker:
            self.play(Create(circle), run_time=tracker.duration)
        with self.voiceover(text="Move it right.") as tracker:
            self.play(circle.animate.shift(RIGHT), run_time=tracker.duration)
        with self.voiceover(text="Now remove it.") as tracker:
            self.play(FadeOut(circle), run_time=tracker.duration)
"""


class FixedGenerator:
    def __init__(self, result: GenerationResult) -> None:
        self.config = GenerationConfig()
        self._result = result

    def generate(self, prompt: str) -> GenerationResult:
        return self._result

    def health(self) -> ModelHealth:
        return ModelHealth(True, self.config.model, True)


class RecordingSourceRenderer:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self.received: tuple[str, str, str] | None = None

    def render_source(self, job_id: str, source: str, scene_class: str) -> RenderOutcome:
        self.received = (job_id, source, scene_class)
        self.video_path.write_bytes(b"video")
        return RenderOutcome(
            video_path=self.video_path,
            renderer="docker:test-image@sha256:abc",
            elapsed_seconds=1.25,
            logs="rendered",
        )


class SequenceSourceRenderer:
    def __init__(self, root: Path) -> None:
        self._root = root
        self.calls: list[str] = []

    def render_source(self, job_id: str, source: str, scene_class: str) -> RenderOutcome:
        self.calls.append(job_id)
        video_path = self._root / f"{job_id}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"video-{job_id}".encode())
        return RenderOutcome(video_path, "test", 1, "rendered")


class FailingSourceRenderer:
    def render_source(self, job_id: str, source: str, scene_class: str) -> RenderOutcome:
        raise RenderFailed("render failed")


class TimeoutThenSourceRenderer(SequenceSourceRenderer):
    def render_source(self, job_id: str, source: str, scene_class: str) -> RenderOutcome:
        self.calls.append(job_id)
        if len(self.calls) == 1:
            raise RenderTimedOut("render timed out")
        video_path = self._root / f"{job_id}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"video-{job_id}".encode())
        return RenderOutcome(video_path, "test", 1, "rendered")


class SequenceValidator:
    name = "sequence"
    expected_checks = ("lesson_duration",)

    def __init__(self, reports: list[ValidationReport]) -> None:
        self._reports = iter(reports)

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        return next(self._reports)


class ManifestRecordingValidator:
    name = "manifest_recorder"
    expected_checks = ("manifest_available",)

    def __init__(self) -> None:
        self.manifest_paths: list[Path] = []

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        manifest_path = attempt.validation_input_path
        assert manifest_path is not None
        assert manifest_path.is_file()
        self.manifest_paths.append(manifest_path)
        return ValidationReport(self.name, ValidationStatus.PASS)


class FixedAttemptValidator:
    def __init__(self, name: str, check: str, *, raises: bool = False) -> None:
        self.name = name
        self.expected_checks = (check,)
        self._raises = raises

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        if self._raises:
            raise RuntimeError("private validator failure")
        return ValidationReport(self.name, ValidationStatus.PASS)


class NarrationRecordingProcessor:
    """Replace the silent outcome before validation and record processor input."""

    def __init__(self, narrated_video: Path, captions: Path) -> None:
        self._narrated_video = narrated_video
        self._captions = captions
        self.calls: list[tuple[str, str, str, Path]] = []

    def process(
        self,
        *,
        job_id: str,
        prompt: str,
        source: str,
        outcome: RenderOutcome,
        artifact_dir: Path,
    ) -> RenderOutcome:
        self.calls.append((job_id, prompt, source, artifact_dir))
        self._narrated_video.write_bytes(b"narrated")
        self._captions.write_text("WEBVTT\n", encoding="utf-8")
        return replace(
            outcome,
            video_path=self._narrated_video,
            silent_video_path=outcome.video_path,
            captions_path=self._captions,
            narration_status=NarrationStatus.READY,
        )


def _passing_validation_diagnostics(
    *,
    attempt_count: int,
    repair_count: int,
    infrastructure_retry_count: int,
    selected_attempt: int,
    advisories: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return the complete public diagnostics for a passing media report."""
    diagnostics: dict[str, object] = {
        "attempt_count": attempt_count,
        "repair_count": repair_count,
        "infrastructure_retry_count": infrastructure_retry_count,
        "selected_attempt": selected_attempt,
        "validation_status": "pass",
        "validation_axes": [
            {
                "validator": "media",
                "status": "pass",
                "finding_count": 0,
                "advisory_count": len(advisories or []),
                "provenance": {},
            }
        ],
        "generation_provenance": {
            "model": SHARED_ADAPTER_MODEL,
            "provider": "configured_generator",
            "inference_path": "lora_adapter",
        },
        "narration_status": "not_requested",
    }
    if advisories:
        diagnostics["validation_advisories"] = advisories
    return diagnostics


class FailingGenerator:
    def __init__(self) -> None:
        self.config = GenerationConfig()

    def generate(self, prompt: str) -> GenerationResult:
        raise ProviderError("provider unavailable")


class RecordingGenerator:
    def __init__(self, result: GenerationResult | Exception) -> None:
        self.config = GenerationConfig(model="intermediate")
        self._result = result
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> GenerationResult:
        self.prompts.append(prompt)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class SequenceGenerator:
    def __init__(self, results: list[GenerationResult]) -> None:
        self.config = GenerationConfig()
        self._results = iter(results)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> GenerationResult:
        self.prompts.append(prompt)
        return next(self._results)


def _generation(content: str, *, model: str = SHARED_ADAPTER_MODEL) -> GenerationResult:
    return GenerationResult(
        content=content,
        model=model,
        request_id="chatcmpl-123",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        elapsed_seconds=0.5,
        provider_response=json.dumps({"id": "chatcmpl-123", "choices": []}),
    )


def test_pipeline_rejects_more_than_one_repair_attempt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be zero or one"):
        GeneratedLessonPipeline(
            artifact_root=tmp_path / "artifacts",
            prompt="Explain limits.",
            generator=FixedGenerator(_generation(f"```python\n{VALID_SCENE}```")),
            renderer=RecordingSourceRenderer(tmp_path / "unused.mp4"),
            max_repair_attempts=2,
        )


def test_pipeline_persists_generation_evidence_before_isolated_render(tmp_path: Path) -> None:
    response = _generation(f"Here is the scene:\n```python\n{VALID_SCENE}```")
    source_renderer = RecordingSourceRenderer(tmp_path / "lesson.mp4")
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain the derivative of x squared visually.",
        generator=FixedGenerator(response),
        renderer=source_renderer,
    )

    outcome = pipeline.render("generated-123")

    assert outcome.video_path.read_bytes() == b"video"
    assert outcome.generated_code == VALID_SCENE.rstrip()
    assert outcome.narration_diagnostics == {
        "inference_path": "lora_adapter",
        "inference_model": SHARED_ADAPTER_MODEL,
    }
    assert source_renderer.received == (
        "generated-123-attempt-0",
        VALID_SCENE.rstrip(),
        "GeneratedLesson",
    )
    job_dir = tmp_path / "artifacts" / "generated-123"
    attempt_dir = job_dir / "attempts" / "0"
    assert (job_dir / "prompt.txt").read_text() == ("Explain the derivative of x squared visually.")
    assert (attempt_dir / "raw_response.txt").read_text() == response.content
    assert (attempt_dir / "provider_response.json").read_text() == response.provider_response
    assert (attempt_dir / "extracted_scene.py").read_text() == VALID_SCENE.rstrip()
    metadata = json.loads((attempt_dir / "generation.json").read_text())
    assert metadata == {
        "completion_tokens": 20,
        "elapsed_seconds": 0.5,
        "finish_reason": "stop",
        "model": SHARED_ADAPTER_MODEL,
        "prompt_tokens": 10,
        "request_id": "chatcmpl-123",
        "seed": 42,
        "status": "generated",
        "temperature": 0.0,
        "top_p": 1.0,
        "total_tokens": 30,
    }


def test_pipeline_reports_each_generation_stage(tmp_path: Path) -> None:
    observed: list[tuple[str, LessonStage]] = []
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain the derivative visually.",
        generator=FixedGenerator(_generation(f"```python\n{VALID_SCENE}```")),
        renderer=RecordingSourceRenderer(tmp_path / "lesson.mp4"),
        stage_reporter=lambda job_id, stage: observed.append((job_id, stage)),
    )

    pipeline.render("staged-123")

    assert observed == [
        ("staged-123", LessonStage.GENERATING_CODE),
        ("staged-123", LessonStage.VALIDATING_CODE),
        ("staged-123", LessonStage.RENDERING),
    ]


def test_pipeline_reports_render_before_output_validation(tmp_path: Path) -> None:
    events: list[str] = []

    class OrderedValidator(ManifestRecordingValidator):
        def validate(self, attempt: RenderedAttempt) -> ValidationReport:
            events.append("validation")
            return super().validate(attempt)

    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain the derivative visually.",
        generator=FixedGenerator(_generation(f"```python\n{VALID_SCENE}```")),
        renderer=RecordingSourceRenderer(tmp_path / "lesson.mp4"),
        validator=OrderedValidator(),
        render_reporter=lambda _job_id, _attempt, _outcome: events.append("render"),
    )

    pipeline.render("render-reported-123")

    assert events == ["render", "validation"]


def test_pipeline_reports_initial_render_then_narrates_before_validation(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    processor = NarrationRecordingProcessor(
        tmp_path / "narrated.mp4",
        tmp_path / "captions.vtt",
    )

    class NarrationValidator(ManifestRecordingValidator):
        def validate(self, attempt: RenderedAttempt) -> ValidationReport:
            events.append("validation")
            assert attempt.narration_required is True
            assert attempt.captions_required is True
            assert attempt.outcome.video_path.read_bytes() == b"narrated"
            assert attempt.outcome.narration_status is NarrationStatus.READY
            return super().validate(attempt)

    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain the derivative visually.",
        generator=FixedGenerator(_generation(f"```python\n{VALID_SCENE}```")),
        renderer=RecordingSourceRenderer(tmp_path / "silent.mp4"),
        validator=NarrationValidator(),
        outcome_processor=processor,
        narration_required=True,
        captions_required=True,
        render_reporter=lambda _job_id, _attempt, outcome: events.append(
            outcome.narration_status.value
        ),
    )

    outcome = pipeline.render("narrated-123")

    attempt_dir = tmp_path / "artifacts" / "narrated-123" / "attempts" / "0"
    assert processor.calls == [
        (
            "narrated-123",
            "Explain the derivative visually.",
            VALID_SCENE.rstrip(),
            attempt_dir,
        )
    ]
    assert events == ["not_requested", "ready", "validation"]
    assert outcome.video_path.read_bytes() == b"narrated"
    assert outcome.silent_video_path == tmp_path / "silent.mp4"


def test_narration_planning_keeps_original_prompt_during_repair(tmp_path: Path) -> None:
    processor = NarrationRecordingProcessor(
        tmp_path / "narrated.mp4",
        tmp_path / "captions.vtt",
    )
    failed_report = ValidationReport(
        validator="media",
        status=ValidationStatus.FAIL,
        findings=(
            ValidationFinding(
                code="duration_out_of_range",
                message="The lesson is too short.",
                repair_instruction="Extend the visual explanation.",
            ),
        ),
    )
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain the derivative visually.",
        generator=SequenceGenerator(
            [
                _generation(f"```python\n{VALID_SCENE}```"),
                _generation(f"```python\n{VALID_SCENE}```"),
            ]
        ),
        renderer=SequenceSourceRenderer(tmp_path / "rendered"),
        validator=SequenceValidator(
            [failed_report, ValidationReport("media", ValidationStatus.PASS)]
        ),
        outcome_processor=processor,
        narration_required=True,
        captions_required=True,
        max_repair_attempts=1,
    )

    pipeline.render("narration-repair-123")

    assert [call[1] for call in processor.calls] == [
        "Explain the derivative visually.",
        "Explain the derivative visually.",
    ]


def test_narration_failure_keeps_initial_video_and_skips_validation(tmp_path: Path) -> None:
    store = JobStore()
    job = store.create("Explain limits.", narration_requested=True)
    store.mark_running(job.id)
    validator = ManifestRecordingValidator()

    class FailingProcessor:
        def process(self, **_kwargs: object) -> RenderOutcome:
            raise JobExecutionError(
                "Narration could not be attached to the lesson",
                diagnostics={
                    "failure_stage": "narration",
                    "failure_kind": "operational",
                    "narration_status": "unavailable",
                },
            )

    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain limits.",
        generator=FixedGenerator(_generation(f"```python\n{VALID_SCENE}```")),
        renderer=RecordingSourceRenderer(tmp_path / "silent.mp4"),
        validator=validator,
        outcome_processor=FailingProcessor(),
        narration_required=True,
        captions_required=True,
        render_reporter=lambda job_id, _attempt, outcome: store.mark_initial_render(
            job_id, outcome
        ),
    )

    with pytest.raises(JobExecutionError) as caught:
        pipeline.render(job.id)
    failed = store.mark_failed(job.id, caught.value)

    assert failed.initial_video_path == str(tmp_path / "silent.mp4")
    assert failed.narration_status is NarrationStatus.UNAVAILABLE
    assert failed.diagnostics["failure_stage"] == "narration"
    assert validator.manifest_paths == []


def test_pipeline_persists_one_validation_input_before_validation(tmp_path: Path) -> None:
    validator = ManifestRecordingValidator()
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain the derivative visually.",
        generator=FixedGenerator(_generation(f"```python\n{VALID_SCENE}```")),
        renderer=RecordingSourceRenderer(tmp_path / "lesson.mp4"),
        validator=validator,
    )

    outcome = pipeline.render("manifest-123")

    assert len(validator.manifest_paths) == 1
    validation_input = json.loads(validator.manifest_paths[0].read_text())
    assert validation_input["schema_version"] == "validation-input.v1"
    assert validation_input["expected_checks"] == ["manifest_available"]
    assert validation_input["source"]["sha256"]
    assert validation_input["render"]["renderer"] == "docker:test-image@sha256:abc"
    assert validation_input["media"]["video"]["sha256"]
    expected_axes = [
        {
            "validator": "manifest_recorder",
            "status": "pass",
            "finding_count": 0,
            "advisory_count": 0,
            "provenance": {},
        }
    ]
    assert outcome.validation_diagnostics is not None
    assert outcome.validation_diagnostics["validation_axes"] == expected_axes
    assert outcome.validation_diagnostics["generation_provenance"] == {
        "model": SHARED_ADAPTER_MODEL,
        "provider": "configured_generator",
        "inference_path": "lora_adapter",
    }
    assert outcome.validation_diagnostics["narration_status"] == "not_requested"
    attempt_dir = tmp_path / "artifacts" / "manifest-123" / "attempts" / "0"
    attempt_manifest = json.loads((attempt_dir / "attempt.json").read_text())
    selected = json.loads(
        (tmp_path / "artifacts" / "manifest-123" / "selected_attempt.json").read_text()
    )
    assert attempt_manifest["validation_axes"] == expected_axes
    assert attempt_manifest["validation_input"]["sha256"]
    assert selected["validation_axes"] == expected_axes
    assert (
        selected["generation_provenance"] == outcome.validation_diagnostics["generation_provenance"]
    )
    assert selected["narration_status"] == outcome.validation_diagnostics["narration_status"]


def test_pipeline_retains_every_axis_when_one_concurrent_validator_raises(
    tmp_path: Path,
) -> None:
    validator = ValidatorSuite(
        (
            FixedAttemptValidator("media", "media_check"),
            FixedAttemptValidator("spatial", "spatial_check", raises=True),
            FixedAttemptValidator("visual_evidence", "visual_check"),
        )
    )
    generator = RecordingGenerator(_generation(f"```python\n{VALID_SCENE}```"))
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain a derivative visually.",
        generator=generator,
        renderer=RecordingSourceRenderer(tmp_path / "lesson.mp4"),
        validator=validator,
    )

    with pytest.raises(OutputValidationError) as caught:
        pipeline.render("validator-exception-123")

    axes = caught.value.diagnostics["validation_axes"]
    assert isinstance(axes, list)
    assert [axis["status"] for axis in axes] == ["pass", "validator_error", "pass"]
    job_dir = tmp_path / "artifacts" / "validator-exception-123"
    validation = json.loads((job_dir / "attempts" / "0" / "validation.json").read_text())
    manifest = json.loads((job_dir / "attempts" / "0" / "attempt.json").read_text())
    assert [report["status"] for report in validation["component_reports"]] == [
        "pass",
        "validator_error",
        "pass",
    ]
    assert manifest["validation_axes"] == axes
    assert not (job_dir / "selected_attempt.json").exists()
    assert len(generator.prompts) == 1


def test_pipeline_repairs_one_media_failure_and_preserves_both_attempts(
    tmp_path: Path,
) -> None:
    generator = SequenceGenerator(
        [
            _generation(f"```python\n{VALID_SCENE}```"),
            _generation(f"```python\n{VALID_SCENE}```"),
        ]
    )
    renderer = SequenceSourceRenderer(tmp_path / "rendered")
    failed_report = ValidationReport(
        validator="media",
        status=ValidationStatus.FAIL,
        findings=(
            ValidationFinding(
                code="duration_out_of_range",
                message="RAW VALIDATOR PROSE MUST NOT ENTER REPAIR PROMPT",
                evidence={"actual_seconds": 7, "minimum_seconds": 30},
                repair_instruction="Extend the lesson to at least 30 seconds.",
            ),
        ),
    )
    validator = SequenceValidator([failed_report, ValidationReport("media", ValidationStatus.PASS)])
    observed: list[LessonStage] = []
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain distance between two points.",
        generator=generator,
        renderer=renderer,
        validator=validator,
        max_repair_attempts=1,
        stage_reporter=lambda _job_id, stage: observed.append(stage),
    )

    outcome = pipeline.render("repair-123")

    assert renderer.calls == ["repair-123-attempt-0", "repair-123-attempt-1"]
    assert generator.prompts[0] == "Explain distance between two points."
    assert "duration_out_of_range" in generator.prompts[1]
    assert "Extend the lesson to at least 30 seconds." in generator.prompts[1]
    assert "RAW VALIDATOR PROSE" not in generator.prompts[1]
    assert "Preserve the original mathematical topic" in generator.prompts[1]
    assert f"```python\n{VALID_SCENE}```" in generator.prompts[1]
    assert outcome.video_path.read_bytes() == b"video-repair-123-attempt-1"
    assert outcome.validation_diagnostics == _passing_validation_diagnostics(
        attempt_count=2,
        repair_count=1,
        infrastructure_retry_count=0,
        selected_attempt=1,
    )
    job_dir = tmp_path / "artifacts" / "repair-123"
    assert (
        json.loads((job_dir / "attempts" / "0" / "validation.json").read_text())["status"] == "fail"
    )
    assert (
        json.loads((job_dir / "attempts" / "1" / "validation.json").read_text())["status"] == "pass"
    )
    selected = json.loads((job_dir / "selected_attempt.json").read_text())
    assert selected["attempt_number"] == 1
    assert selected["video_sha256"]
    manifest = json.loads((job_dir / "attempts" / "1" / "attempt.json").read_text())
    assert manifest["schema_version"] == "rendered-attempt.v1"
    assert manifest["expected_checks"] == ["lesson_duration"]
    assert manifest["original_prompt"]["sha256"]
    assert manifest["generation_prompt"]["sha256"]
    assert manifest["raw_response"]["sha256"]
    assert manifest["provider_response"]["sha256"]
    assert manifest["source"]["sha256"]
    assert manifest["media"]["video"]["sha256"] == selected["video_sha256"]
    assert manifest["provenance"] == {
        "infrastructure_retry_count": 0,
        "inference_path": "lora_adapter",
        "model": SHARED_ADAPTER_MODEL,
        "provider": "configured_generator",
        "renderer": "test",
    }
    assert observed == [
        LessonStage.GENERATING_CODE,
        LessonStage.VALIDATING_CODE,
        LessonStage.RENDERING,
        LessonStage.VALIDATING_OUTPUT,
        LessonStage.REPAIRING,
        LessonStage.GENERATING_CODE,
        LessonStage.VALIDATING_CODE,
        LessonStage.RENDERING,
        LessonStage.VALIDATING_OUTPUT,
    ]


def test_pipeline_repairs_static_source_failure_before_rendering(tmp_path: Path) -> None:
    generator = SequenceGenerator(
        [
            _generation("No Python scene was produced."),
            _generation(f"```python\n{VALID_SCENE}```"),
        ]
    )
    renderer = SequenceSourceRenderer(tmp_path / "rendered")
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain a derivative.",
        generator=generator,
        renderer=renderer,
        validator=SequenceValidator([ValidationReport("media", ValidationStatus.PASS)]),
        max_repair_attempts=1,
    )

    pipeline.render("source-repair-123")

    assert len(generator.prompts) == 2
    assert "source_admission_failed" in generator.prompts[1]
    assert renderer.calls == ["source-repair-123-attempt-1"]
    first_validation = json.loads(
        (
            tmp_path / "artifacts" / "source-repair-123" / "attempts" / "0" / "validation.json"
        ).read_text()
    )
    assert first_validation["status"] == "fail"


def test_pipeline_accepts_first_passing_attempt_without_repair(tmp_path: Path) -> None:
    generator = RecordingGenerator(_generation(f"```python\n{VALID_SCENE}```"))
    renderer = SequenceSourceRenderer(tmp_path / "rendered")
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain a derivative.",
        generator=generator,
        renderer=renderer,
        validator=SequenceValidator([ValidationReport("media", ValidationStatus.PASS)]),
    )

    outcome = pipeline.render("initial-pass-123")

    assert len(generator.prompts) == 1
    assert renderer.calls == ["initial-pass-123-attempt-0"]
    assert outcome.validation_diagnostics == _passing_validation_diagnostics(
        attempt_count=1,
        repair_count=0,
        infrastructure_retry_count=0,
        selected_attempt=0,
    )


def test_pipeline_publishes_duration_advisory_without_repair(tmp_path: Path) -> None:
    generator = RecordingGenerator(_generation(f"```python\n{VALID_SCENE}```"))
    advisory = ValidationFinding(
        code="duration_outside_target",
        message="Lesson duration is outside the target range.",
        evidence={
            "actual_seconds": 11,
            "target_minimum_seconds": 30,
            "target_maximum_seconds": 45,
        },
    )
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain a derivative.",
        generator=generator,
        renderer=SequenceSourceRenderer(tmp_path / "rendered"),
        validator=SequenceValidator(
            [
                ValidationReport(
                    "media",
                    ValidationStatus.PASS,
                    advisories=(advisory,),
                )
            ]
        ),
    )

    outcome = pipeline.render("duration-advisory-123")

    assert len(generator.prompts) == 1
    assert outcome.validation_diagnostics == _passing_validation_diagnostics(
        attempt_count=1,
        repair_count=0,
        infrastructure_retry_count=0,
        selected_attempt=0,
        advisories=[advisory.to_dict()],
    )


def test_pipeline_preserves_both_attempts_when_repair_also_fails(tmp_path: Path) -> None:
    generator = SequenceGenerator(
        [
            _generation(f"```python\n{VALID_SCENE}```"),
            _generation(f"```python\n{VALID_SCENE}```"),
        ]
    )
    renderer = SequenceSourceRenderer(tmp_path / "rendered")
    failure = ValidationReport(
        validator="media",
        status=ValidationStatus.FAIL,
        findings=(
            ValidationFinding(
                code="duration_out_of_range",
                message="Lesson duration is outside the configured range.",
                repair_instruction="Extend the lesson to at least 30 seconds.",
            ),
        ),
    )
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain a derivative.",
        generator=generator,
        renderer=renderer,
        validator=SequenceValidator([failure, failure]),
    )

    with pytest.raises(GeneratedLessonError) as caught:
        pipeline.render("repair-failed-123")

    assert caught.value.diagnostics["attempt_count"] == 2
    assert caught.value.diagnostics["repair_count"] == 1
    job_dir = tmp_path / "artifacts" / "repair-failed-123"
    assert (job_dir / "attempts" / "0" / "attempt.json").is_file()
    assert (job_dir / "attempts" / "1" / "attempt.json").is_file()
    assert not (job_dir / "selected_attempt.json").exists()
    assert len(generator.prompts) == 2


def test_pipeline_respects_zero_repair_budget(tmp_path: Path) -> None:
    generator = RecordingGenerator(_generation(f"```python\n{VALID_SCENE}```"))
    failure = ValidationReport(
        validator="media",
        status=ValidationStatus.FAIL,
        findings=(
            ValidationFinding(
                code="duration_out_of_range",
                message="Lesson duration is outside the configured range.",
                repair_instruction="Extend the lesson to at least 30 seconds.",
            ),
        ),
    )
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain a derivative.",
        generator=generator,
        renderer=SequenceSourceRenderer(tmp_path / "rendered"),
        validator=SequenceValidator([failure]),
        max_repair_attempts=0,
    )

    with pytest.raises(GeneratedLessonError) as caught:
        pipeline.render("no-repair-123")

    assert caught.value.diagnostics["attempt_count"] == 1
    assert caught.value.diagnostics["repair_count"] == 0
    assert len(generator.prompts) == 1
    assert not (tmp_path / "artifacts" / "no-repair-123" / "attempts" / "1").exists()


def test_pipeline_does_not_repair_an_uncertain_aggregate(tmp_path: Path) -> None:
    generator = RecordingGenerator(_generation(f"```python\n{VALID_SCENE}```"))
    report = ValidationReport(
        validator="validation_suite",
        status=ValidationStatus.UNCERTAIN,
        findings=(
            ValidationFinding(
                code="object_off_frame",
                message="Object is outside the frame.",
                repair_instruction="Move the object inside the frame.",
            ),
            ValidationFinding(
                code="visual_evidence_uncertain",
                message="Sampled frames are inconclusive.",
            ),
        ),
    )
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain a derivative.",
        generator=generator,
        renderer=SequenceSourceRenderer(tmp_path / "rendered"),
        validator=SequenceValidator([report]),
        max_repair_attempts=1,
    )

    with pytest.raises(GeneratedLessonError) as caught:
        pipeline.render("uncertain-123")

    assert caught.value.diagnostics["validation_status"] == "uncertain"
    assert caught.value.diagnostics["repair_count"] == 0
    assert len(generator.prompts) == 1
    assert not (tmp_path / "artifacts" / "uncertain-123" / "attempts" / "1").exists()


def test_pipeline_does_not_use_model_repair_for_render_failures(tmp_path: Path) -> None:
    generator = RecordingGenerator(_generation(f"```python\n{VALID_SCENE}```"))
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain a derivative.",
        generator=generator,
        renderer=FailingSourceRenderer(),
        validator=SequenceValidator([ValidationReport("media", ValidationStatus.PASS)]),
        max_repair_attempts=1,
    )
    with pytest.raises(RenderFailed) as caught:
        pipeline.render("render-failed-123")

    assert len(generator.prompts) == 1
    assert caught.value.diagnostics["failure_kind"] == "operational"
    assert caught.value.diagnostics["repair_count"] == 0
    assert Path(str(caught.value.diagnostics["attempt_manifest"])).is_file()


def test_pipeline_retries_transient_render_without_consuming_model_repair(
    tmp_path: Path,
) -> None:
    generator = RecordingGenerator(_generation(f"```python\n{VALID_SCENE}```"))
    renderer = TimeoutThenSourceRenderer(tmp_path / "rendered")
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain a derivative.",
        generator=generator,
        renderer=renderer,
        validator=SequenceValidator([ValidationReport("media", ValidationStatus.PASS)]),
    )

    outcome = pipeline.render("render-retry-123")

    assert len(generator.prompts) == 1
    assert renderer.calls == [
        "render-retry-123-attempt-0",
        "render-retry-123-attempt-0-retry-1",
    ]
    assert outcome.validation_diagnostics == _passing_validation_diagnostics(
        attempt_count=1,
        repair_count=0,
        infrastructure_retry_count=1,
        selected_attempt=0,
    )
    manifest = json.loads(
        (
            tmp_path / "artifacts" / "render-retry-123" / "attempts" / "0" / "attempt.json"
        ).read_text()
    )
    assert manifest["provenance"]["infrastructure_retry_count"] == 1


def test_pipeline_reports_render_retries_from_attempts_before_selected_attempt(
    tmp_path: Path,
) -> None:
    generator = SequenceGenerator(
        [
            _generation(f"```python\n{VALID_SCENE}```"),
            _generation(f"```python\n{VALID_SCENE}```"),
        ]
    )
    renderer = TimeoutThenSourceRenderer(tmp_path / "rendered")
    failure = ValidationReport(
        validator="media",
        status=ValidationStatus.FAIL,
        findings=(
            ValidationFinding(
                code="duration_out_of_range",
                message="Duration failed.",
                repair_instruction="Extend the lesson.",
            ),
        ),
    )
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain a derivative.",
        generator=generator,
        renderer=renderer,
        validator=SequenceValidator([failure, ValidationReport("media", ValidationStatus.PASS)]),
    )

    outcome = pipeline.render("retry-before-repair-123")

    assert outcome.validation_diagnostics == _passing_validation_diagnostics(
        attempt_count=2,
        repair_count=1,
        infrastructure_retry_count=1,
        selected_attempt=1,
    )
    job_dir = tmp_path / "artifacts" / "retry-before-repair-123" / "attempts"
    first_manifest = json.loads((job_dir / "0" / "attempt.json").read_text())
    second_manifest = json.loads((job_dir / "1" / "attempt.json").read_text())
    assert first_manifest["provenance"]["infrastructure_retry_count"] == 1
    assert second_manifest["provenance"]["infrastructure_retry_count"] == 0


def test_pipeline_reports_validator_execution_error_without_model_repair(
    tmp_path: Path,
) -> None:
    generator = RecordingGenerator(_generation(f"```python\n{VALID_SCENE}```"))
    report = ValidationReport(
        validator="media",
        status=ValidationStatus.VALIDATOR_ERROR,
        findings=(
            ValidationFinding(
                code="media_inspection_failed",
                message="Rendered media could not be inspected.",
            ),
        ),
    )
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain a derivative.",
        generator=generator,
        renderer=SequenceSourceRenderer(tmp_path / "rendered"),
        validator=SequenceValidator([report]),
    )

    with pytest.raises(GeneratedLessonError) as caught:
        pipeline.render("validator-error-123")

    assert caught.value.diagnostics["validation_status"] == "validator_error"
    assert caught.value.diagnostics["attempt_count"] == 1
    assert caught.value.diagnostics["repair_count"] == 0
    assert caught.value.diagnostics["validation_axes"] == [
        {
            "validator": "media",
            "status": "validator_error",
            "finding_count": 1,
            "advisory_count": 0,
            "provenance": {},
        }
    ]
    assert caught.value.diagnostics["generation_provenance"] == {
        "model": SHARED_ADAPTER_MODEL,
        "provider": "configured_generator",
        "inference_path": "lora_adapter",
    }
    assert caught.value.diagnostics["narration_status"] == "not_requested"
    assert len(generator.prompts) == 1


def test_voiceover_pipeline_marks_rendered_audio_ready(tmp_path: Path) -> None:
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain circles.",
        generator=FixedGenerator(_generation(f"```python\n{VOICEOVER_SCENE}```")),
        renderer=RecordingSourceRenderer(tmp_path / "lesson.mp4"),
        voiceover=True,
    )

    outcome = pipeline.render("voiceover-123")

    assert outcome.narration_status is NarrationStatus.READY


def test_pipeline_records_silent_fallback_status_in_artifacts(tmp_path: Path) -> None:
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain circles.",
        generator=FixedGenerator(_generation(f"```python\n{VALID_SCENE}```")),
        renderer=RecordingSourceRenderer(tmp_path / "lesson.mp4"),
        narration_status_override=NarrationStatus.UNAVAILABLE,
    )

    outcome = pipeline.render("silent-fallback-123")

    job_dir = tmp_path / "artifacts" / "silent-fallback-123"
    validation_input = json.loads(
        (job_dir / "attempts" / "0" / "validation_input.json").read_text()
    )
    selected = json.loads((job_dir / "selected_attempt.json").read_text())
    assert outcome.narration_status is NarrationStatus.UNAVAILABLE
    assert validation_input["media"]["narration_status"] == "unavailable"
    assert selected["narration_status"] == "unavailable"
    assert outcome.validation_diagnostics is not None
    assert outcome.validation_diagnostics["narration_status"] == "unavailable"


def test_pipeline_reports_explicit_lora_route(tmp_path: Path) -> None:
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain Fourier transforms.",
        generator=FixedGenerator(_generation(f"```python\n{VALID_SCENE}```", model="advanced")),
        renderer=RecordingSourceRenderer(tmp_path / "lesson.mp4"),
        inference_path="lora_adapter",
    )

    outcome = pipeline.render("lora-123")

    assert outcome.narration_diagnostics == {
        "inference_path": "lora_adapter",
        "inference_model": "advanced",
    }


def test_pipeline_preserves_raw_response_when_extraction_fails(tmp_path: Path) -> None:
    response = _generation("I cannot provide code.")
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Prompt",
        generator=FixedGenerator(response),
        renderer=RecordingSourceRenderer(tmp_path / "unused.mp4"),
    )

    with pytest.raises(SceneValidationError) as caught:
        pipeline.render("failed-123")

    job_dir = tmp_path / "artifacts" / "failed-123"
    assert (job_dir / "attempts" / "0" / "raw_response.txt").read_text() == (
        "I cannot provide code."
    )
    assert (job_dir / "attempts" / "1" / "raw_response.txt").read_text() == (
        "I cannot provide code."
    )
    metadata = json.loads((job_dir / "attempts" / "1" / "generation.json").read_text())
    assert metadata["status"] == "parse_failed"
    assert metadata["model"] == SHARED_ADAPTER_MODEL
    assert caught.value.diagnostics["attempt_count"] == 2
    assert caught.value.diagnostics["repair_count"] == 1
    assert caught.value.diagnostics["validation_reports"][0]["validator"] == "source"
    assert caught.value.diagnostics["findings"][0]["code"] == "source_admission_failed"
    assert "admission error" in str(caught.value.diagnostics["findings"][0]["repair_instruction"])
    assert Path(str(caught.value.diagnostics["attempt_manifest"])).is_file()


def test_pipeline_preserves_timing_and_stage_when_provider_fails(tmp_path: Path) -> None:
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Prompt",
        generator=FailingGenerator(),
        renderer=RecordingSourceRenderer(tmp_path / "unused.mp4"),
    )

    with pytest.raises(GeneratedLessonError) as caught:
        pipeline.render("provider-failed-123")

    assert caught.value.diagnostics["failure_stage"] == "provider"
    assert caught.value.diagnostics["failure_kind"] == "operational"
    assert caught.value.diagnostics["attempt_count"] == 1
    assert caught.value.diagnostics["repair_count"] == 0
    job_dir = tmp_path / "artifacts" / "provider-failed-123"
    assert (job_dir / "prompt.txt").read_text() == "Prompt"
    metadata = json.loads((job_dir / "attempts" / "0" / "generation.json").read_text())
    assert metadata["status"] == "provider_failed"
    assert metadata["failure_stage"] == "provider"
    assert metadata["elapsed_seconds"] >= 0
    assert (job_dir / "attempts" / "0" / "attempt.json").is_file()


def test_pipeline_rejects_unsafe_job_id_before_writing_or_generation(tmp_path: Path) -> None:
    generator = FixedGenerator(_generation(f"```python\n{VALID_SCENE}```"))
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Prompt",
        generator=generator,
        renderer=RecordingSourceRenderer(tmp_path / "unused.mp4"),
    )

    with pytest.raises(GeneratedLessonError, match="job id is not safe") as caught:
        pipeline.render("../escaped")

    assert caught.value.diagnostics == {"failure_stage": "validation"}
    assert not (tmp_path / "escaped").exists()
