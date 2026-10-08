"""Verify generated-source admission, rendering, validation, and repair.
The tests preserve evidence and routing behavior across pipeline outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from math_tutor.attempts import RenderedAttempt
from math_tutor.domain import LessonStage
from math_tutor.generated_lesson import (
    ExtractionError,
    GeneratedLessonError,
    GeneratedLessonPipeline,
    GenerationFallbackRenderer,
    SceneValidationError,
    SpecialistGuidedLessonPipeline,
    VoiceoverFallbackRenderer,
    extract_and_validate_raw_scene,
    extract_and_validate_scene,
)
from math_tutor.generation import (
    GenerationConfig,
    GenerationResult,
    ModelHealth,
    ProviderError,
    TokenUsage,
)
from math_tutor.jobs import RenderOutcome
from math_tutor.narration import NarrationStatus
from math_tutor.renderer import RenderFailed, RenderTimedOut
from math_tutor.validation.models import (
    ValidationFinding,
    ValidationReport,
    ValidationStatus,
)

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
    expected_checks = ("lesson_duration",)

    def __init__(self, reports: list[ValidationReport]) -> None:
        self._reports = iter(reports)

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        return next(self._reports)


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


class RecordingPromptRenderer:
    def __init__(self, outcome: RenderOutcome | Exception) -> None:
        self._outcome = outcome
        self.calls: list[tuple[str, str | None]] = []

    def render(self, job_id: str, prompt: str | None = None) -> RenderOutcome:
        self.calls.append((job_id, prompt))
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class SequencePromptRenderer:
    def __init__(self, outcomes: list[RenderOutcome | Exception]) -> None:
        self._outcomes = iter(outcomes)
        self.calls: list[tuple[str, str | None]] = []

    def render(self, job_id: str, prompt: str | None = None) -> RenderOutcome:
        self.calls.append((job_id, prompt))
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _generation(content: str, *, model: str = "Qwen/Qwen3-4B") -> GenerationResult:
    return GenerationResult(
        content=content,
        model=model,
        request_id="chatcmpl-123",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        elapsed_seconds=0.5,
        provider_response=json.dumps({"id": "chatcmpl-123", "choices": []}),
    )


def test_extracts_one_python_fence_and_validates_generated_scene() -> None:
    extracted = extract_and_validate_scene(f"```python\n{VALID_SCENE}```")

    assert extracted.source == VALID_SCENE.rstrip()
    assert extracted.scene_class == "GeneratedLesson"


def test_raw_manim_scene_accepts_training_style_class_without_rewriting_it() -> None:
    source = VALID_SCENE.replace("GeneratedLesson", "TriangleProof")

    extracted = extract_and_validate_raw_scene(source)

    assert extracted.source == source.strip()
    assert extracted.scene_class == "TriangleProof"


def test_raw_manim_scene_rejects_unsafe_import_before_rendering() -> None:
    source = VALID_SCENE.replace("from manim import *", "import os\nfrom manim import *")

    with pytest.raises(SceneValidationError, match="import 'os' is not allowed"):
        extract_and_validate_raw_scene(source)


def test_accepts_voiceover_scene_with_three_timed_narration_blocks() -> None:
    extracted = extract_and_validate_scene(
        f"```python\n{VOICEOVER_SCENE}```",
        voiceover=True,
    )

    assert extracted.source == VOICEOVER_SCENE.rstrip()


@pytest.mark.parametrize(
    ("source", "voiceover", "message"),
    [
        (VALID_SCENE, True, "inherit directly from VoiceoverScene"),
        (VOICEOVER_SCENE, False, "inherit directly from Scene"),
        (
            VOICEOVER_SCENE.replace(
                "from manim_voiceover import VoiceoverScene",
                "import os\nfrom manim_voiceover import VoiceoverScene",
            ),
            True,
            "import 'os' is not allowed",
        ),
        (
            VOICEOVER_SCENE.replace(
                "        with self.voiceover",
                "        open('/tmp/x')\n        with self.voiceover",
                1,
            ),
            True,
            "call 'open' is not allowed",
        ),
    ],
)
def test_voiceover_validation_rejects_wrong_mode_and_unsafe_code(
    source: str,
    voiceover: bool,
    message: str,
) -> None:
    with pytest.raises(SceneValidationError, match=message):
        extract_and_validate_scene(f"```python\n{source}```", voiceover=voiceover)


@pytest.mark.parametrize(
    "unsafe_import",
    [
        "from manim_voiceover import os",
        "from manim_voiceover.services.elevenlabs import os",
        "from manim_voiceover.services.elevenlabs import *",
        "from .manim_voiceover import VoiceoverScene",
    ],
)
def test_voiceover_validation_rejects_transitive_or_broad_imports(
    unsafe_import: str,
) -> None:
    source = VOICEOVER_SCENE.replace(
        "from manim_voiceover import VoiceoverScene",
        f"from manim_voiceover import VoiceoverScene\n{unsafe_import}",
    )

    with pytest.raises(SceneValidationError, match="not allowed"):
        extract_and_validate_scene(f"```python\n{source}```", voiceover=True)


def test_validation_rejects_aliasing_dynamic_execution() -> None:
    source = VALID_SCENE.replace(
        "    def construct(self):",
        "    def construct(self):\n        run = exec\n        run('print(1)')",
    )

    with pytest.raises(SceneValidationError, match="exec"):
        extract_and_validate_scene(f"```python\n{source}```")


def test_voiceover_validation_requires_three_to_six_blocks_and_tracker_duration() -> None:
    one_block = VOICEOVER_SCENE.split(
        '        with self.voiceover(text="Move it right.") as tracker:'
    )[0]
    one_block += "\n"

    with pytest.raises(SceneValidationError, match="3 to 6 voiceover blocks"):
        extract_and_validate_scene(f"```python\n{one_block}```", voiceover=True)

    without_duration = VOICEOVER_SCENE.replace("tracker.duration", "1")
    with pytest.raises(SceneValidationError, match="tracker.duration"):
        extract_and_validate_scene(f"```python\n{without_duration}```", voiceover=True)


def test_voiceover_validation_disables_optional_whisper_transcription() -> None:
    default_transcription = VOICEOVER_SCENE.replace(
        "                transcription_model=None,\n",
        "",
    )

    with pytest.raises(SceneValidationError, match="transcription_model=None"):
        extract_and_validate_scene(
            f"```python\n{default_transcription}```",
            voiceover=True,
        )


def test_voiceover_validation_rejects_deprecated_elevenlabs_model() -> None:
    deprecated_model = VOICEOVER_SCENE.replace(
        'model="eleven_multilingual_v2"',
        'model="eleven_monolingual_v1"',
    )

    with pytest.raises(SceneValidationError, match="eleven_multilingual_v2"):
        extract_and_validate_scene(
            f"```python\n{deprecated_model}```",
            voiceover=True,
        )


def test_rejects_ambiguous_multiple_code_fences() -> None:
    response = f"```python\n{VALID_SCENE}```\n```python\n{VALID_SCENE}```"

    with pytest.raises(ExtractionError, match="exactly one Python code fence"):
        extract_and_validate_scene(response)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("import os\n" + VALID_SCENE, "import 'os' is not allowed"),
        (VALID_SCENE + "\nopen('/tmp/file', 'w')", "call 'open' is not allowed"),
        (
            VALID_SCENE + "\n__builtins__['open']('/tmp/file', 'w')",
            "not allowed",
        ),
        (
            VALID_SCENE + "\ngetattr(__builtins__, 'ev' + 'al')('1 + 1')",
            "not allowed",
        ),
        (VALID_SCENE + "\n# null byte:\x00", "not valid Python"),
        ("class GeneratedLesson(Scene)\n    pass", "not valid Python"),
        (
            "from manim import *\nclass WrongName(Scene):\n    pass",
            "class named GeneratedLesson",
        ),
    ],
)
def test_rejects_invalid_or_unsafe_generated_code(source: str, message: str) -> None:
    with pytest.raises(SceneValidationError, match=message):
        extract_and_validate_scene(f"```python\n{source}\n```")


def test_reports_parser_value_error_as_a_parse_failure() -> None:
    with (
        patch("math_tutor.generated_lesson.ast.parse", side_effect=ValueError("bad source")),
        pytest.raises(SceneValidationError, match="not valid Python") as caught,
    ):
        extract_and_validate_scene(f"```python\n{VALID_SCENE}```")

    assert caught.value.diagnostics == {"failure_stage": "parse", "line": None}


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
    assert outcome.narration_diagnostics == {
        "inference_path": "base_model",
        "inference_model": "Qwen/Qwen3-4B",
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
        "model": "Qwen/Qwen3-4B",
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
    validator = SequenceValidator(
        [failed_report, ValidationReport("media", ValidationStatus.PASS)]
    )
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
    assert outcome.video_path.read_bytes() == b"video-repair-123-attempt-1"
    assert outcome.validation_diagnostics == {
        "attempt_count": 2,
        "repair_count": 1,
        "infrastructure_retry_count": 0,
        "selected_attempt": 1,
        "validation_status": "pass",
    }
    job_dir = tmp_path / "artifacts" / "repair-123"
    assert json.loads((job_dir / "attempts" / "0" / "validation.json").read_text())[
        "status"
    ] == "fail"
    assert json.loads((job_dir / "attempts" / "1" / "validation.json").read_text())[
        "status"
    ] == "pass"
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
        "inference_path": "base_model",
        "model": "Qwen/Qwen3-4B",
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
        validator=SequenceValidator(
            [ValidationReport("media", ValidationStatus.PASS)]
        ),
        max_repair_attempts=1,
    )

    pipeline.render("source-repair-123")

    assert len(generator.prompts) == 2
    assert "source_admission_failed" in generator.prompts[1]
    assert renderer.calls == ["source-repair-123-attempt-1"]
    first_validation = json.loads(
        (
            tmp_path
            / "artifacts"
            / "source-repair-123"
            / "attempts"
            / "0"
            / "validation.json"
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
        validator=SequenceValidator(
            [ValidationReport("media", ValidationStatus.PASS)]
        ),
    )

    outcome = pipeline.render("initial-pass-123")

    assert len(generator.prompts) == 1
    assert renderer.calls == ["initial-pass-123-attempt-0"]
    assert outcome.validation_diagnostics == {
        "attempt_count": 1,
        "repair_count": 0,
        "infrastructure_retry_count": 0,
        "selected_attempt": 0,
        "validation_status": "pass",
    }


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
    assert not (
        tmp_path / "artifacts" / "no-repair-123" / "attempts" / "1"
    ).exists()


def test_pipeline_does_not_use_model_repair_for_render_failures(tmp_path: Path) -> None:
    generator = RecordingGenerator(_generation(f"```python\n{VALID_SCENE}```"))
    pipeline = GeneratedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        prompt="Explain a derivative.",
        generator=generator,
        renderer=FailingSourceRenderer(),
        validator=SequenceValidator(
            [ValidationReport("media", ValidationStatus.PASS)]
        ),
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
        validator=SequenceValidator(
            [ValidationReport("media", ValidationStatus.PASS)]
        ),
    )

    outcome = pipeline.render("render-retry-123")

    assert len(generator.prompts) == 1
    assert renderer.calls == [
        "render-retry-123-attempt-0",
        "render-retry-123-attempt-0-retry-1",
    ]
    assert outcome.validation_diagnostics == {
        "attempt_count": 1,
        "repair_count": 0,
        "infrastructure_retry_count": 1,
        "selected_attempt": 0,
        "validation_status": "pass",
    }
    manifest = json.loads(
        (
            tmp_path
            / "artifacts"
            / "render-retry-123"
            / "attempts"
            / "0"
            / "attempt.json"
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
        validator=SequenceValidator(
            [failure, ValidationReport("media", ValidationStatus.PASS)]
        ),
    )

    outcome = pipeline.render("retry-before-repair-123")

    assert outcome.validation_diagnostics == {
        "attempt_count": 2,
        "repair_count": 1,
        "infrastructure_retry_count": 1,
        "selected_attempt": 1,
        "validation_status": "pass",
    }
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
        status=ValidationStatus.ERROR,
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

    assert caught.value.diagnostics["validation_status"] == "error"
    assert caught.value.diagnostics["attempt_count"] == 1
    assert caught.value.diagnostics["repair_count"] == 0
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

    with pytest.raises(ExtractionError) as caught:
        pipeline.render("failed-123")

    job_dir = tmp_path / "artifacts" / "failed-123"
    assert (job_dir / "attempts" / "0" / "raw_response.txt").read_text() == (
        "I cannot provide code."
    )
    assert (job_dir / "attempts" / "1" / "raw_response.txt").read_text() == (
        "I cannot provide code."
    )
    metadata = json.loads((job_dir / "attempts" / "1" / "generation.json").read_text())
    assert metadata["status"] == "extraction_failed"
    assert metadata["model"] == "Qwen/Qwen3-4B"
    assert caught.value.diagnostics["attempt_count"] == 2
    assert caught.value.diagnostics["repair_count"] == 1
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


def test_voiceover_fallback_returns_primary_with_ready_narration(tmp_path: Path) -> None:
    outcome = RenderOutcome(tmp_path / "voice.mp4", "voice", 1, "rendered")
    primary = RecordingPromptRenderer(outcome)
    fallback = RecordingPromptRenderer(
        RenderOutcome(tmp_path / "silent.mp4", "silent", 1, "rendered")
    )

    result = VoiceoverFallbackRenderer(primary=primary, fallback=fallback).render(
        "job-1", "Explain limits"
    )

    assert result.narration_status is NarrationStatus.READY
    assert primary.calls == [("job-1", "Explain limits")]
    assert fallback.calls == []


def test_voiceover_fallback_retries_render_failure_as_silent_lesson(tmp_path: Path) -> None:
    primary = RecordingPromptRenderer(RenderFailed("ElevenLabs failed"))
    silent = RenderOutcome(tmp_path / "silent.mp4", "silent", 2, "rendered")
    fallback = RecordingPromptRenderer(silent)

    result = VoiceoverFallbackRenderer(primary=primary, fallback=fallback).render(
        "job-2", "Explain limits"
    )

    assert result.video_path == silent.video_path
    assert result.narration_status is NarrationStatus.UNAVAILABLE
    assert result.narration_diagnostics == {"voiceover_error": "ElevenLabs failed"}
    assert fallback.calls == [("job-2-silent", "Explain limits")]


def test_voiceover_fallback_does_not_retry_generation_failure(tmp_path: Path) -> None:
    primary = RecordingPromptRenderer(GeneratedLessonError("invalid scene"))
    fallback = RecordingPromptRenderer(
        RenderOutcome(tmp_path / "silent.mp4", "silent", 1, "rendered")
    )

    with pytest.raises(GeneratedLessonError, match="invalid scene"):
        VoiceoverFallbackRenderer(primary=primary, fallback=fallback).render(
            "job-3", "Explain limits"
        )

    assert fallback.calls == []


def test_generation_fallback_uses_base_model_and_reports_specialist_failure(
    tmp_path: Path,
) -> None:
    primary = RecordingPromptRenderer(GeneratedLessonError("specialist timed out"))
    fallback_outcome = RenderOutcome(
        tmp_path / "base.mp4",
        "base",
        1,
        "rendered",
        narration_diagnostics={
            "inference_path": "base_model",
            "inference_model": "Qwen/Qwen3-4B",
        },
    )
    fallback = RecordingPromptRenderer(fallback_outcome)

    result = GenerationFallbackRenderer(primary=primary, fallback=fallback).render(
        "job-4", "Explain limits"
    )

    assert result.narration_diagnostics == {
        "inference_path": "base_model",
        "inference_model": "Qwen/Qwen3-4B",
        "routing_fallback": "base_model",
        "specialist_error": "specialist timed out",
    }
    assert fallback.calls == [("job-4-base", "Explain limits")]


def test_specialist_guidance_is_normalized_by_base_pipeline(tmp_path: Path) -> None:
    specialist_draft = "```python\nfrom manim import *\nclass Draft(Scene):\n    pass\n```"
    specialist = RecordingGenerator(_generation(specialist_draft, model="intermediate"))
    normalized = RenderOutcome(
        tmp_path / "normalized.mp4",
        "voiceover",
        1,
        "rendered",
        narration_diagnostics={
            "inference_path": "base_model",
            "inference_model": "Qwen/Qwen3-4B",
        },
    )
    normalizer = RecordingPromptRenderer(normalized)
    pipeline = SpecialistGuidedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        specialist=specialist,
        normalizer=normalizer,
    )

    result = pipeline.render("job-5", "Explain completing the square visually.")

    assert specialist.prompts == ["Explain completing the square visually."]
    assert normalizer.calls[0][0] == "job-5-normalized"
    normalization_prompt = normalizer.calls[0][1]
    assert normalization_prompt is not None
    assert "Explain completing the square visually." in normalization_prompt
    assert specialist_draft in normalization_prompt
    assert "mathematical and visual guidance" in normalization_prompt
    assert result.narration_diagnostics == {
        "inference_path": "lora_adapter_with_base_normalizer",
        "inference_model": "Qwen/Qwen3-4B",
        "specialist_model": "intermediate",
        "normalization_model": "Qwen/Qwen3-4B",
        "specialist_elapsed_seconds": 0.5,
        "specialist_completion_tokens": 20,
    }
    job_dir = tmp_path / "artifacts" / "job-5"
    assert (job_dir / "prompt.txt").read_text() == ("Explain completing the square visually.")
    assert (job_dir / "specialist_response.txt").read_text() == specialist_draft
    metadata = json.loads((job_dir / "specialist_generation.json").read_text())
    assert metadata["status"] == "generated"
    assert metadata["model"] == "intermediate"


def test_specialist_failure_falls_back_once_to_direct_base_generation(
    tmp_path: Path,
) -> None:
    specialist = RecordingGenerator(ProviderError("specialist timed out"))
    base = RenderOutcome(
        tmp_path / "base.mp4",
        "voiceover",
        1,
        "rendered",
        narration_diagnostics={
            "inference_path": "base_model",
            "inference_model": "Qwen/Qwen3-4B",
        },
    )
    normalizer = RecordingPromptRenderer(base)
    pipeline = SpecialistGuidedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        specialist=specialist,
        normalizer=normalizer,
    )

    result = pipeline.render("job-6", "Explain limits visually.")

    assert normalizer.calls == [("job-6-base", "Explain limits visually.")]
    assert result.narration_diagnostics == {
        "inference_path": "base_model",
        "inference_model": "Qwen/Qwen3-4B",
        "routing_fallback": "base_model",
        "specialist_model": "intermediate",
        "specialist_error": "Modal specialist generation could not be completed",
    }
    metadata = json.loads(
        (tmp_path / "artifacts" / "job-6" / "specialist_generation.json").read_text()
    )
    assert metadata["status"] == "provider_failed"
    assert metadata["failure_stage"] == "specialist_generation"


def test_normalization_failure_retries_original_prompt_through_base_model(
    tmp_path: Path,
) -> None:
    specialist = RecordingGenerator(_generation("raw scene", model="advanced"))
    base = RenderOutcome(
        tmp_path / "base.mp4",
        "voiceover",
        1,
        "rendered",
        narration_diagnostics={
            "inference_path": "base_model",
            "inference_model": "Qwen/Qwen3-4B",
        },
    )
    normalizer = SequencePromptRenderer(
        [SceneValidationError("normalized scene was invalid"), base]
    )
    pipeline = SpecialistGuidedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        specialist=specialist,
        normalizer=normalizer,
    )

    result = pipeline.render("job-7", "Explain eigenvectors visually.")

    assert normalizer.calls[0][0] == "job-7-normalized"
    assert normalizer.calls[1] == ("job-7-base", "Explain eigenvectors visually.")
    assert result.narration_diagnostics == {
        "inference_path": "base_model",
        "inference_model": "Qwen/Qwen3-4B",
        "routing_fallback": "base_model",
        "specialist_model": "advanced",
        "normalization_error": "normalized scene was invalid",
    }


def test_failed_base_fallback_reports_pipeline_context(tmp_path: Path) -> None:
    specialist = RecordingGenerator(ProviderError("specialist timed out"))
    normalizer = RecordingPromptRenderer(
        SceneValidationError(
            "base scene was invalid",
            diagnostics={"failure_stage": "validation"},
        )
    )
    pipeline = SpecialistGuidedLessonPipeline(
        artifact_root=tmp_path / "artifacts",
        specialist=specialist,
        normalizer=normalizer,
    )

    with pytest.raises(GeneratedLessonError) as caught:
        pipeline.render("job-8", "Explain tensors visually.")

    assert str(caught.value) == "base model could not generate the final lesson"
    assert caught.value.diagnostics == {
        "failure_stage": "validation",
        "pipeline_stage": "base_fallback",
        "specialist_model": "intermediate",
    }
