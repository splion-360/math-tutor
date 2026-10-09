"""Verify application composition for the shared adapter and validators.
The tests inspect provider and pipeline wiring through injected settings."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from math_tutor.generation.provider import (
    BASE_MODEL,
    SHARED_ADAPTER_MODEL,
    GenerationConfig,
    ModelHealth,
)
from math_tutor.settings import Settings


def test_build_app_without_endpoint_keeps_shared_adapter_unavailable(tmp_path: Path) -> None:
    import math_tutor.main as main

    settings = Settings(
        modal_vllm_base_url=None,
        artifact_root=tmp_path / "artifacts",
    )

    with TestClient(main.build_app(settings)) as client:
        response = client.get("/model/health")

    assert response.json() == {
        "reachable": False,
        "model": SHARED_ADAPTER_MODEL,
        "model_available": False,
        "error": "Modal inference endpoint is not configured",
    }


def test_build_app_wires_shared_adapter_and_parallel_validators(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed_models: list[GenerationConfig] = []
    observed_visual_connections: list[dict[str, object]] = []
    observed_validators: list[tuple[object, ...]] = []
    validator_callbacks: list[object] = []
    observed_renderers: list[dict[str, object]] = []
    observed_narration_providers: list[dict[str, object]] = []
    observed_planners: list[dict[str, object]] = []
    observed_assemblers: list[dict[str, object]] = []
    observed_outcome_processors: list[dict[str, object]] = []
    observed_pipelines: list[dict[str, object]] = []
    observed_services: list[dict[str, object]] = []
    close_callbacks: list[object] = []
    closed: list[str] = []

    class RecordingModelClient:
        def __init__(
            self,
            *,
            api_key: str,
            config: GenerationConfig,
            base_url: str,
            timeout_seconds: float,
        ) -> None:
            assert api_key == "modal-secret"
            assert base_url == "https://workspace--shared.modal.run/v1"
            assert timeout_seconds == 90
            self.config = config
            observed_models.append(config)

        def health(self) -> ModelHealth:
            return ModelHealth(True, self.config.model, True)

        def close(self) -> None:
            closed.append(self.config.model)

    class RecordingVisualClient:
        model = "visual-model"
        revision = "visual-revision"

        def __init__(self, **kwargs: object) -> None:
            observed_visual_connections.append(kwargs)

        def close(self) -> None:
            closed.append("visual")

    class RecordingSuite:
        expected_checks = ("media", "spatial", "visual")

        def __init__(
            self,
            validators: tuple[object, ...],
            report_callback: object,
        ) -> None:
            observed_validators.append(validators)
            validator_callbacks.append(report_callback)

    class RecordingPipeline:
        def __init__(self, **kwargs: object) -> None:
            observed_pipelines.append(kwargs)

    class RecordingRenderer:
        def __init__(self, **kwargs: object) -> None:
            observed_renderers.append(kwargs)

    class RecordingNarrationProvider:
        def __init__(self, **kwargs: object) -> None:
            observed_narration_providers.append(kwargs)

    class RecordingAssembler:
        def __init__(self, **kwargs: object) -> None:
            observed_assemblers.append(kwargs)

    class RecordingPlanner:
        def __init__(self, model: object, **kwargs: object) -> None:
            observed_planners.append({"model": model, **kwargs})

    class RecordingOutcomeProcessor:
        def __init__(self, **kwargs: object) -> None:
            observed_outcome_processors.append(kwargs)

    class RecordingLessonService:
        def __init__(self, **kwargs: object) -> None:
            observed_services.append(kwargs)

    def capture_app(service: object, **kwargs: object) -> object:
        close_callbacks.append(kwargs["close_model"])
        return object()

    import math_tutor.main as main

    monkeypatch.setattr(main, "ModalVllmClient", RecordingModelClient)
    monkeypatch.setattr(main, "ModalVisualModelClient", RecordingVisualClient)
    monkeypatch.setattr(main, "ValidatorSuite", RecordingSuite)
    monkeypatch.setattr(main, "DockerManimRenderer", RecordingRenderer)
    monkeypatch.setattr(main, "ElevenLabsNarrationProvider", RecordingNarrationProvider)
    monkeypatch.setattr(main, "FfmpegMediaAssembler", RecordingAssembler)
    monkeypatch.setattr(main, "ModelNarrationPlanner", RecordingPlanner)
    monkeypatch.setattr(main, "NarrationOutcomeProcessor", RecordingOutcomeProcessor)
    monkeypatch.setattr(main, "GeneratedLessonPipeline", RecordingPipeline)
    monkeypatch.setattr(main, "LessonService", RecordingLessonService)
    monkeypatch.setattr(main, "create_app", capture_app)

    main.build_app(
        Settings(
            modal_vllm_base_url="https://workspace--shared.modal.run/v1",
            modal_vllm_api_key="modal-secret",
            modal_vllm_timeout_seconds=90,
            modal_visual_model_base_url="https://workspace--visual.modal.run/v1",
            modal_visual_model_api_key="visual-secret",
            modal_visual_model_timeout_seconds=17,
            elevenlabs_api_key="elevenlabs-secret",
            elevenlabs_voice_id="narrator-voice",
            artifact_root=tmp_path / "artifacts",
        )
    )

    assert [config.model for config in observed_models] == [
        SHARED_ADAPTER_MODEL,
        BASE_MODEL,
    ]
    assert "VoiceoverScene" not in observed_models[0].system_prompt
    assert observed_models[1].system_prompt == main.NARRATION_SYSTEM_PROMPT
    assert observed_models[1].max_tokens == 512
    assert observed_models[1].response_format == main.NARRATION_RESPONSE_FORMAT
    assert observed_visual_connections == [
        {
            "base_url": "https://workspace--visual.modal.run/v1",
            "api_key": "visual-secret",
            "timeout_seconds": 17,
        }
    ]
    assert [validator.name for validator in observed_validators[0]] == [
        "media",
        "spatial",
        "visual_evidence",
    ]
    assert observed_validators[0][2]._max_model_retries == 1
    assert callable(validator_callbacks[0])
    assert observed_renderers == [
        {
            "artifact_root": tmp_path / "artifacts",
            "scene_path": (
                Path(main.__file__).parent / "rendering" / "scenes" / "pythagorean_theorem.py"
            ),
            "image": main.DEFAULT_MANIM_IMAGE,
            "timeout_seconds": 90,
        }
    ]
    assert observed_narration_providers == [
        {
            "api_key": "elevenlabs-secret",
            "voice_id": "narrator-voice",
            "duration_probe": main.probe_audio_duration,
        }
    ]
    assert len(observed_planners) == 1
    assert observed_planners[0]["model"].config.model == BASE_MODEL
    assert observed_planners[0]["max_retries"] == 1
    assert observed_assemblers == [{"timeout_seconds": 180}]
    assert len(observed_outcome_processors) == 1
    processor = observed_outcome_processors[0]
    assert processor["planner"].__class__ is RecordingPlanner
    assert processor["provider"].__class__ is RecordingNarrationProvider
    assert processor["assembler"].__class__ is RecordingAssembler
    assert processor["duration_probe"] is main.probe_audio_duration
    assert len(observed_pipelines) == 1
    pipeline = observed_pipelines[0]
    assert pipeline["artifact_root"] == tmp_path / "artifacts"
    assert pipeline["prompt"] == main.GENERATED_DEMO_PROMPT
    assert pipeline["inference_path"] == "lora_adapter"
    assert pipeline["generation_provider"] == "modal_vllm"
    assert pipeline["validator"].expected_checks == (
        "media",
        "spatial",
        "visual",
    )
    assert pipeline["max_repair_attempts"] == 1
    assert callable(pipeline["stage_reporter"])
    assert callable(pipeline["render_reporter"])
    assert pipeline["narration_required"] is True
    assert pipeline["captions_required"] is True
    assert pipeline["renderer"].__class__ is RecordingRenderer
    assert pipeline["outcome_processor"].__class__ is RecordingOutcomeProcessor
    assert observed_services[0]["narration_requested"] is True
    close_callback = close_callbacks[0]
    assert callable(close_callback)
    close_callback()
    assert closed == [SHARED_ADAPTER_MODEL, BASE_MODEL, "visual"]
