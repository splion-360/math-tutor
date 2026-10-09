"""Verify application composition for the shared adapter and validators.
The tests inspect provider and pipeline wiring through injected settings."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from math_tutor.generation.provider import (
    SHARED_ADAPTER_MODEL,
    GenerationConfig,
    ModelHealth,
)
from math_tutor.jobs import LessonService
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
    observed_pipelines: list[dict[str, object]] = []
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
            closed.append("generation")

    class RecordingVisualClient:
        model = "visual-model"
        revision = "visual-revision"

        def __init__(self, **kwargs: object) -> None:
            observed_visual_connections.append(kwargs)

        def close(self) -> None:
            closed.append("visual")

    class RecordingSuite:
        expected_checks = ("media", "spatial", "visual")

        def __init__(self, validators: tuple[object, ...]) -> None:
            observed_validators.append(validators)

    class RecordingPipeline:
        def __init__(self, **kwargs: object) -> None:
            observed_pipelines.append(kwargs)

    def capture_app(service: LessonService, **kwargs: object) -> object:
        close_callbacks.append(kwargs["close_model"])
        return object()

    import math_tutor.main as main

    monkeypatch.setattr(main, "ModalVllmClient", RecordingModelClient)
    monkeypatch.setattr(main, "ModalVisualModelClient", RecordingVisualClient)
    monkeypatch.setattr(main, "ValidatorSuite", RecordingSuite)
    monkeypatch.setattr(main, "GeneratedLessonPipeline", RecordingPipeline)
    monkeypatch.setattr(main, "create_app", capture_app)

    main.build_app(
        Settings(
            modal_vllm_base_url="https://workspace--shared.modal.run/v1",
            modal_vllm_api_key="modal-secret",
            modal_vllm_timeout_seconds=90,
            modal_visual_model_base_url="https://workspace--visual.modal.run/v1",
            modal_visual_model_api_key="visual-secret",
            modal_visual_model_timeout_seconds=17,
            artifact_root=tmp_path / "artifacts",
        )
    )

    assert [config.model for config in observed_models] == [SHARED_ADAPTER_MODEL]
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
    close_callback = close_callbacks[0]
    assert callable(close_callback)
    close_callback()
    assert closed == ["generation", "visual"]
