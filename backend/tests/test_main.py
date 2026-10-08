"""Verify application composition for Modal, specialist adapters, and narration.
The tests inspect provider and renderer wiring through injected settings."""

from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep

from fastapi.testclient import TestClient

from math_tutor.domain import Difficulty, LessonStage, LessonStatus
from math_tutor.generation.provider import (
    FROZEN_MODEL,
    SPECIALIST_SYSTEM_PROMPT,
    GenerationConfig,
    GenerationResult,
    ModelHealth,
    ProviderError,
)
from math_tutor.jobs import LessonService, RenderOutcome
from math_tutor.rendering.manim import VOICEOVER_MANIM_IMAGE, RenderError
from math_tutor.settings import Settings


def test_build_app_without_secret_keeps_provider_unavailable(tmp_path: Path) -> None:
    import math_tutor.main as main

    settings = Settings(
        _env_file=None,
        modal_vllm_base_url=None,
        artifact_root=tmp_path / "artifacts",
    )

    with TestClient(main.build_app(settings)) as client:
        response = client.get("/model/health")

    assert response.json() == {
        "reachable": False,
        "model": FROZEN_MODEL,
        "model_available": False,
        "error": "Modal inference endpoint is not configured",
    }


def test_build_app_uses_modal_base_model_by_default_and_keeps_specialists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed_models: list[str] = []
    observed_configs: list[GenerationConfig] = []
    observed_timeouts: list[float] = []

    class RecordingModalClient:
        def __init__(
            self,
            *,
            api_key: str,
            config: GenerationConfig,
            base_url: str,
            timeout_seconds: float = 60,
        ) -> None:
            assert api_key == "modal-secret"
            assert base_url == "https://workspace--qwen.modal.direct/v1"
            observed_models.append(config.model)
            observed_configs.append(config)
            observed_timeouts.append(timeout_seconds)
            self.config = config

        def generate(self, prompt: str) -> GenerationResult:
            raise AssertionError("generation is not part of this test")

        def health(self) -> ModelHealth:
            return ModelHealth(True, self.config.model, True)

        def close(self) -> None:
            return None

    import math_tutor.main as main

    monkeypatch.setattr(main, "ModalVllmClient", RecordingModalClient)
    settings = Settings(
        _env_file=None,
        modal_vllm_base_url="https://workspace--qwen.modal.direct/v1",
        modal_vllm_api_key="modal-secret",
        modal_vllm_timeout_seconds=90,
        modal_specialist_timeout_seconds=60,
        artifact_root=tmp_path / "artifacts",
    )

    with TestClient(main.build_app(settings)) as client:
        health = client.get("/model/health")

    assert health.json() == {
        "reachable": True,
        "model": FROZEN_MODEL,
        "model_available": True,
        "error": None,
    }
    assert observed_models == [FROZEN_MODEL, "foundational", "intermediate", "advanced"]
    assert observed_timeouts == [90, 60, 60, 60]
    assert all(config.max_tokens == 4096 for config in observed_configs)
    assert "VoiceoverScene" not in observed_configs[0].system_prompt
    assert all(config.system_prompt == SPECIALIST_SYSTEM_PROMPT for config in observed_configs[1:])


def test_build_app_configures_voiceover_generation_when_elevenlabs_is_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed_clients: list[GenerationConfig] = []
    observed_renderers: list[dict[str, object]] = []

    class RecordingModelClient:
        def __init__(
            self,
            *,
            api_key: str,
            config: GenerationConfig,
            base_url: str,
            timeout_seconds: float = 60,
        ) -> None:
            observed_clients.append(config)
            self.config = config

        def generate(self, prompt: str) -> GenerationResult:
            raise AssertionError("generation is not part of this test")

        def health(self) -> ModelHealth:
            return ModelHealth(True, self.config.model, True)

        def close(self) -> None:
            return None

    class RecordingDockerRenderer:
        def __init__(self, **kwargs: object) -> None:
            observed_renderers.append(kwargs)

        def render(self, job_id: str):
            raise AssertionError("rendering is not part of this test")

        def render_source(self, job_id: str, source: str, scene_class: str):
            raise AssertionError("rendering is not part of this test")

    import math_tutor.main as main

    monkeypatch.setattr(main, "ModalVllmClient", RecordingModelClient)
    monkeypatch.setattr(main, "DockerManimRenderer", RecordingDockerRenderer)
    settings = Settings(
        _env_file=None,
        modal_vllm_base_url="https://workspace--qwen.modal.direct/v1",
        modal_vllm_api_key="modal-secret",
        elevenlabs_api_key="eleven-secret",
        elevenlabs_voice_id="voice-123",
        artifact_root=tmp_path / "artifacts",
    )

    with TestClient(main.build_app(settings)) as client:
        assert client.get("/model/health").status_code == 200

    assert len(observed_clients) == 5
    base_configs = [config for config in observed_clients if config.model == FROZEN_MODEL]
    voiceover_config = next(
        config for config in base_configs if "VoiceoverScene" in config.system_prompt
    )
    silent_config = next(
        config for config in base_configs if "VoiceoverScene" not in config.system_prompt
    )
    assert 'voice_id="voice-123"' in voiceover_config.system_prompt
    assert silent_config.model == voiceover_config.model
    voiceover_renderer = next(
        item for item in observed_renderers if item.get("image") == VOICEOVER_MANIM_IMAGE
    )
    assert voiceover_renderer["network"] == "bridge"
    assert voiceover_renderer["environment"] == {"ELEVEN_API_KEY": "eleven-secret"}
    assert voiceover_renderer["require_audio"] is True


def test_specialist_and_voiceover_fallbacks_report_against_the_root_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = tmp_path / "silent.mp4"
    captured_service: list[LessonService] = []

    class FailingSpecialistClient:
        def __init__(
            self,
            *,
            api_key: str,
            config: GenerationConfig,
            base_url: str,
            timeout_seconds: float = 60,
        ) -> None:
            self.config = config

        def generate(self, prompt: str) -> GenerationResult:
            raise ProviderError("specialist unavailable")

        def health(self) -> ModelHealth:
            return ModelHealth(True, self.config.model, True)

        def close(self) -> None:
            return None

    class FallbackPipeline:
        def __init__(self, **kwargs: object) -> None:
            self.voiceover = bool(kwargs.get("voiceover", False))
            self.stage_reporter = kwargs["stage_reporter"]

        def render(self, job_id: str, prompt: str | None = None) -> RenderOutcome:
            if self.voiceover:
                raise RenderError("voiceover unavailable")
            self.stage_reporter(job_id, LessonStage.GENERATING_CODE)
            video.write_bytes(b"video")
            return RenderOutcome(video, "silent", 0.1, "rendered")

    import math_tutor.main as main

    def capture_app(service: LessonService, **kwargs: object) -> LessonService:
        captured_service.append(service)
        return service

    monkeypatch.setattr(main, "ModalVllmClient", FailingSpecialistClient)
    monkeypatch.setattr(main, "GeneratedLessonPipeline", FallbackPipeline)
    monkeypatch.setattr(main, "create_app", capture_app)
    settings = Settings(
        _env_file=None,
        modal_vllm_base_url="https://workspace--qwen.modal.direct/v1",
        modal_vllm_api_key="modal-secret",
        elevenlabs_api_key="eleven-secret",
        artifact_root=tmp_path / "artifacts",
    )

    main.build_app(settings)
    service = captured_service[0]
    try:
        submitted = service.submit(
            "Explain fractions.",
            difficulty=Difficulty.FOUNDATIONAL,
            routing_policy="explicit_difficulty",
        )
        deadline = monotonic() + 2
        while monotonic() < deadline:
            lesson = service.get(submitted.id)
            if lesson.status in {LessonStatus.READY, LessonStatus.FAILED}:
                break
            sleep(0.01)
    finally:
        service.close()

    assert lesson.status is LessonStatus.READY
    assert lesson.stage is LessonStage.READY
    assert lesson.narration_status.value == "unavailable"
