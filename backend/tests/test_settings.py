"""Verify environment-backed service configuration and safe defaults.
Tests cover credentials, provider connections, and operational policy."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from math_tutor.settings import Settings, get_settings


def test_settings_rejects_more_than_one_repair_attempt() -> None:
    with pytest.raises(ValidationError):
        Settings(validation_max_repair_attempts=2)


def test_settings_loads_secret_and_keeps_operational_code_defaults(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MODAL_VLLM_BASE_URL", "https://workspace--qwen.modal.direct/v1")
    monkeypatch.setenv("MODAL_VLLM_API_KEY", "modal-secret")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "elevenlabs-secret")
    monkeypatch.setenv("MODAL_VLLM_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("MODAL_SPECIALIST_TIMEOUT_SECONDS", "6")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "environment-voice")
    monkeypatch.setenv("ARTIFACT_ROOT", "/tmp/environment-artifacts")
    monkeypatch.setenv("RENDER_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("MAX_PENDING_JOBS", "2")
    monkeypatch.setenv("VALIDATION_MAX_REPAIR_ATTEMPTS", "7")
    monkeypatch.setenv("SPATIAL_UNSAFE_MARGIN", "0.1")
    monkeypatch.setenv("SPATIAL_MAX_WIDTH_RATIO", "0.2")
    monkeypatch.setenv("SPATIAL_MAX_HEIGHT_RATIO", "0.3")
    monkeypatch.setenv("SPATIAL_SEVERE_OVERLAP_RATIO", "0.4")
    monkeypatch.setenv("SPATIAL_PERSISTENT_CHECKPOINTS", "9")

    settings = Settings.from_environment(env_file=None)

    assert settings.modal_vllm_api_key == SecretStr("modal-secret")
    assert "modal-secret" not in repr(settings)
    assert settings.modal_vllm_base_url == "https://workspace--qwen.modal.direct/v1"
    assert settings.artifact_root == Path("artifacts")
    assert settings.render_timeout_seconds == 90
    assert settings.max_pending_jobs == 8
    assert settings.validation_max_repair_attempts == 1
    assert settings.spatial_unsafe_margin == 0.25
    assert settings.spatial_max_width_ratio == 0.9
    assert settings.spatial_max_height_ratio == 0.9
    assert settings.spatial_severe_overlap_ratio == 0.35
    assert settings.spatial_persistent_checkpoints == 2
    assert settings.modal_vllm_timeout_seconds == 15 * 60


def test_get_settings_returns_one_cached_settings_object(monkeypatch) -> None:
    monkeypatch.delenv("MODAL_VLLM_API_KEY", raising=False)
    monkeypatch.delenv("MODAL_VLLM_BASE_URL", raising=False)
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second


def test_settings_loads_optional_modal_vllm_connection(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_VLLM_BASE_URL", "https://workspace--qwen.modal.direct/v1")
    monkeypatch.setenv("MODAL_VLLM_API_KEY", "modal-secret")

    settings = Settings.from_environment(env_file=None)

    assert settings.modal_vllm_base_url == "https://workspace--qwen.modal.direct/v1"
    assert settings.modal_vllm_api_key == SecretStr("modal-secret")
    assert settings.modal_vllm_timeout_seconds == 15 * 60
    assert "modal-secret" not in repr(settings)


def test_settings_loads_separate_visual_model_connection(monkeypatch) -> None:
    monkeypatch.setenv(
        "MODAL_VISUAL_MODEL_BASE_URL",
        "https://workspace--visual.modal.direct/v1",
    )
    monkeypatch.setenv("MODAL_VISUAL_MODEL_API_KEY", "visual-secret")
    monkeypatch.setenv("MODAL_VISUAL_MODEL_TIMEOUT_SECONDS", "30")

    settings = Settings.from_environment(env_file=None)

    assert settings.modal_visual_model_base_url == ("https://workspace--visual.modal.direct/v1")
    assert settings.modal_visual_model_api_key == SecretStr("visual-secret")
    assert settings.modal_visual_model_timeout_seconds == 10 * 60
    assert "visual-secret" not in repr(settings)
