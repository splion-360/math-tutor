"""Verify environment-backed service configuration and safe defaults.
Tests cover credentials, provider connections, and operational policy."""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr

from math_tutor.settings import Settings, get_settings


def test_settings_loads_secret_and_keeps_operational_code_defaults(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "nebius-secret")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "elevenlabs-secret")

    settings = Settings(_env_file=None)

    assert settings.nebius_api_key == SecretStr("nebius-secret")
    assert settings.elevenlabs_api_key == SecretStr("elevenlabs-secret")
    assert "nebius-secret" not in repr(settings)
    assert "elevenlabs-secret" not in repr(settings)
    assert settings.nebius_base_url == "https://api.tokenfactory.nebius.com/v1"
    assert settings.artifact_root == Path("artifacts")
    assert settings.render_timeout_seconds == 90
    assert settings.max_pending_jobs == 8
    assert settings.validation_max_repair_attempts == 1


def test_get_settings_returns_one_cached_settings_object(monkeypatch) -> None:
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second


def test_settings_loads_optional_modal_vllm_connection(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_VLLM_BASE_URL", "https://workspace--qwen.modal.direct/v1")
    monkeypatch.setenv("MODAL_VLLM_API_KEY", "modal-secret")

    settings = Settings(_env_file=None)

    assert settings.modal_vllm_base_url == "https://workspace--qwen.modal.direct/v1"
    assert settings.modal_vllm_api_key == SecretStr("modal-secret")
    assert settings.modal_vllm_timeout_seconds == 120
    assert settings.modal_specialist_timeout_seconds == 60
    assert "modal-secret" not in repr(settings)
