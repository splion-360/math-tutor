"""Verify Modal generation requests, responses, health checks, and failures.
The tests exercise the provider boundary without external network access."""

from __future__ import annotations

import json

import httpx
import pytest

from math_tutor.generation.provider import (
    SHARED_ADAPTER_MODEL,
    VOICEOVER_SYSTEM_PROMPT,
    GenerationConfig,
    ModalVllmClient,
    ProviderError,
)

MODAL_BASE_URL = "https://workspace--qwen.modal.direct/v1"


def test_generate_sends_frozen_decoding_config_and_preserves_usage() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers["authorization"]
        observed["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "model": SHARED_ADAPTER_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "```python\nfrom manim import *\n```",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 23,
                    "completion_tokens": 11,
                    "total_tokens": 34,
                },
            },
        )

    config = GenerationConfig()
    client = ModalVllmClient(
        api_key="modal-secret",
        config=config,
        base_url=MODAL_BASE_URL,
        transport=httpx.MockTransport(handler),
    )

    result = client.generate("Explain a derivative visually.")

    assert observed == {
        "url": "https://workspace--qwen.modal.direct/v1/chat/completions",
        "authorization": "Bearer modal-secret",
        "payload": {
            "model": SHARED_ADAPTER_MODEL,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": "Explain a derivative visually."},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 4096,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    assert result.content == "```python\nfrom manim import *\n```"
    assert result.model == SHARED_ADAPTER_MODEL
    assert result.request_id == "chatcmpl-123"
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 23
    assert result.usage.completion_tokens == 11
    assert result.usage.total_tokens == 34
    assert result.elapsed_seconds >= 0
    assert json.loads(result.provider_response)["id"] == "chatcmpl-123"


def test_system_prompt_matches_the_shared_adapter_training_contract() -> None:
    config = GenerationConfig()

    assert config.system_prompt == (
        "You generate concise, runnable Manim Community Edition Python scenes for math "
        "tutoring. Return only Python code."
    )


def test_generate_includes_optional_structured_response_format() -> None:
    observed: dict[str, object] = {}
    response_format: dict[str, object] = {
        "type": "json_schema",
        "json_schema": {"name": "narration", "schema": {"type": "object"}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-json",
                "choices": [
                    {
                        "message": {"content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = ModalVllmClient(
        api_key="modal-secret",
        config=GenerationConfig(response_format=response_format),
        base_url=MODAL_BASE_URL,
        transport=httpx.MockTransport(handler),
    )

    client.generate("Return JSON.")

    assert observed["response_format"] == response_format


def test_voiceover_prompt_requires_timed_narration_blocks() -> None:
    assert "VoiceoverScene" in VOICEOVER_SYSTEM_PROMPT
    assert "ElevenLabsService" in VOICEOVER_SYSTEM_PROMPT
    assert "3 to 6" in VOICEOVER_SYSTEM_PROMPT
    assert "tracker.duration" in VOICEOVER_SYSTEM_PROMPT
    assert "transcription_model=None" in VOICEOVER_SYSTEM_PROMPT
    assert 'model="eleven_multilingual_v2"' in VOICEOVER_SYSTEM_PROMPT


def test_health_distinguishes_reachable_api_from_unavailable_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "Qwen/Qwen3-30B-A3B-Instruct-2507"}]},
        )

    client = ModalVllmClient(
        api_key="modal-secret",
        config=GenerationConfig(),
        base_url=MODAL_BASE_URL,
        transport=httpx.MockTransport(handler),
    )

    health = client.health()

    assert health.reachable is True
    assert health.model == SHARED_ADAPTER_MODEL
    assert health.model_available is False
    assert health.error is None


def test_health_follows_modal_web_server_startup_redirect() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(303, headers={"location": str(request.url)})
        return httpx.Response(200, json={"data": [{"id": SHARED_ADAPTER_MODEL}]})

    client = ModalVllmClient(
        api_key="modal-secret",
        config=GenerationConfig(),
        base_url=MODAL_BASE_URL,
        transport=httpx.MockTransport(handler),
    )

    health = client.health()

    assert requests == 2
    assert health.reachable is True
    assert health.model_available is True


def test_provider_error_does_not_expose_credentials_or_response_body() -> None:
    secret = "modal-do-not-leak"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"rejected credential {secret}")

    client = ModalVllmClient(
        api_key=secret,
        config=GenerationConfig(),
        base_url=MODAL_BASE_URL,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ProviderError) as caught:
        client.generate("Prompt")

    assert str(caught.value) == "Modal vLLM generation request failed with HTTP 401"
    assert secret not in str(caught.value)


def test_malformed_success_response_is_a_sanitized_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "chatcmpl-broken", "choices": []})

    client = ModalVllmClient(
        api_key="modal-secret",
        config=GenerationConfig(),
        base_url=MODAL_BASE_URL,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ProviderError, match="malformed response"):
        client.generate("Prompt")


def test_modal_vllm_sends_shared_adapter_to_openai_compatible_endpoint() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["authorization"] = request.headers.get("authorization")
        observed["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-modal-123",
                "model": SHARED_ADAPTER_MODEL,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "```python\npass\n```"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
            },
        )

    config = GenerationConfig(model=SHARED_ADAPTER_MODEL)
    client = ModalVllmClient(
        api_key="modal-token",
        config=config,
        base_url="https://workspace--qwen.modal.direct/v1",
        transport=httpx.MockTransport(handler),
    )

    result = client.generate("Explain a derivative visually.")

    assert observed == {
        "authorization": "Bearer modal-token",
        "payload": {
            "model": SHARED_ADAPTER_MODEL,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": "Explain a derivative visually."},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 4096,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    assert result.model == SHARED_ADAPTER_MODEL
