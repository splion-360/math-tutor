"""Verify Modal visual-model requests and credential-safe failure handling.
Tests cover pinned provenance, frame payloads, timeouts, and unavailable service."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from math_tutor.validation.visual import (
    FrameSample,
    VisualModelTimedOut,
    VisualModelUnavailable,
)
from math_tutor.validation.visual_provider import (
    QWEN3_VL_MODEL,
    QWEN3_VL_REVISION,
    ModalVisualModelClient,
    UnavailableVisualModelClient,
)

MODAL_BASE_URL = "https://workspace--visual.modal.direct/v1"


def _frames(tmp_path: Path) -> tuple[FrameSample, ...]:
    samples = []
    for index, timestamp in enumerate((2.5, 5.0, 7.5), start=1):
        path = tmp_path / f"frame-{index:02d}.png"
        path.write_bytes(b"png")
        samples.append(
            FrameSample(
                id=f"frame-{index:02d}",
                timestamp_seconds=timestamp,
                path=path,
                sha256="a" * 64,
            )
        )
    return tuple(samples)


def test_visual_client_sends_pinned_multimodal_schema_request(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["authorization"] = request.headers["authorization"]
        observed["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "visual-123",
                "model": QWEN3_VL_MODEL,
                "choices": [{"message": {"content": '{"status":"pass","findings":[]}'}}],
            },
        )

    client = ModalVisualModelClient(
        base_url=MODAL_BASE_URL,
        api_key="modal-secret",
        transport=httpx.MockTransport(handler),
    )

    result = client.inspect(prompt="Inspect only visible evidence.", frames=_frames(tmp_path))

    assert observed["authorization"] == "Bearer modal-secret"
    payload = observed["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == QWEN3_VL_MODEL
    assert payload["temperature"] == 0
    assert payload["seed"] == 42
    assert payload["response_format"]["json_schema"]["strict"] is True
    schema = payload["response_format"]["json_schema"]["schema"]
    finding_properties = schema["properties"]["findings"]["items"]["properties"]
    assert "uniqueItems" not in finding_properties["frame_ids"]
    assert "uniqueItems" not in finding_properties["regions"]
    content = payload["messages"][1]["content"]
    assert content[0]["text"].endswith("frame-03: 7.500 seconds")
    assert content[1]["image_url"]["url"] == "data:image/png;base64,cG5n"
    assert result.model == QWEN3_VL_MODEL
    assert result.revision == QWEN3_VL_REVISION
    assert result.request_id == "visual-123"
    assert result.content == '{"status":"pass","findings":[]}'


def test_visual_client_maps_timeout_to_bounded_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("provider secret", request=request)

    client = ModalVisualModelClient(
        base_url=MODAL_BASE_URL,
        api_key="modal-secret",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(VisualModelTimedOut, match="deadline") as caught:
        client.inspect(prompt="Inspect.", frames=_frames(tmp_path))

    assert "provider secret" not in str(caught.value)


def test_visual_client_maps_non_success_to_unavailable_without_response_body(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="modal-secret internal trace")

    client = ModalVisualModelClient(
        base_url=MODAL_BASE_URL,
        api_key="modal-secret",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(VisualModelUnavailable, match="HTTP 503") as caught:
        client.inspect(prompt="Inspect.", frames=_frames(tmp_path))

    assert "internal trace" not in str(caught.value)


def test_unconfigured_visual_client_is_explicitly_unavailable(tmp_path: Path) -> None:
    client = UnavailableVisualModelClient()

    with pytest.raises(VisualModelUnavailable, match="not configured"):
        client.inspect(prompt="Inspect.", frames=_frames(tmp_path))

    assert client.model == QWEN3_VL_MODEL
    assert client.revision == QWEN3_VL_REVISION
