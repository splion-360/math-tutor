"""Verify ElevenLabs narration requests and normalized audio artifacts.
The tests keep provider errors bounded and credentials out of diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from math_tutor.rendering.elevenlabs import ElevenLabsError, ElevenLabsNarrationProvider
from math_tutor.rendering.narration import NarrationPlan, NarrationSegment


class FakeResponse:
    def __init__(self, content: bytes = b"fake mp3", error: Exception | None = None) -> None:
        self.content = content
        self._error = error

    def raise_for_status(self) -> None:
        if self._error:
            raise self._error


class RecordingTransport:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"url": url, **kwargs})
        return self.responses.pop(0)


def test_provider_synthesizes_ordered_segments_with_measured_durations(tmp_path: Path) -> None:
    transport = RecordingTransport([FakeResponse(b"first"), FakeResponse(b"second")])
    measured = {"000-intro.mp3": 1.25, "001-proof.mp3": 2.75}
    provider = ElevenLabsNarrationProvider(
        api_key="secret-key",
        voice_id="voice-123",
        model_id="eleven_multilingual_v2",
        transport=transport,
        duration_probe=lambda path: measured[path.name],
    )
    plan = NarrationPlan(
        lesson_id="lesson",
        segments=(
            NarrationSegment("intro", "Start here.", "intro-visible"),
            NarrationSegment("proof", "Now prove it.", "proof-visible"),
        ),
    )

    result = provider.synthesize(plan, tmp_path / "audio")

    assert [segment.id for segment in result.segments] == ["intro", "proof"]
    assert [segment.duration_seconds for segment in result.segments] == [1.25, 2.75]
    assert [segment.audio_path.name for segment in result.segments] == [
        "000-intro.mp3",
        "001-proof.mp3",
    ]
    assert result.provider == "elevenlabs"
    assert result.model_id == "eleven_multilingual_v2"
    assert transport.requests[0] == {
        "url": "https://api.elevenlabs.io/v1/text-to-speech/voice-123/stream",
        "headers": {"xi-api-key": "secret-key", "accept": "audio/mpeg"},
        "json": {"text": "Start here.", "model_id": "eleven_multilingual_v2"},
        "params": {"output_format": "mp3_44100_128"},
        "timeout": 30.0,
    }
    assert (tmp_path / "audio" / "000-intro.mp3").read_bytes() == b"first"


def test_provider_failure_does_not_expose_api_key(tmp_path: Path) -> None:
    transport = RecordingTransport([FakeResponse(error=RuntimeError("secret-key rejected"))])
    provider = ElevenLabsNarrationProvider(
        api_key="secret-key",
        voice_id="voice-123",
        transport=transport,
        duration_probe=lambda _: 1.0,
    )
    plan = NarrationPlan(
        lesson_id="lesson",
        segments=(NarrationSegment("intro", "Start.", "intro-visible"),),
    )

    with pytest.raises(ElevenLabsError) as captured:
        provider.synthesize(plan, tmp_path / "audio")

    assert "intro" in str(captured.value)
    assert "secret-key" not in str(captured.value)
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_provider_rejects_unsafe_segment_id_before_writing(tmp_path: Path) -> None:
    provider = ElevenLabsNarrationProvider(
        api_key="secret-key",
        voice_id="voice-123",
        transport=RecordingTransport([]),
        duration_probe=lambda _: 1.0,
    )
    plan = NarrationPlan(
        lesson_id="lesson",
        segments=(NarrationSegment("../escape", "Start.", "intro-visible"),),
    )

    with pytest.raises(ElevenLabsError, match="safe artifact name"):
        provider.synthesize(plan, tmp_path / "audio")
