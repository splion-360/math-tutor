"""Generate narration audio and word timing through the ElevenLabs API.
The provider returns normalized narration values to the rendering layer."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import httpx

from math_tutor.rendering.narration import (
    NarrationPlan,
    SynthesizedNarration,
    SynthesizedSegment,
)

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
_SAFE_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class HttpResponse(Protocol):
    """Minimal response contract required by the narration provider."""

    content: bytes

    def raise_for_status(self) -> None:
        """Raise when the response status is unsuccessful."""
        ...


class HttpTransport(Protocol):
    """HTTP boundary used by the narration provider."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, str],
        params: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        """Send one narration synthesis request."""
        ...


class HttpxTransport:
    """HTTPX implementation of the narration transport contract."""

    def __init__(self) -> None:
        self._client = httpx.Client()

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, str],
        params: Mapping[str, str],
        timeout: float,
    ) -> httpx.Response:
        """Send one request through the owned HTTPX client."""
        return self._client.post(
            url,
            headers=headers,
            json=json,
            params=params,
            timeout=timeout,
        )


class ElevenLabsError(RuntimeError):
    """A sanitized narration-provider failure safe for job diagnostics."""


class ElevenLabsNarrationProvider:
    """Synthesize narration segments and record bounded artifact provenance."""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        duration_probe: Callable[[Path], float],
        model_id: str = DEFAULT_MODEL_ID,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        timeout_seconds: float = 30.0,
        transport: HttpTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not _SAFE_ARTIFACT_ID.fullmatch(voice_id):
            raise ValueError("voice_id contains unsupported characters")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._timeout_seconds = timeout_seconds
        self._transport = transport or HttpxTransport()
        self._duration_probe = duration_probe

    def synthesize(
        self,
        plan: NarrationPlan,
        output_dir: Path,
    ) -> SynthesizedNarration:
        """Synthesize every plan segment into an ordered narration result."""
        for segment in plan.segments:
            if not _SAFE_ARTIFACT_ID.fullmatch(segment.id):
                raise ElevenLabsError(
                    f"segment '{segment.id}' cannot be used as a safe artifact name"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        synthesized: list[SynthesizedSegment] = []

        for index, segment in enumerate(plan.segments):
            audio_path = output_dir / f"{index:03d}-{segment.id}.mp3"
            try:
                response = self._transport.post(
                    f"{ELEVENLABS_BASE_URL}/text-to-speech/{self._voice_id}/stream",
                    headers={"xi-api-key": self._api_key, "accept": "audio/mpeg"},
                    json={"text": segment.text, "model_id": self._model_id},
                    params={"output_format": self._output_format},
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                if not response.content:
                    raise ValueError("provider returned empty audio")
                audio_path.write_bytes(response.content)
                duration = self._duration_probe(audio_path)
                synthesized.append(
                    SynthesizedSegment(
                        id=segment.id,
                        cue=segment.cue,
                        text=segment.text,
                        audio_path=audio_path,
                        duration_seconds=duration,
                        sha256=sha256(response.content).hexdigest(),
                    )
                )
            except Exception as error:
                raise ElevenLabsError(
                    f"ElevenLabs synthesis failed for segment '{segment.id}'"
                ) from error

        return SynthesizedNarration(
            lesson_id=plan.lesson_id,
            provider="elevenlabs",
            model_id=self._model_id,
            segments=tuple(synthesized),
            plan=plan,
        )
