"""Call the configured Modal generation endpoint and normalize its responses.
Generation contracts and provider failures remain independent of lesson orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

import httpx

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BASE_MODEL_REVISION = "1b4199c4f36b0cef378bfb12390c18780c18af4c"
SHARED_ADAPTER_MODEL = "shared-lora-qwen3-4b-manim-v1"
SYSTEM_PROMPT = (
    "You generate concise, runnable Manim Community Edition Python scenes for math tutoring. "
    "Return only Python code."
)
VOICEOVER_SYSTEM_PROMPT = """You generate one self-contained narrated Manim Community Python scene.
Return exactly one Python code fence and no prose.
Start the code with exactly these five lines:
from manim import *
import math
import numpy as np
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.elevenlabs import ElevenLabsService
Define exactly one renderable class named GeneratedLesson that inherits from VoiceoverScene.
At the beginning of construct, call exactly once:
self.set_speech_service(
    ElevenLabsService(
        voice_id="__VOICE_ID__",
        model="eleven_multilingual_v2",
        transcription_model=None,
    )
)
Create 3 to 6 short narration blocks using `with self.voiceover(text="...") as tracker:`.
Place those blocks directly and sequentially in construct; do not wrap them in loops or conditions.
Each block must contain exactly one timed call: either `self.play(..., run_time=tracker.duration)`
or `self.wait(tracker.duration)`. Never call `self.play` or `self.wait` outside a narration block.
Keep narration concise and explain the mathematics being shown.
For directions, use only UP, DOWN, LEFT, RIGHT, UL, UR, DL, or DR; never use LR.
Never pass a Mobject method to self.play; use object.animate.method(arguments) instead.
Do not access files, subprocesses, environment variables, dynamic execution, or any network API
except the configured ElevenLabsService.
Keep the lesson concise and use only APIs available in Manim Community v0.19.
"""


class ProviderError(RuntimeError):
    """A credential-safe model provider failure."""


@dataclass(frozen=True)
class GenerationConfig:
    """Deterministic decoding configuration for one model client."""

    model: str = SHARED_ADAPTER_MODEL
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 4096
    seed: int = 42
    system_prompt: str = SYSTEM_PROMPT


@dataclass(frozen=True)
class TokenUsage:
    """Token counts returned by the inference endpoint."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class GenerationResult:
    """Normalized model response and request provenance."""

    content: str
    model: str
    request_id: str
    finish_reason: str | None
    usage: TokenUsage
    elapsed_seconds: float
    provider_response: str


@dataclass(frozen=True)
class ModelHealth:
    """Reachability and configured-model availability reported by a provider."""

    reachable: bool
    model: str
    model_available: bool
    error: str | None = None


class ModalVllmClient:
    """Minimal client for the OpenAI-compatible vLLM server deployed on Modal."""

    def __init__(
        self,
        *,
        api_key: str,
        config: GenerationConfig,
        base_url: str,
        timeout_seconds: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("Modal vLLM base URL is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.config = config
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
            follow_redirects=True,
        )

    def generate(self, prompt: str) -> GenerationResult:
        """Generate one response through the configured Modal endpoint.

        Args:
            prompt: User or repair prompt sent as the chat user message.

        Returns:
            Normalized response, usage, latency, and provider evidence.

        Raises:
            ProviderError: If the request fails or returns a malformed response.
        """
        started = monotonic()
        try:
            response = self._client.post(
                "/chat/completions",
                json={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": self.config.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_tokens,
                    "seed": self.config.seed,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        except httpx.HTTPError as error:
            raise ProviderError("Modal vLLM generation request could not be completed") from error
        elapsed = monotonic() - started
        if response.status_code != 200:
            raise ProviderError(
                f"Modal vLLM generation request failed with HTTP {response.status_code}"
            )
        try:
            payload: Any = response.json()
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            usage = payload.get("usage", {})
            if not isinstance(content, str) or not content:
                raise TypeError
            return GenerationResult(
                content=content,
                model=str(payload.get("model") or self.config.model),
                request_id=str(payload.get("id") or ""),
                finish_reason=(
                    str(choice["finish_reason"])
                    if choice.get("finish_reason") is not None
                    else None
                ),
                usage=TokenUsage(
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    total_tokens=int(usage.get("total_tokens", 0)),
                ),
                elapsed_seconds=elapsed,
                provider_response=response.text,
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ProviderError("Modal vLLM generation returned a malformed response") from error

    def health(self) -> ModelHealth:
        """Check endpoint reachability and configured-model availability."""
        try:
            response = self._client.get("/models")
        except httpx.HTTPError:
            return ModelHealth(
                reachable=False,
                model=self.config.model,
                model_available=False,
                error="Modal vLLM model catalog request could not be completed",
            )
        if response.status_code != 200:
            return ModelHealth(
                reachable=False,
                model=self.config.model,
                model_available=False,
                error=f"Modal vLLM model catalog returned HTTP {response.status_code}",
            )
        try:
            payload: Any = response.json()
            model_ids = {
                str(item["id"])
                for item in payload["data"]
                if isinstance(item, dict) and "id" in item
            }
        except (KeyError, TypeError, ValueError):
            return ModelHealth(
                reachable=True,
                model=self.config.model,
                model_available=False,
                error="Modal vLLM model catalog returned a malformed response",
            )
        return ModelHealth(
            reachable=True,
            model=self.config.model,
            model_available=self.config.model in model_ids,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


class UnavailableModelClient:
    """Expose a configured model boundary when no endpoint is available."""

    def __init__(self, config: GenerationConfig, reason: str) -> None:
        self.config = config
        self._reason = reason

    def generate(self, prompt: str) -> GenerationResult:
        """Raise the configuration failure recorded for this client."""
        raise ProviderError(self._reason)

    def health(self) -> ModelHealth:
        """Return the stored unavailability reason."""
        return ModelHealth(
            reachable=False,
            model=self.config.model,
            model_available=False,
            error=self._reason,
        )

    def close(self) -> None:
        """Release no resources because this client owns none."""
        return None
