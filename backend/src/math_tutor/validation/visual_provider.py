"""Call the separately configured Modal visual-model endpoint with retained frames.
The client pins model provenance and normalizes provider failures without leaking data."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx

from math_tutor.validation.visual import (
    FrameSample,
    VisualModelError,
    VisualModelResult,
    VisualModelTimedOut,
    VisualModelUnavailable,
)

QWEN3_VL_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_VL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

_SYSTEM_PROMPT = """You inspect sampled frames from a rendered math lesson.
Judge only visible cropping or truncation, elements outside the frame, an explicitly requested
visual element missing from all supplied frames, obvious rendering corruption,
caption-to-visible-content mismatch, or severe clutter that prevents reading individual elements.
Do not judge pedagogy, mathematical correctness, or general coherence.
Return only JSON matching the supplied schema. Use uncertain when the frames do not prove a claim.
"""

_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "findings"],
    "properties": {
        "status": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
        "findings": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rule", "frame_ids", "regions"],
                "properties": {
                    "rule": {
                        "type": "string",
                        "enum": [
                            "visible_cropping_or_truncation",
                            "element_outside_frame",
                            "missing_requested_visual_element",
                            "obvious_rendering_corruption",
                            "caption_visible_content_mismatch",
                            "severe_readability_clutter",
                        ],
                    },
                    "frame_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                    "regions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": [
                                "full_frame",
                                "top",
                                "bottom",
                                "left",
                                "right",
                                "center",
                                "top_left",
                                "top_right",
                                "bottom_left",
                                "bottom_right",
                            ],
                        },
                    },
                },
            },
        },
    },
}


class ModalVisualModelClient:
    """Inspect timestamped PNG frames through an OpenAI-compatible Modal endpoint."""

    model = QWEN3_VL_MODEL
    revision = QWEN3_VL_REVISION

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 45,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Configure the server-side visual-model connection.

        Args:
            base_url: OpenAI-compatible Modal endpoint base URL.
            api_key: Proxy credential retained in the backend process.
            timeout_seconds: Maximum duration for one visual inspection.
            transport: Optional HTTP transport used by focused tests.

        Raises:
            ValueError: If the URL is blank or timeout is not positive.
        """
        if not base_url:
            raise ValueError("Modal visual-model base URL is required")
        if timeout_seconds <= 0:
            raise ValueError("visual-model timeout must be positive")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
        )

    def inspect(
        self,
        *,
        prompt: str,
        frames: tuple[FrameSample, ...],
    ) -> VisualModelResult:
        """Request constrained findings for retained frames.

        Args:
            prompt: Bounded inspection context with the original request and captions.
            frames: Three to five retained PNG samples.

        Returns:
            Normalized model content and exact pinned provenance.

        Raises:
            ValueError: If the frame count is outside the supported range.
            VisualModelTimedOut: If the endpoint exceeds its deadline.
            VisualModelUnavailable: If the endpoint is unreachable or rejects the request.
            VisualModelError: If a successful response has an invalid provider envelope.
        """
        if not 3 <= len(frames) <= 5:
            raise ValueError("visual-model inspection requires three to five frames")
        frame_timestamps = "\n".join(
            f"{frame.id}: {frame.timestamp_seconds:.3f} seconds" for frame in frames
        )
        content: list[dict[str, object]] = [
            {"type": "text", "text": f"{prompt}\n\nFrame timestamps:\n{frame_timestamps}"}
        ]
        content.extend(self._image_content(frame.path) for frame in frames)
        try:
            response = self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1200,
                    "seed": 42,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "visual_evidence_report",
                            "strict": True,
                            "schema": _RESPONSE_SCHEMA,
                        },
                    },
                },
            )
        except httpx.TimeoutException as error:
            raise VisualModelTimedOut(
                "visual-model request exceeded its configured deadline"
            ) from error
        except httpx.HTTPError as error:
            raise VisualModelUnavailable("visual-model request could not be completed") from error
        if response.status_code != 200:
            raise VisualModelUnavailable(
                f"visual-model request failed with HTTP {response.status_code}"
            )
        try:
            payload: Any = response.json()
            content_value = payload["choices"][0]["message"]["content"]
            if not isinstance(content_value, str) or not content_value:
                raise TypeError
            request_id = str(payload.get("id") or "")
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise VisualModelError("visual-model endpoint returned a malformed response") from error
        return VisualModelResult(
            content=content_value,
            model=self.model,
            revision=self.revision,
            request_id=request_id,
            provider_response=response.text,
        )

    @staticmethod
    def _image_content(path: Path) -> dict[str, object]:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()


class UnavailableVisualModelClient:
    """Represent an intentionally unconfigured visual-model endpoint."""

    model = QWEN3_VL_MODEL
    revision = QWEN3_VL_REVISION

    def inspect(
        self,
        *,
        prompt: str,
        frames: tuple[FrameSample, ...],
    ) -> VisualModelResult:
        """Raise a bounded unavailable result without inspecting frame contents.

        Args:
            prompt: Unused inspection prompt.
            frames: Unused retained frame evidence.

        Raises:
            VisualModelUnavailable: Always, because no endpoint is configured.
        """
        del prompt, frames
        raise VisualModelUnavailable("visual-model endpoint is not configured")
