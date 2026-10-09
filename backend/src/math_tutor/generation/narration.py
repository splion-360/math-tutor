"""Generate bounded narration plans from completed lesson prompts and source.
The shared LoRA remains dedicated to Manim code while the base model writes speech."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from math_tutor.domain import NarrationStatus
from math_tutor.generation.provider import GenerationConfig, GenerationResult, ProviderError
from math_tutor.jobs import JobExecutionError
from math_tutor.rendering.narration import (
    NarrationPlan,
    NarrationSegment,
    PlannedNarration,
)

NARRATION_SYSTEM_PROMPT = """Write concise spoken narration for an already-rendered math animation.
Return only one JSON object with this exact shape: {"segments":[{"text":"..."}]}.
Use two to four sequential segments. Explain the mathematical idea and the visible progression.
Use spoken math instead of raw LaTeX. Do not mention code, Manim, prompts, or these instructions.
Keep the total narration close to the requested word count so it fits the video duration.
"""
_JSON_FENCE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```$", re.IGNORECASE | re.DOTALL)


class NarrationGenerator(Protocol):
    """Generate one narration-model response."""

    config: GenerationConfig

    def generate(self, prompt: str) -> GenerationResult:
        """Return one provider-normalized generation result."""
        ...


class NarrationPlanningError(JobExecutionError):
    """Raised when narration generation cannot produce a bounded plan."""


class ModelNarrationPlanner:
    """Convert a base-model response into a validated narration plan."""

    def __init__(self, generator: NarrationGenerator) -> None:
        self._generator = generator

    def create_plan(
        self,
        *,
        lesson_id: str,
        prompt: str,
        source: str,
        target_duration_seconds: float,
    ) -> PlannedNarration:
        """Generate a narration plan for one completed Manim render.

        Args:
            lesson_id: Parent lesson job identifier.
            prompt: Original lesson request.
            source: Admitted Manim source used for the render.
            target_duration_seconds: Measured silent-video duration.

        Returns:
            Validated narration plan with provider evidence.

        Raises:
            NarrationPlanningError: If generation fails or returns invalid JSON.
        """
        request = json.dumps(
            {
                "lesson_prompt": prompt,
                "manim_source": source,
                "target_duration_seconds": round(target_duration_seconds, 3),
                "target_word_count": max(12, min(90, round(target_duration_seconds * 2))),
            },
            ensure_ascii=False,
        )
        try:
            result = self._generator.generate(request)
            texts = _parse_segments(result.content)
        except ProviderError as error:
            raise NarrationPlanningError(
                "Narration transcript generation failed",
                diagnostics={
                    "failure_stage": "narration",
                    "failure_kind": "operational",
                    "narration_status": NarrationStatus.UNAVAILABLE.value,
                },
            ) from error
        except (TypeError, ValueError) as error:
            raise NarrationPlanningError(
                "Narration transcript was not valid",
                diagnostics={
                    "failure_stage": "narration",
                    "failure_kind": "model_output",
                    "narration_status": NarrationStatus.UNAVAILABLE.value,
                },
            ) from error
        plan = NarrationPlan(
            lesson_id=lesson_id,
            segments=tuple(
                NarrationSegment(
                    id=f"segment-{index:02d}",
                    text=text,
                    cue=f"visual-sequence-{index:02d}",
                )
                for index, text in enumerate(texts, start=1)
            ),
        )
        return PlannedNarration(
            plan=plan,
            model=result.model,
            raw_response=result.content,
            provider_response=result.provider_response,
        )


def _parse_segments(content: str) -> tuple[str, ...]:
    fenced = _JSON_FENCE.fullmatch(content.strip())
    payload: Any = json.loads(fenced.group(1) if fenced is not None else content)
    if not isinstance(payload, dict) or set(payload) != {"segments"}:
        raise TypeError("narration response must contain only segments")
    segments = payload["segments"]
    if not isinstance(segments, list) or not 2 <= len(segments) <= 4:
        raise ValueError("narration response must contain two to four segments")
    texts: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {"text"}:
            raise TypeError("narration segment must contain only text")
        text = segment["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > 500:
            raise ValueError("narration segment text is invalid")
        texts.append(text.strip())
    return tuple(texts)
