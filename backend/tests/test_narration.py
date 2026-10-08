"""Verify narration value contracts and immutable normalized metadata.
The tests cover plan ordering, identifiers, durations, and digests."""

from __future__ import annotations

from pathlib import Path

import pytest

from math_tutor.rendering.narration import (
    NarrationPlan,
    NarrationSegment,
    NarrationStatus,
    SynthesizedNarration,
    SynthesizedSegment,
)


def test_narration_plan_preserves_authoritative_segment_order() -> None:
    plan = NarrationPlan(
        lesson_id="pythagorean-theorem",
        segments=(
            NarrationSegment("introduce", "Consider a right triangle.", "triangle-visible"),
            NarrationSegment("equation", "Its sides satisfy a² + b² = c².", "equation-visible"),
        ),
    )

    assert plan.schema_version == "narration-plan.v1"
    assert [segment.id for segment in plan.segments] == ["introduce", "equation"]


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ((), "at least one segment"),
        (
            (
                NarrationSegment("same", "First.", "first-visible"),
                NarrationSegment("same", "Second.", "second-visible"),
            ),
            "unique",
        ),
    ],
)
def test_narration_plan_rejects_invalid_segment_collections(
    segments: tuple[NarrationSegment, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        NarrationPlan(lesson_id="lesson", segments=segments)


@pytest.mark.parametrize("field", ["id", "text", "cue"])
def test_narration_segment_rejects_blank_required_values(field: str) -> None:
    values = {"id": "segment", "text": "Explain it.", "cue": "visual-visible"}
    values[field] = "  "

    with pytest.raises(ValueError, match=field):
        NarrationSegment(**values)


def test_synthesized_narration_must_match_plan_order() -> None:
    plan = NarrationPlan(
        lesson_id="lesson",
        segments=(NarrationSegment("one", "One.", "one-visible"),),
    )
    wrong = SynthesizedSegment(
        id="two",
        cue="two-visible",
        text="Two.",
        audio_path=Path("audio/two.mp3"),
        duration_seconds=1.5,
        sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="plan order"):
        SynthesizedNarration(
            lesson_id="lesson",
            provider="fake",
            model_id="fake-model",
            segments=(wrong,),
            plan=plan,
        )


def test_narration_status_values_are_stable_api_contracts() -> None:
    assert [status.value for status in NarrationStatus] == [
        "not_requested",
        "pending",
        "ready",
        "unavailable",
    ]
