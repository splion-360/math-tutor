"""Verify bounded repair packets built from validator findings.
Tests ensure raw messages and invalid evidence cannot enter regeneration prompts."""

from __future__ import annotations

import json

import pytest

from math_tutor.generation.repair import build_repair_prompt
from math_tutor.validation.models import (
    ValidationFinding,
    ValidationReport,
    ValidationStatus,
)


def test_repair_prompt_excludes_finding_message_and_preserves_requirements() -> None:
    report = ValidationReport(
        validator="media",
        status=ValidationStatus.FAIL,
        findings=(
            ValidationFinding(
                code="duration_out_of_range",
                message="raw prose must not be copied",
                evidence={"actual_seconds": 7},
                repair_instruction="Extend the lesson to at least 30 seconds.",
            ),
        ),
    )

    prompt = build_repair_prompt(
        original_prompt="Explain distance.",
        previous_output="scene source",
        report=report,
    )

    assert "duration_out_of_range" in prompt
    assert "raw prose must not be copied" not in prompt
    assert "original_mathematical_topic" in prompt


def test_repair_packet_rejects_non_serializable_evidence() -> None:
    report = ValidationReport(
        validator="media",
        status=ValidationStatus.FAIL,
        findings=(
            ValidationFinding(
                code="duration_out_of_range",
                message="Duration failed.",
                evidence={"invalid": object()},
                repair_instruction="Extend the lesson.",
            ),
        ),
    )

    with pytest.raises(ValueError, match="JSON serializable"):
        build_repair_prompt(
            original_prompt="Explain distance.",
            previous_output="scene source",
            report=report,
        )


def test_repair_prompt_bounds_combined_findings_and_large_evidence() -> None:
    findings = tuple(
        ValidationFinding(
            code=f"spatial_{index}",
            message="Spatial violation.",
            evidence={
                "object_id": f"object-{index}",
                "checkpoint_indices": list(range(1_000)),
            },
            repair_instruction="Move the object inside the frame.",
        )
        for index in range(21)
    )
    report = ValidationReport(
        validator="validation_suite",
        status=ValidationStatus.FAIL,
        findings=findings,
    )

    prompt = build_repair_prompt(
        original_prompt="Explain distance.",
        previous_output="scene source",
        report=report,
    )

    feedback_text = prompt.split("<structured_validation_feedback>\n", 1)[1].split(
        "\n</structured_validation_feedback>", 1
    )[0]
    feedback = json.loads(feedback_text)
    assert len(feedback["findings"]) == 20
    assert feedback["omitted_finding_count"] == 1
    assert feedback["findings"][0]["evidence"]["evidence_truncated"] is True
    assert feedback["findings"][0]["evidence"]["checkpoint_indices_total_count"] == 1_000
