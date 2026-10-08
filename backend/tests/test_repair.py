"""Verify bounded repair packets built from validator findings.
Tests ensure raw messages and invalid evidence cannot enter regeneration prompts."""

from __future__ import annotations

import pytest

from math_tutor.repair import build_repair_prompt
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
