"""Verify sequential composition of independent attempt validators.
The suite preserves axis reports while exposing one repair-compatible result."""

from __future__ import annotations

from pathlib import Path

from math_tutor.jobs import RenderOutcome
from math_tutor.validation.models import (
    RenderedAttempt,
    ValidationFinding,
    ValidationReport,
    ValidationStatus,
)
from math_tutor.validation.suite import ValidatorSuite


class _Validator:
    def __init__(self, name: str, report: ValidationReport, check: str) -> None:
        self.name = name
        self.expected_checks = (check,)
        self._report = report

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        return self._report


def test_suite_retains_axis_reports_and_flattens_repairable_findings(
    tmp_path: Path,
) -> None:
    media = _Validator(
        "media",
        ValidationReport(validator="media", status=ValidationStatus.PASS),
        "video_decodable",
    )
    spatial_finding = ValidationFinding(
        code="object_off_frame",
        message="Object outside frame.",
        evidence={"object_id": "object-0"},
        repair_instruction="Move the object inside the frame.",
    )
    spatial = _Validator(
        "spatial",
        ValidationReport(
            validator="spatial",
            status=ValidationStatus.FAIL,
            findings=(spatial_finding,),
        ),
        "objects_inside_frame",
    )

    report = ValidatorSuite((media, spatial)).validate(_attempt(tmp_path))

    assert report.status is ValidationStatus.FAIL
    assert report.findings == (spatial_finding,)
    assert report.component_reports == (media._report, spatial._report)
    assert report.to_dict()["component_reports"][1]["validator"] == "spatial"
    assert report.repairable_findings == (spatial_finding,)


def test_suite_error_takes_precedence_over_failure(tmp_path: Path) -> None:
    failed = _Validator(
        "media",
        ValidationReport(
            validator="media",
            status=ValidationStatus.FAIL,
            findings=(ValidationFinding(code="duration", message="Bad duration."),),
        ),
        "duration",
    )
    errored = _Validator(
        "spatial",
        ValidationReport(
            validator="spatial",
            status=ValidationStatus.ERROR,
            findings=(ValidationFinding(code="trace_error", message="No trace."),),
        ),
        "trace",
    )

    report = ValidatorSuite((failed, errored)).validate(_attempt(tmp_path))

    assert report.status is ValidationStatus.ERROR
    assert len(report.findings) == 2


def test_suite_rejects_duplicate_checks() -> None:
    report = ValidationReport(validator="one", status=ValidationStatus.PASS)

    try:
        ValidatorSuite((_Validator("one", report, "same"), _Validator("two", report, "same")))
    except ValueError as error:
        assert "duplicate expected check" in str(error)
    else:
        raise AssertionError("duplicate checks must be rejected")


def _attempt(tmp_path: Path) -> RenderedAttempt:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    return RenderedAttempt(
        number=0,
        artifact_dir=tmp_path,
        prompt="Explain a theorem",
        source="class Lesson: pass",
        scene_class="Lesson",
        outcome=RenderOutcome(video, "test", 1, "rendered"),
        narration_required=False,
        captions_required=False,
    )
