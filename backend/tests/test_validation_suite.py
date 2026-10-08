"""Verify concurrent execution and aggregation of attempt validators.
The suite preserves axis reports while exposing one repair-compatible result."""

from __future__ import annotations

from pathlib import Path
from threading import Barrier

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


class _BarrierValidator(_Validator):
    def __init__(self, name: str, check: str, barrier: Barrier) -> None:
        super().__init__(
            name,
            ValidationReport(validator=name, status=ValidationStatus.PASS),
            check,
        )
        self._barrier = barrier

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        self._barrier.wait(timeout=1)
        return super().validate(attempt)


class _RaisingValidator(_Validator):
    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        raise RuntimeError("provider response must not escape validation")


def test_suite_starts_all_validators_concurrently(tmp_path: Path) -> None:
    barrier = Barrier(3)
    validators = tuple(
        _BarrierValidator(name, f"{name}_check", barrier)
        for name in ("media", "spatial", "visual_evidence")
    )

    report = ValidatorSuite(validators).validate(_attempt(tmp_path))

    assert report.status is ValidationStatus.PASS
    assert [component.validator for component in report.component_reports] == [
        "media",
        "spatial",
        "visual_evidence",
    ]


def test_suite_isolates_validator_exceptions_and_collects_every_axis(tmp_path: Path) -> None:
    passed = ValidationReport(validator="media", status=ValidationStatus.PASS)
    validators = (
        _Validator("media", passed, "media_check"),
        _RaisingValidator("spatial", passed, "spatial_check"),
        _Validator("visual_evidence", passed, "visual_check"),
    )

    report = ValidatorSuite(validators).validate(_attempt(tmp_path))

    assert report.status is ValidationStatus.VALIDATOR_ERROR
    assert [component.status for component in report.component_reports] == [
        ValidationStatus.PASS,
        ValidationStatus.VALIDATOR_ERROR,
        ValidationStatus.PASS,
    ]
    assert report.component_reports[1].findings[0].code == "validator_exception"
    assert "provider response" not in report.component_reports[1].findings[0].message


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
    assert report.axis_summaries() == [
        {
            "validator": "media",
            "status": "pass",
            "finding_count": 0,
            "advisory_count": 0,
            "provenance": {},
        },
        {
            "validator": "spatial",
            "status": "fail",
            "finding_count": 1,
            "advisory_count": 0,
            "provenance": {},
        },
    ]


def test_suite_validator_error_takes_precedence_over_failure(tmp_path: Path) -> None:
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
            status=ValidationStatus.VALIDATOR_ERROR,
            findings=(ValidationFinding(code="trace_error", message="No trace."),),
        ),
        "trace",
    )

    report = ValidatorSuite((failed, errored)).validate(_attempt(tmp_path))

    assert report.status is ValidationStatus.VALIDATOR_ERROR
    assert len(report.findings) == 2


def test_suite_aggregates_visual_uncertainty_and_validator_errors(tmp_path: Path) -> None:
    passed = _Validator(
        "media",
        ValidationReport(validator="media", status=ValidationStatus.PASS),
        "video_decodable",
    )

    for status in (ValidationStatus.UNCERTAIN, ValidationStatus.VALIDATOR_ERROR):
        visual_finding = ValidationFinding(
            code=f"visual_{status.value}",
            message=f"Visual validation returned {status.value}.",
        )
        visual = _Validator(
            "visual",
            ValidationReport(
                validator="visual",
                status=status,
                findings=(visual_finding,),
            ),
            f"visual_{status.value}",
        )

        report = ValidatorSuite((passed, visual)).validate(_attempt(tmp_path))

        assert report.status is status
        assert report.findings == (visual_finding,)
        assert report.component_reports == (passed._report, visual._report)


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
