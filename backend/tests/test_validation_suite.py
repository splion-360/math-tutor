"""Verify concurrent execution and aggregation of attempt validators.
The suite preserves axis reports while exposing one repair-compatible result."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Barrier
from time import monotonic, sleep

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


class _DelayedValidator(_Validator):
    def __init__(
        self,
        name: str,
        report: ValidationReport,
        check: str,
        delay_seconds: float = 0.15,
    ) -> None:
        super().__init__(name, report, check)
        self._delay_seconds = delay_seconds

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        sleep(self._delay_seconds)
        return super().validate(attempt)


def test_suite_starts_all_validators_concurrently(tmp_path: Path) -> None:
    barrier = Barrier(3)
    validators = tuple(
        _BarrierValidator(name, f"{name}_check", barrier)
        for name in ("media", "spatial", "visual_evidence")
    )

    suite = ValidatorSuite(validators)
    report = suite.validate(_attempt(tmp_path, suite.expected_checks))

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

    suite = ValidatorSuite(validators)
    report = suite.validate(_attempt(tmp_path, suite.expected_checks))

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

    suite = ValidatorSuite((media, spatial))
    report = suite.validate(_attempt(tmp_path, suite.expected_checks))

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

    suite = ValidatorSuite((failed, errored))
    report = suite.validate(_attempt(tmp_path, suite.expected_checks))

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

        suite = ValidatorSuite((passed, visual))
        report = suite.validate(_attempt(tmp_path, suite.expected_checks))

        assert report.status is status
        assert report.findings == (visual_finding,)
        assert report.component_reports == (passed._report, visual._report)


def test_suite_uncertainty_prevents_repairable_failure_from_becoming_aggregate_status(
    tmp_path: Path,
) -> None:
    repairable_failure = _Validator(
        "spatial",
        ValidationReport(
            validator="spatial",
            status=ValidationStatus.FAIL,
            findings=(
                ValidationFinding(
                    code="object_off_frame",
                    message="Object is outside the frame.",
                    repair_instruction="Move the object inside the frame.",
                ),
            ),
        ),
        "object_bounds",
    )
    uncertain = _Validator(
        "visual_evidence",
        ValidationReport(
            validator="visual_evidence",
            status=ValidationStatus.UNCERTAIN,
            findings=(
                ValidationFinding(
                    code="visual_evidence_uncertain",
                    message="Sampled frames are inconclusive.",
                ),
            ),
        ),
        "sampled_frames",
    )

    suite = ValidatorSuite((repairable_failure, uncertain))
    report = suite.validate(_attempt(tmp_path, suite.expected_checks))

    assert report.status is ValidationStatus.UNCERTAIN
    assert len(report.repairable_findings) == 1


def test_suite_finishes_three_delayed_axes_in_parallel(tmp_path: Path) -> None:
    validators = tuple(
        _DelayedValidator(
            name,
            ValidationReport(validator=name, status=ValidationStatus.PASS),
            f"{name}_check",
        )
        for name in ("media", "spatial", "visual_evidence")
    )
    suite = ValidatorSuite(validators)

    started = monotonic()
    suite.validate(_attempt(tmp_path, suite.expected_checks))
    elapsed = monotonic() - started

    assert elapsed < 0.35


def test_suite_reports_each_axis_as_it_finishes_without_reordering_final_report(
    tmp_path: Path,
) -> None:
    completed: list[str] = []
    validators = (
        _DelayedValidator(
            "slow",
            ValidationReport(validator="slow", status=ValidationStatus.PASS),
            "slow_check",
            delay_seconds=0.12,
        ),
        _DelayedValidator(
            "fast",
            ValidationReport(validator="fast", status=ValidationStatus.PASS),
            "fast_check",
            delay_seconds=0.01,
        ),
    )
    suite = ValidatorSuite(
        validators,
        report_callback=lambda _attempt, report: completed.append(report.validator),
    )

    report = suite.validate(_attempt(tmp_path, suite.expected_checks))

    assert completed == ["fast", "slow"]
    assert [component.validator for component in report.component_reports] == ["slow", "fast"]


def test_suite_rejects_a_manifest_with_different_required_checks(tmp_path: Path) -> None:
    validators = (
        _RaisingValidator(
            "media",
            ValidationReport(validator="media", status=ValidationStatus.PASS),
            "video_decodable",
        ),
    )
    suite = ValidatorSuite(validators)

    report = suite.validate(_attempt(tmp_path, ("different_check",)))

    assert report.status is ValidationStatus.VALIDATOR_ERROR
    assert report.component_reports[0].findings[0].code == ("validation_input_contract_mismatch")


def test_suite_rejects_duplicate_checks() -> None:
    report = ValidationReport(validator="one", status=ValidationStatus.PASS)

    try:
        ValidatorSuite((_Validator("one", report, "same"), _Validator("two", report, "same")))
    except ValueError as error:
        assert "duplicate expected check" in str(error)
    else:
        raise AssertionError("duplicate checks must be rejected")


def _attempt(tmp_path: Path, expected_checks: tuple[str, ...]) -> RenderedAttempt:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    validation_input = tmp_path / "validation_input.json"
    validation_input.write_text(
        json.dumps(
            {
                "schema_version": "validation-input.v1",
                "expected_checks": list(expected_checks),
            }
        ),
        encoding="utf-8",
    )
    return RenderedAttempt(
        number=0,
        artifact_dir=tmp_path,
        prompt="Explain a theorem",
        source="class Lesson: pass",
        scene_class="Lesson",
        outcome=RenderOutcome(video, "test", 1, "rendered"),
        narration_required=False,
        captions_required=False,
        validation_input_path=validation_input,
    )
