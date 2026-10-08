"""Run independent attempt validators concurrently and combine their reports.
The suite retains per-axis evidence and exposes confirmed findings to repair logic."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor

from math_tutor.validation.models import (
    AttemptValidator,
    RenderedAttempt,
    ValidationFinding,
    ValidationReport,
    ValidationStatus,
)

_STATUS_PRECEDENCE = (
    ValidationStatus.VALIDATOR_ERROR,
    ValidationStatus.FAIL,
    ValidationStatus.UNCERTAIN,
)


class ValidatorSuite:
    """Run independent validators concurrently and aggregate in declaration order."""

    name = "validation_suite"

    def __init__(self, validators: Sequence[AttemptValidator]) -> None:
        """Configure a non-empty sequence of validators.

        Args:
            validators: Independent validators executed in the provided order.

        Raises:
            ValueError: If no validators are provided or checks are duplicated.
        """
        if not validators:
            raise ValueError("validator suite must contain at least one validator")
        expected_checks = tuple(
            check for validator in validators for check in validator.expected_checks
        )
        if len(expected_checks) != len(set(expected_checks)):
            raise ValueError("validator suite contains a duplicate expected check")
        self._validators = tuple(validators)
        self.expected_checks = expected_checks

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        """Run each validator and return one repair-compatible aggregate report."""
        with ThreadPoolExecutor(
            max_workers=len(self._validators),
            thread_name_prefix="lesson-validator",
        ) as executor:
            futures = tuple(
                executor.submit(validator.validate, attempt) for validator in self._validators
            )
            reports = tuple(
                self._resolve_report(validator, future)
                for validator, future in zip(self._validators, futures, strict=True)
            )
        status = ValidationStatus.PASS
        for candidate in _STATUS_PRECEDENCE:
            if any(report.status is candidate for report in reports):
                status = candidate
                break
        return ValidationReport(
            validator=self.name,
            status=status,
            findings=tuple(finding for report in reports for finding in report.findings),
            advisories=tuple(advisory for report in reports for advisory in report.advisories),
            component_reports=reports,
        )

    @staticmethod
    def _resolve_report(
        validator: AttemptValidator,
        future: Future[ValidationReport],
    ) -> ValidationReport:
        """Return one report while containing an unexpected validator exception."""
        try:
            return future.result()
        except Exception:
            return ValidationReport(
                validator=validator.name,
                status=ValidationStatus.VALIDATOR_ERROR,
                findings=(
                    ValidationFinding(
                        code="validator_exception",
                        message=f"{validator.name} validation could not be completed.",
                    ),
                ),
            )
