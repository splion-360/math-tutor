"""Compose independent attempt validators into one sequential validation result.
The suite retains per-axis reports and exposes flattened findings to repair logic."""

from __future__ import annotations

from collections.abc import Sequence

from math_tutor.validation.models import (
    AttemptValidator,
    RenderedAttempt,
    ValidationReport,
    ValidationStatus,
)


class ValidatorSuite:
    """Run independent validators in order and aggregate their reports."""

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
        reports = tuple(validator.validate(attempt) for validator in self._validators)
        status = ValidationStatus.PASS
        if any(report.status is ValidationStatus.ERROR for report in reports):
            status = ValidationStatus.ERROR
        elif any(report.status is ValidationStatus.FAIL for report in reports):
            status = ValidationStatus.FAIL
        return ValidationReport(
            validator=self.name,
            status=status,
            findings=tuple(finding for report in reports for finding in report.findings),
            advisories=tuple(advisory for report in reports for advisory in report.advisories),
            component_reports=reports,
        )
