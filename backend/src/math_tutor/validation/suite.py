"""Run independent attempt validators concurrently and combine their reports.
The suite retains per-axis evidence and exposes confirmed findings to repair logic."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from math_tutor.validation.models import (
    AttemptValidator,
    RenderedAttempt,
    ValidationFinding,
    ValidationReport,
    ValidationStatus,
)

_STATUS_PRECEDENCE = (
    ValidationStatus.VALIDATOR_ERROR,
    ValidationStatus.UNCERTAIN,
    ValidationStatus.FAIL,
)


class ValidatorSuite:
    """Run independent validators concurrently and aggregate in declaration order."""

    name = "validation_suite"

    def __init__(
        self,
        validators: Sequence[AttemptValidator],
        report_callback: Callable[[RenderedAttempt, ValidationReport], None] | None = None,
    ) -> None:
        """Configure a non-empty sequence of validators.

        Args:
            validators: Independent validators executed in the provided order.
            report_callback: Optional callback invoked as each validator finishes.

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
        self._report_callback = report_callback
        self.expected_checks = expected_checks

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        """Run each validator and return one repair-compatible aggregate report."""
        manifest_error = self._validation_input_error(attempt)
        if manifest_error is not None:
            reports = tuple(
                ValidationReport(
                    validator=validator.name,
                    status=ValidationStatus.VALIDATOR_ERROR,
                    findings=(manifest_error,),
                )
                for validator in self._validators
            )
            for report in reports:
                self._report(attempt, report)
            return self._aggregate(reports)
        with ThreadPoolExecutor(
            max_workers=len(self._validators),
            thread_name_prefix="lesson-validator",
        ) as executor:
            futures = {
                executor.submit(validator.validate, attempt): (index, validator)
                for index, validator in enumerate(self._validators)
            }
            completed: dict[int, ValidationReport] = {}
            for future in as_completed(futures):
                index, validator = futures[future]
                report = self._resolve_report(validator, future)
                completed[index] = report
                self._report(attempt, report)
            reports = tuple(completed[index] for index in range(len(self._validators)))
        return self._aggregate(reports)

    def _report(self, attempt: RenderedAttempt, report: ValidationReport) -> None:
        """Publish one completed axis through the configured progress callback."""
        if self._report_callback is not None:
            self._report_callback(attempt, report)

    def _validation_input_error(
        self,
        attempt: RenderedAttempt,
    ) -> ValidationFinding | None:
        """Check the immutable manifest shared by every configured validator.

        Args:
            attempt: Rendered attempt carrying the validation-input manifest path.

        Returns:
            Controlled finding when the manifest is absent or does not declare the
            configured required checks; otherwise ``None``.
        """
        path = attempt.validation_input_path
        if path is None:
            return ValidationFinding(
                code="validation_input_missing",
                message="The shared validation input is missing.",
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ValidationFinding(
                code="validation_input_invalid",
                message="The shared validation input could not be read.",
            )
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != "validation-input.v1"
            or payload.get("expected_checks") != list(self.expected_checks)
        ):
            return ValidationFinding(
                code="validation_input_contract_mismatch",
                message="The shared validation input does not match the required checks.",
            )
        return None

    def _aggregate(self, reports: tuple[ValidationReport, ...]) -> ValidationReport:
        """Combine ordered axis reports using deterministic status precedence.

        Args:
            reports: Component reports in validator declaration order.

        Returns:
            Repair-compatible aggregate with flattened findings and advisories.
        """
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
        """Resolve one report while containing an unexpected validator exception.

        Args:
            validator: Validator associated with the submitted future.
            future: Concurrent validation execution to resolve.

        Returns:
            Validator report or a controlled validator-error report.
        """
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
