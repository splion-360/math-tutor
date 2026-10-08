"""Define structured findings shared by independent lesson validators.
These types form the boundary between validation and repair orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from math_tutor.attempts import RenderedAttempt


class ValidationStatus(StrEnum):
    """Final status reported by one validator."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationFinding:
    """One bounded validation result with optional repair guidance.

    Args:
        code: Stable machine-readable rule identifier.
        message: Controlled explanation suitable for logs and the API.
        evidence: Structured measurements that support the finding.
        repair_instruction: Bounded instruction for model repair, when applicable.
    """

    code: str
    message: str
    evidence: Mapping[str, object] = field(default_factory=dict)
    repair_instruction: str | None = None

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise ValueError("validation finding code and message must not be blank")
        if self.repair_instruction is not None and not self.repair_instruction.strip():
            raise ValueError("repair instruction must be omitted or non-blank")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the finding."""
        return {
            "code": self.code,
            "message": self.message,
            "evidence": dict(self.evidence),
            "repair_instruction": self.repair_instruction,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Result produced by one validator for one attempt.

    Args:
        validator: Stable validator name.
        status: Overall validation result.
        findings: Evidence-bearing findings produced by the validator.
    """

    validator: str
    status: ValidationStatus
    findings: tuple[ValidationFinding, ...] = ()

    def __post_init__(self) -> None:
        if not self.validator.strip():
            raise ValueError("validator name must not be blank")
        if self.status is ValidationStatus.PASS and self.findings:
            raise ValueError("passing validation reports must not contain findings")
        if self.status is not ValidationStatus.PASS and not self.findings:
            raise ValueError("failed validation reports must contain findings")

    @property
    def repairable_findings(self) -> tuple[ValidationFinding, ...]:
        """Return findings that can be sent to the generator for repair."""
        return tuple(
            finding for finding in self.findings if finding.repair_instruction is not None
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the report."""
        return {
            "validator": self.validator,
            "status": self.status.value,
            "findings": [finding.to_dict() for finding in self.findings],
        }


class AttemptValidator(Protocol):
    """Validate a rendered lesson attempt without mutating it."""

    expected_checks: tuple[str, ...]

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        """Evaluate one attempt and return structured evidence."""
        ...
