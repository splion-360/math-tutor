"""Validate observable Manim geometry from immutable render-time traces.
The rules use explicit frame, size, margin, and intersection thresholds."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from math_tutor.validation.models import (
    RenderedAttempt,
    ValidationFinding,
    ValidationReport,
    ValidationStatus,
)

_MAX_MEASUREMENT_ERRORS = 20
_TEXT_OBJECT_TYPES = {
    "Code",
    "DecimalNumber",
    "Integer",
    "MarkupText",
    "Matrix",
    "MathTex",
    "Paragraph",
    "Tex",
    "Text",
    "Title",
    "Variable",
}


class SpatialTraceError(RuntimeError):
    """Raised when a geometry trace cannot be parsed reliably."""


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned object bounds in Manim scene coordinates."""

    left: float
    bottom: float
    right: float
    top: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("object bounds must be finite")
        if self.right < self.left or self.top < self.bottom:
            raise ValueError("object bounds must be ordered")

    @property
    def values(self) -> tuple[float, float, float, float]:
        """Return bounds in left, bottom, right, top order."""
        return (self.left, self.bottom, self.right, self.top)

    @property
    def width(self) -> float:
        """Return the horizontal extent."""
        return self.right - self.left

    @property
    def height(self) -> float:
        """Return the vertical extent."""
        return self.top - self.bottom

    @property
    def area(self) -> float:
        """Return the axis-aligned bounding-box area."""
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        """Return JSON-safe named bounds."""
        return {
            "left": self.left,
            "bottom": self.bottom,
            "right": self.right,
            "top": self.top,
        }


@dataclass(frozen=True)
class TracedObject:
    """One top-level scene object observed at a checkpoint."""

    id: str
    type: str
    bounds: Bounds
    is_container: bool = False


@dataclass(frozen=True)
class SpatialCheckpoint:
    """Visible-object geometry after one stable scene operation."""

    index: int
    kind: str
    time_seconds: float
    objects: tuple[TracedObject, ...]
    measurement_errors: tuple[Mapping[str, str], ...] = ()


@dataclass(frozen=True)
class SpatialTrace:
    """Frame dimensions and ordered stable geometry checkpoints."""

    frame_width: float
    frame_height: float
    checkpoints: tuple[SpatialCheckpoint, ...]


class SpatialTraceLoader(Protocol):
    """Load one immutable render-time geometry trace."""

    def load(self, path: Path) -> SpatialTrace:
        """Return a validated spatial trace."""
        ...


class JsonSpatialTraceLoader:
    """Parse the versioned JSON trace emitted by the Manim recorder."""

    def load(self, path: Path) -> SpatialTrace:
        """Load and validate one trace file.

        Args:
            path: Spatial trace generated beside rendered media.

        Returns:
            Normalized frame and checkpoint geometry.

        Raises:
            SpatialTraceError: If the artifact is missing or malformed.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("trace root must be an object")
            if payload.get("schema_version") != "manim-spatial-trace.v1":
                raise ValueError("unsupported spatial trace schema")
            frame = _mapping(payload.get("frame"))
            width = _positive_number(frame.get("width"))
            height = _positive_number(frame.get("height"))
            raw_checkpoints = payload.get("checkpoints")
            if not isinstance(raw_checkpoints, list) or not raw_checkpoints:
                raise ValueError("trace must contain checkpoints")
            checkpoints = tuple(_checkpoint(item) for item in raw_checkpoints)
            indices = [checkpoint.index for checkpoint in checkpoints]
            if indices != sorted(set(indices)):
                raise ValueError("checkpoint indices must be unique and ordered")
            times = [checkpoint.time_seconds for checkpoint in checkpoints]
            if times != sorted(times):
                raise ValueError("checkpoint times must be ordered")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise SpatialTraceError("spatial trace is unavailable or invalid") from error
        return SpatialTrace(width, height, checkpoints)


@dataclass(frozen=True)
class SpatialValidationPolicy:
    """Explicit thresholds for deterministic spatial checks.

    Args:
        unsafe_margin: Minimum distance from every frame edge in scene units.
        max_width_ratio: Maximum object width as a fraction of frame width.
        max_height_ratio: Maximum object height as a fraction of frame height.
        severe_overlap_ratio: Intersection area divided by the smaller box area.
        persistent_checkpoints: Consecutive checkpoints required for overlap failure.
    """

    unsafe_margin: float = 0.25
    max_width_ratio: float = 0.9
    max_height_ratio: float = 0.9
    severe_overlap_ratio: float = 0.35
    persistent_checkpoints: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.unsafe_margin) or self.unsafe_margin < 0:
            raise ValueError("unsafe margin must be finite and non-negative")
        if not 0 < self.max_width_ratio <= 1:
            raise ValueError("maximum width ratio must be in (0, 1]")
        if not 0 < self.max_height_ratio <= 1:
            raise ValueError("maximum height ratio must be in (0, 1]")
        if not 0 < self.severe_overlap_ratio <= 1:
            raise ValueError("severe overlap ratio must be in (0, 1]")
        if self.persistent_checkpoints <= 0:
            raise ValueError("persistent checkpoints must be positive")


@dataclass(frozen=True)
class _ObjectViolation:
    checkpoint_index: int
    object: TracedObject


@dataclass(frozen=True)
class _Intersection:
    checkpoint_index: int
    first: TracedObject
    second: TracedObject
    ratio: float
    time_ordinal: int


class SpatialValidator:
    """Evaluate frame placement and persistent intersections from geometry traces."""

    name = "spatial"
    expected_checks: tuple[str, ...] = (
        "objects_inside_frame",
        "safe_frame_margins",
        "object_size_within_frame",
        "persistent_severe_intersections",
    )

    def __init__(
        self,
        *,
        loader: SpatialTraceLoader | None = None,
        policy: SpatialValidationPolicy | None = None,
    ) -> None:
        """Configure trace loading and deterministic spatial thresholds."""
        self._loader = loader or JsonSpatialTraceLoader()
        self._policy = policy or SpatialValidationPolicy()

    def validate(self, attempt: RenderedAttempt) -> ValidationReport:
        """Validate the geometry trace retained for one rendered attempt."""
        path = attempt.outcome.spatial_trace_path
        if path is None:
            return self._trace_error_report()
        try:
            trace = self._loader.load(path)
        except SpatialTraceError:
            return self._trace_error_report()

        measurement_errors = tuple(
            error for checkpoint in trace.checkpoints for error in checkpoint.measurement_errors
        )
        if measurement_errors:
            return ValidationReport(
                validator=self.name,
                status=ValidationStatus.VALIDATOR_ERROR,
                findings=(
                    ValidationFinding(
                        code="spatial_measurement_incomplete",
                        message="Some visible object bounds could not be measured.",
                        evidence={
                            "error_count": len(measurement_errors),
                            "errors": [
                                dict(error)
                                for error in measurement_errors[:_MAX_MEASUREMENT_ERRORS]
                            ],
                        },
                    ),
                ),
            )

        off_frame: dict[str, list[_ObjectViolation]] = defaultdict(list)
        unsafe_margin: dict[str, list[_ObjectViolation]] = defaultdict(list)
        oversized: dict[str, list[_ObjectViolation]] = defaultdict(list)
        intersections: dict[tuple[str, str], list[_Intersection]] = defaultdict(list)
        frame_left = -trace.frame_width / 2
        frame_right = trace.frame_width / 2
        frame_bottom = -trace.frame_height / 2
        frame_top = trace.frame_height / 2
        time_ordinals = _time_ordinals(trace.checkpoints)

        for checkpoint in trace.checkpoints:
            for traced in checkpoint.objects:
                bounds = traced.bounds
                outside = (
                    bounds.left < frame_left
                    or bounds.right > frame_right
                    or bounds.bottom < frame_bottom
                    or bounds.top > frame_top
                )
                violation = _ObjectViolation(checkpoint.index, traced)
                if outside:
                    off_frame[traced.id].append(violation)
                elif (
                    min(
                        bounds.left - frame_left,
                        frame_right - bounds.right,
                        bounds.bottom - frame_bottom,
                        frame_top - bounds.top,
                    )
                    < self._policy.unsafe_margin
                ):
                    unsafe_margin[traced.id].append(violation)
                if (
                    bounds.width / trace.frame_width > self._policy.max_width_ratio
                    or bounds.height / trace.frame_height > self._policy.max_height_ratio
                ):
                    oversized[traced.id].append(violation)
            for index, first in enumerate(checkpoint.objects):
                for second in checkpoint.objects[index + 1 :]:
                    ratio = _overlap_ratio(first.bounds, second.bounds)
                    if ratio >= self._policy.severe_overlap_ratio:
                        first_id, second_id = sorted((first.id, second.id))
                        pair = (first_id, second_id)
                        intersections[pair].append(
                            _Intersection(
                                checkpoint.index,
                                first,
                                second,
                                ratio,
                                time_ordinals[checkpoint.index],
                            )
                        )

        intersection_findings, intersection_advisories = self._intersection_results(intersections)
        findings = [
            *self._object_findings(
                "object_off_frame",
                off_frame,
                trace,
                "Move and scale this object so its full bounds remain inside the frame.",
            ),
            *self._object_findings(
                "unsafe_frame_margin",
                unsafe_margin,
                trace,
                (
                    "Move this object away from the frame edge so it meets the configured "
                    "safe margin."
                ),
            ),
            *self._object_findings(
                "oversized_object",
                oversized,
                trace,
                "Scale this object to fit within the configured frame-size ratios.",
            ),
            *intersection_findings,
        ]
        return ValidationReport(
            validator=self.name,
            status=ValidationStatus.FAIL if findings else ValidationStatus.PASS,
            findings=tuple(findings),
            advisories=tuple(intersection_advisories),
        )

    def _object_findings(
        self,
        code: str,
        grouped: Mapping[str, list[_ObjectViolation]],
        trace: SpatialTrace,
        instruction: str,
    ) -> list[ValidationFinding]:
        messages = {
            "object_off_frame": "An object extends outside the camera frame.",
            "unsafe_frame_margin": "An object is too close to the camera edge.",
            "oversized_object": "An object occupies too much of the camera frame.",
        }
        findings: list[ValidationFinding] = []
        for object_id in sorted(grouped):
            violations = grouped[object_id]
            first = violations[0].object
            findings.append(
                ValidationFinding(
                    code=code,
                    message=messages[code],
                    evidence={
                        "object_id": object_id,
                        "object_type": first.type,
                        "checkpoint_indices": [
                            violation.checkpoint_index for violation in violations
                        ],
                        "observed_bounds": first.bounds.to_dict(),
                        "frame": {
                            "width": trace.frame_width,
                            "height": trace.frame_height,
                        },
                        "thresholds": self._policy_evidence(),
                    },
                    repair_instruction=instruction,
                )
            )
        return findings

    def _intersection_results(
        self,
        grouped: Mapping[tuple[str, str], list[_Intersection]],
    ) -> tuple[list[ValidationFinding], list[ValidationFinding]]:
        findings: list[ValidationFinding] = []
        advisories: list[ValidationFinding] = []
        for pair in sorted(grouped):
            for run in _consecutive_runs(grouped[pair]):
                if len(run) < self._policy.persistent_checkpoints:
                    continue
                first = run[0]
                types = {
                    first.first.id: first.first.type,
                    first.second.id: first.second.type,
                }
                containers = {
                    first.first.id: first.first.is_container,
                    first.second.id: first.second.is_container,
                }
                object_types = [types[object_id] for object_id in pair]
                evidence = {
                    "object_ids": list(pair),
                    "object_types": object_types,
                    "object_is_container": [containers[object_id] for object_id in pair],
                    "checkpoint_indices": [item.checkpoint_index for item in run],
                    "minimum_overlap_ratio": min(item.ratio for item in run),
                    "thresholds": self._policy_evidence(),
                }
                if all(_is_text_object(object_type) for object_type in object_types):
                    findings.append(
                        ValidationFinding(
                            code="persistent_severe_intersection",
                            message="Text overlaps another object across stable checkpoints.",
                            evidence=evidence,
                            repair_instruction=(
                                "Reposition or resize these objects so their bounding boxes "
                                "no longer overlap severely."
                            ),
                        )
                    )
                else:
                    advisories.append(
                        ValidationFinding(
                            code="persistent_geometric_intersection",
                            message=("Non-text object bounds intersect across stable checkpoints."),
                            evidence=evidence,
                        )
                    )
        return findings, advisories

    def _policy_evidence(self) -> dict[str, float | int]:
        return {
            "unsafe_margin": self._policy.unsafe_margin,
            "max_width_ratio": self._policy.max_width_ratio,
            "max_height_ratio": self._policy.max_height_ratio,
            "severe_overlap_ratio": self._policy.severe_overlap_ratio,
            "persistent_checkpoints": self._policy.persistent_checkpoints,
        }

    def _trace_error_report(self) -> ValidationReport:
        return ValidationReport(
            validator=self.name,
            status=ValidationStatus.VALIDATOR_ERROR,
            findings=(
                ValidationFinding(
                    code="spatial_trace_unavailable",
                    message="Render-time geometry could not be inspected.",
                ),
            ),
        )


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("expected an object")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("number must be finite")
    return result


def _positive_number(value: object) -> float:
    result = _number(value)
    if result <= 0:
        raise ValueError("number must be positive")
    return result


def _checkpoint(value: object) -> SpatialCheckpoint:
    payload = _mapping(value)
    index = payload.get("index")
    kind = payload.get("kind")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("checkpoint index must be a non-negative integer")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("checkpoint kind must not be blank")
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("checkpoint objects must be a list")
    objects = tuple(_traced_object(item) for item in raw_objects)
    object_ids = [item.id for item in objects]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("object identifiers must be unique within a checkpoint")
    return SpatialCheckpoint(
        index=index,
        kind=kind,
        time_seconds=_number(payload.get("time_seconds")),
        objects=objects,
        measurement_errors=_measurement_errors(payload.get("measurement_errors", [])),
    )


def _measurement_errors(value: object) -> tuple[Mapping[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError("measurement errors must be a list")
    errors: list[Mapping[str, str]] = []
    for item in value:
        payload = _mapping(item)
        normalized: dict[str, str] = {}
        for key in ("object_id", "object_type", "reason"):
            field = payload.get(key)
            if not isinstance(field, str) or not field.strip():
                raise ValueError("measurement error fields must not be blank")
            normalized[key] = field
        errors.append(normalized)
    return tuple(errors)


def _traced_object(value: object) -> TracedObject:
    payload = _mapping(value)
    object_id = payload.get("id")
    object_type = payload.get("type")
    is_container = payload.get("is_container", False)
    if not isinstance(object_id, str) or not object_id.strip():
        raise ValueError("object id must not be blank")
    if not isinstance(object_type, str) or not object_type.strip():
        raise ValueError("object type must not be blank")
    if not isinstance(is_container, bool):
        raise ValueError("object container flag must be boolean")
    bounds = _mapping(payload.get("bounds"))
    return TracedObject(
        id=object_id,
        type=object_type,
        is_container=is_container,
        bounds=Bounds(
            left=_number(bounds.get("left")),
            bottom=_number(bounds.get("bottom")),
            right=_number(bounds.get("right")),
            top=_number(bounds.get("top")),
        ),
    )


def _overlap_ratio(first: Bounds, second: Bounds) -> float:
    smaller_area = min(first.area, second.area)
    if smaller_area <= 0:
        return 0
    width = max(0.0, min(first.right, second.right) - max(first.left, second.left))
    height = max(0.0, min(first.top, second.top) - max(first.bottom, second.bottom))
    return width * height / smaller_area


def _consecutive_runs(values: list[_Intersection]) -> tuple[tuple[_Intersection, ...], ...]:
    by_time_ordinal: dict[int, _Intersection] = {}
    for item in sorted(values, key=lambda item: item.checkpoint_index):
        by_time_ordinal.setdefault(item.time_ordinal, item)
    ordered = list(by_time_ordinal.values())
    runs: list[list[_Intersection]] = []
    for item in ordered:
        if not runs or item.time_ordinal != runs[-1][-1].time_ordinal + 1:
            runs.append([item])
        else:
            runs[-1].append(item)
    return tuple(tuple(run) for run in runs)


def _time_ordinals(checkpoints: tuple[SpatialCheckpoint, ...]) -> dict[int, int]:
    ordinals: dict[int, int] = {}
    current_ordinal = -1
    previous_time: float | None = None
    for checkpoint in checkpoints:
        if previous_time is None or checkpoint.time_seconds > previous_time:
            current_ordinal += 1
        ordinals[checkpoint.index] = current_ordinal
        previous_time = checkpoint.time_seconds
    return ordinals


def _is_text_object(object_type: str) -> bool:
    return (
        object_type in _TEXT_OBJECT_TYPES
        or object_type.endswith("Tex")
        or object_type.endswith("Text")
    )
