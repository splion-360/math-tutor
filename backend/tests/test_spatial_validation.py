"""Verify deterministic spatial validation of rendered Manim geometry traces.
Tests cover frame bounds, sizing, margins, and persistent intersections."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from math_tutor.jobs import RenderOutcome
from math_tutor.validation.models import RenderedAttempt, ValidationStatus
from math_tutor.validation.spatial import SpatialValidationPolicy, SpatialValidator


def test_spatial_validator_reports_cropped_offscreen_margin_and_oversized_objects(
    tmp_path: Path,
) -> None:
    trace = _write_trace(
        tmp_path,
        [
            _checkpoint(
                0,
                _object("equation", "MathTex", -5.4, -1, -3, 1),
                _object("diagram", "Circle", 5.2, -1, 6, 1),
                _object("label", "Text", -4.9, -0.5, -4, 0.5),
                _object("wide", "MathTex", -4.6, -0.4, 4.6, 0.4),
            )
        ],
    )
    report = SpatialValidator(policy=_policy(persistent_checkpoints=2)).validate(
        _attempt(tmp_path, trace)
    )

    assert report.status is ValidationStatus.FAIL
    assert {finding.code for finding in report.findings} == {
        "object_off_frame",
        "unsafe_frame_margin",
        "oversized_object",
    }
    off_frame_ids = {
        finding.evidence["object_id"]
        for finding in report.findings
        if finding.code == "object_off_frame"
    }
    assert off_frame_ids == {"equation", "diagram"}
    assert all(finding.repair_instruction for finding in report.findings)
    assert all("checkpoint_indices" in finding.evidence for finding in report.findings)


def test_spatial_validator_ignores_one_checkpoint_contact(tmp_path: Path) -> None:
    trace = _write_trace(
        tmp_path,
        [
            _checkpoint(
                0,
                _object("left", "Square", -1, -1, 1, 1),
                _object("right", "Square", 0.5, -1, 2.5, 1),
            ),
            _checkpoint(
                1,
                _object("left", "Square", -2.5, -1, -0.5, 1),
                _object("right", "Square", 0.5, -1, 2.5, 1),
            ),
        ],
    )

    report = SpatialValidator(policy=_policy(persistent_checkpoints=2)).validate(
        _attempt(tmp_path, trace)
    )

    assert report.status is ValidationStatus.PASS
    assert report.findings == ()


def test_spatial_validator_reports_persistent_severe_intersection(tmp_path: Path) -> None:
    trace = _write_trace(
        tmp_path,
        [
            _checkpoint(
                0,
                _object("equation", "MathTex", -1, -1, 1, 1),
                _object("label", "Text", -0.4, -1, 1.6, 1),
            ),
            _checkpoint(
                1,
                _object("equation", "MathTex", -1, -1, 1, 1),
                _object("label", "Text", -0.5, -1, 1.5, 1),
            ),
        ],
    )

    report = SpatialValidator(policy=_policy(persistent_checkpoints=2)).validate(
        _attempt(tmp_path, trace)
    )

    assert report.status is ValidationStatus.FAIL
    assert [finding.code for finding in report.findings] == ["persistent_severe_intersection"]
    evidence = report.findings[0].evidence
    assert evidence["object_ids"] == ["equation", "label"]
    assert evidence["checkpoint_indices"] == [0, 1]
    assert evidence["minimum_overlap_ratio"] == pytest.approx(0.7)


def test_spatial_validator_passes_valid_scene(tmp_path: Path) -> None:
    trace = _write_trace(
        tmp_path,
        [
            _checkpoint(
                0,
                _object("equation", "MathTex", -3, 1, 3, 2),
                _object("diagram", "Circle", -1, -2, 1, 0),
            ),
            _checkpoint(
                1,
                _object("equation", "MathTex", -3, 1, 3, 2),
                _object("diagram", "Circle", -1, -2, 1, 0),
            ),
        ],
    )

    report = SpatialValidator(policy=_policy(persistent_checkpoints=2)).validate(
        _attempt(tmp_path, trace)
    )

    assert report.status is ValidationStatus.PASS
    assert report.findings == ()


def test_axes_and_plotted_curve_intersection_is_advisory(tmp_path: Path) -> None:
    axes = _object("axes", "Axes", -3, -2, 3, 2)
    curve = _object("curve", "ParametricFunction", -3, -2, 3, 2)
    trace = _write_trace(
        tmp_path,
        [
            _checkpoint(0, axes, curve),
            _checkpoint(1, axes, curve),
        ],
    )

    report = SpatialValidator(policy=_policy(persistent_checkpoints=2)).validate(
        _attempt(tmp_path, trace)
    )

    assert report.status is ValidationStatus.PASS
    assert report.findings == ()
    assert [advisory.code for advisory in report.advisories] == [
        "persistent_geometric_intersection"
    ]


def test_spatial_validator_reports_missing_trace_as_validator_error(tmp_path: Path) -> None:
    report = SpatialValidator().validate(_attempt(tmp_path, None))

    assert report.status is ValidationStatus.ERROR
    assert report.findings[0].code == "spatial_trace_unavailable"
    assert report.findings[0].repair_instruction is None


def test_spatial_validator_reports_unmeasured_object_as_error(tmp_path: Path) -> None:
    checkpoint = _checkpoint(0)
    checkpoint["measurement_errors"] = [
        {
            "object_id": "object-0",
            "object_type": "CustomMobject",
            "reason": "bounds_unavailable",
        }
    ]
    trace = _write_trace(tmp_path, [checkpoint])

    report = SpatialValidator().validate(_attempt(tmp_path, trace))

    assert report.status is ValidationStatus.ERROR
    assert report.findings[0].code == "spatial_measurement_incomplete"


def test_same_time_final_checkpoint_does_not_make_overlap_persistent(
    tmp_path: Path,
) -> None:
    first = _object("equation", "MathTex", -1, -1, 1, 1)
    second = _object("label", "Text", -0.5, -1, 1.5, 1)
    first_checkpoint = _checkpoint(0, first, second)
    second_checkpoint = _checkpoint(1, first, second)
    second_checkpoint["time_seconds"] = first_checkpoint["time_seconds"]
    trace = _write_trace(tmp_path, [first_checkpoint, second_checkpoint])

    report = SpatialValidator(policy=_policy(persistent_checkpoints=2)).validate(
        _attempt(tmp_path, trace)
    )

    assert report.status is ValidationStatus.PASS


def test_duplicate_wait_checkpoints_do_not_break_longer_persistence(
    tmp_path: Path,
) -> None:
    equation = _object("equation", "MathTex", -1, -1, 1, 1)
    label = _object("label", "Text", -0.5, -1, 1.5, 1)
    checkpoints = []
    for index, time_seconds in enumerate((1, 1, 2, 2, 3, 3)):
        checkpoint = _checkpoint(index, equation, label)
        checkpoint["time_seconds"] = time_seconds
        checkpoints.append(checkpoint)
    trace = _write_trace(tmp_path, checkpoints)

    report = SpatialValidator(policy=_policy(persistent_checkpoints=3)).validate(
        _attempt(tmp_path, trace)
    )

    assert report.status is ValidationStatus.FAIL
    assert report.findings[0].evidence["checkpoint_indices"] == [0, 2, 4]


def test_spatial_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="unsafe margin"):
        SpatialValidationPolicy(unsafe_margin=-0.1)
    with pytest.raises(ValueError, match="width ratio"):
        SpatialValidationPolicy(max_width_ratio=1.1)
    with pytest.raises(ValueError, match="overlap ratio"):
        SpatialValidationPolicy(severe_overlap_ratio=0)
    with pytest.raises(ValueError, match="persistent checkpoints"):
        SpatialValidationPolicy(persistent_checkpoints=0)


def _policy(*, persistent_checkpoints: int) -> SpatialValidationPolicy:
    return SpatialValidationPolicy(
        unsafe_margin=0.25,
        max_width_ratio=0.9,
        max_height_ratio=0.9,
        severe_overlap_ratio=0.35,
        persistent_checkpoints=persistent_checkpoints,
    )


def _attempt(tmp_path: Path, trace_path: Path | None) -> RenderedAttempt:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"video")
    return RenderedAttempt(
        number=0,
        artifact_dir=tmp_path,
        prompt="Explain geometry.",
        source="from manim import *",
        scene_class="GeneratedLesson",
        outcome=RenderOutcome(
            video_path=video,
            renderer="test",
            elapsed_seconds=1,
            logs="rendered",
            spatial_trace_path=trace_path,
        ),
        narration_required=False,
        captions_required=False,
    )


def _write_trace(tmp_path: Path, checkpoints: list[dict[str, object]]) -> Path:
    path = tmp_path / "spatial_trace.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "manim-spatial-trace.v1",
                "frame": {"width": 10, "height": 6},
                "checkpoints": checkpoints,
            }
        ),
        encoding="utf-8",
    )
    return path


def _checkpoint(index: int, *objects: dict[str, object]) -> dict[str, object]:
    return {
        "index": index,
        "kind": "play" if index == 0 else "final",
        "time_seconds": float(index),
        "objects": list(objects),
    }


def _object(
    object_id: str,
    object_type: str,
    left: float,
    bottom: float,
    right: float,
    top: float,
) -> dict[str, object]:
    return {
        "id": object_id,
        "type": object_type,
        "bounds": {
            "left": left,
            "bottom": bottom,
            "right": right,
            "top": top,
        },
    }
