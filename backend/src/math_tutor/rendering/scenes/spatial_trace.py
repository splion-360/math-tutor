"""Capture top-level scene-object bounds at stable Manim checkpoints.
The recorder has no Manim dependency and writes a versioned immutable trace."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol


class SceneGeometry(Protocol):
    """Minimum scene surface required by the spatial trace recorder."""

    mobjects: list[object]
    time: float


class SpatialTraceRecorder:
    """Collect stable object identifiers and axis-aligned scene bounds."""

    def __init__(self, output_path: Path, *, frame_width: float, frame_height: float) -> None:
        """Configure one trace artifact.

        Args:
            output_path: JSON path written after rendering completes.
            frame_width: Camera frame width in Manim scene units.
            frame_height: Camera frame height in Manim scene units.

        Raises:
            ValueError: If either frame dimension is not positive and finite.
        """
        if not math.isfinite(frame_width) or frame_width <= 0:
            raise ValueError("frame width must be positive and finite")
        if not math.isfinite(frame_height) or frame_height <= 0:
            raise ValueError("frame height must be positive and finite")
        self._output_path = output_path
        self._frame_width = float(frame_width)
        self._frame_height = float(frame_height)
        self._object_ids: dict[int, tuple[object, str]] = {}
        self._checkpoints: list[dict[str, object]] = []

    def record(self, scene: SceneGeometry, *, kind: str) -> None:
        """Record visible top-level scene objects after one stable operation.

        Args:
            scene: Scene exposing current top-level mobjects and elapsed time.
            kind: Stable operation label such as play, wait, or final.

        Raises:
            ValueError: If the checkpoint label or scene time is invalid.
        """
        if not kind.strip():
            raise ValueError("checkpoint kind must not be blank")
        time_seconds = float(scene.time)
        if not math.isfinite(time_seconds) or time_seconds < 0:
            raise ValueError("scene time must be finite and non-negative")
        objects: list[dict[str, object]] = []
        measurement_errors: list[dict[str, str]] = []
        for mobject in scene.mobjects:
            object_id = self._object_id(mobject)
            object_type = type(mobject).__name__
            if not _is_visible(mobject):
                continue
            try:
                left = _coordinate(mobject, "get_left", 0)
                bottom = _coordinate(mobject, "get_bottom", 1)
                right = _coordinate(mobject, "get_right", 0)
                top = _coordinate(mobject, "get_top", 1)
                if right < left or top < bottom:
                    raise ValueError("unordered bounds")
            except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
                measurement_errors.append(
                    {
                        "object_id": object_id,
                        "object_type": object_type,
                        "reason": "bounds_unavailable",
                    }
                )
                continue
            objects.append(
                {
                    "id": object_id,
                    "type": object_type,
                    "bounds": {
                        "left": left,
                        "bottom": bottom,
                        "right": right,
                        "top": top,
                    },
                }
            )
        self._checkpoints.append(
            {
                "index": len(self._checkpoints),
                "kind": kind,
                "time_seconds": time_seconds,
                "objects": objects,
                "measurement_errors": measurement_errors,
            }
        )

    def write(self) -> None:
        """Atomically write all captured checkpoints as versioned JSON."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._output_path.with_suffix(f"{self._output_path.suffix}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": "manim-spatial-trace.v1",
                    "frame": {
                        "width": self._frame_width,
                        "height": self._frame_height,
                    },
                    "checkpoints": self._checkpoints,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self._output_path)

    def _object_id(self, mobject: object) -> str:
        identity = id(mobject)
        if identity not in self._object_ids:
            object_id = f"object-{len(self._object_ids)}"
            self._object_ids[identity] = (mobject, object_id)
        return self._object_ids[identity][1]


def _coordinate(mobject: object, method_name: str, dimension: int) -> float:
    method = getattr(mobject, method_name)
    point = method()
    coordinate = float(point[dimension])
    if not math.isfinite(coordinate):
        raise ValueError("coordinate must be finite")
    return coordinate


def _is_visible(mobject: object) -> bool:
    family_members = getattr(mobject, "family_members_with_points", None)
    if not callable(family_members):
        return True
    try:
        members = family_members()
    except (AttributeError, TypeError, ValueError):
        return True
    return any(
        _has_positive_opacity(getattr(member, attribute, None))
        for member in members
        for attribute in ("opacity", "fill_opacity", "stroke_opacity")
    )


def _has_positive_opacity(value: object) -> bool:
    if value is None:
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        try:
            return any(float(item) > 0 for item in value)  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return False
