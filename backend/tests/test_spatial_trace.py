"""Verify render-time capture of observable top-level Manim geometry.
Tests use scene-shaped fakes so trace collection stays independent of Manim."""

from __future__ import annotations

import json
from pathlib import Path

from math_tutor.rendering.scenes.spatial_trace import SpatialTraceRecorder


class FakeMobject:
    def __init__(
        self,
        *,
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> None:
        self._left = left
        self._bottom = bottom
        self._right = right
        self._top = top

    def get_left(self) -> tuple[float, float, float]:
        return (self._left, 0, 0)

    def get_right(self) -> tuple[float, float, float]:
        return (self._right, 0, 0)

    def get_bottom(self) -> tuple[float, float, float]:
        return (0, self._bottom, 0)

    def get_top(self) -> tuple[float, float, float]:
        return (0, self._top, 0)


class FakeScene:
    def __init__(self, *mobjects: object) -> None:
        self.mobjects = list(mobjects)
        self.time = 0.0


class FakeStyledMobject(FakeMobject):
    """Expose the style surface used by real Manim family members."""

    def __init__(self, *, opacity: float, **bounds: float) -> None:
        super().__init__(**bounds)
        self.fill_opacity = opacity
        self.stroke_opacity = opacity

    def family_members_with_points(self) -> list[FakeStyledMobject]:
        """Return the only geometry-bearing member in this fake family."""
        return [self]


class FakeGroup(FakeMobject):
    """Expose visible descendants through a group-shaped family surface."""

    def __init__(self, *members: FakeStyledMobject, **bounds: float) -> None:
        super().__init__(**bounds)
        self._members = list(members)

    def family_members_with_points(self) -> list[FakeStyledMobject]:
        """Return the geometry-bearing descendants in this fake group."""
        return self._members


def test_recorder_retains_frame_checkpoints_and_stable_object_ids(tmp_path: Path) -> None:
    path = tmp_path / "spatial_trace.json"
    equation = FakeMobject(left=-3, bottom=1, right=3, top=2)
    diagram = FakeMobject(left=-1, bottom=-2, right=1, top=0)
    scene = FakeScene(equation, diagram)
    recorder = SpatialTraceRecorder(path, frame_width=10, frame_height=6)

    recorder.record(scene, kind="play")
    scene.time = 1.5
    scene.mobjects = [equation]
    recorder.record(scene, kind="final")
    recorder.write()

    trace = json.loads(path.read_text())
    assert trace["schema_version"] == "manim-spatial-trace.v1"
    assert trace["frame"] == {"height": 6.0, "width": 10.0}
    assert [checkpoint["index"] for checkpoint in trace["checkpoints"]] == [0, 1]
    assert [checkpoint["kind"] for checkpoint in trace["checkpoints"]] == [
        "play",
        "final",
    ]
    assert trace["checkpoints"][0]["objects"][0] == {
        "bounds": {"bottom": 1.0, "left": -3.0, "right": 3.0, "top": 2.0},
        "id": "object-0",
        "is_container": False,
        "type": "FakeMobject",
    }
    assert trace["checkpoints"][1]["objects"][0]["id"] == "object-0"
    assert trace["checkpoints"][1]["time_seconds"] == 1.5


def test_recorder_retains_bounded_measurement_error_without_aborting_trace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spatial_trace.json"
    scene = FakeScene(object())
    recorder = SpatialTraceRecorder(path, frame_width=10, frame_height=6)

    recorder.record(scene, kind="final")
    recorder.write()

    checkpoint = json.loads(path.read_text())["checkpoints"][0]
    assert checkpoint["objects"] == []
    assert checkpoint["measurement_errors"] == [
        {"object_id": "object-0", "object_type": "object", "reason": "bounds_unavailable"}
    ]


def test_recorder_excludes_fully_invisible_geometry(tmp_path: Path) -> None:
    path = tmp_path / "spatial_trace.json"
    invisible = FakeStyledMobject(
        opacity=0,
        left=9,
        bottom=-1,
        right=11,
        top=1,
    )
    visible = FakeStyledMobject(
        opacity=1,
        left=-1,
        bottom=-1,
        right=1,
        top=1,
    )
    recorder = SpatialTraceRecorder(path, frame_width=10, frame_height=6)

    recorder.record(FakeScene(invisible, visible), kind="final")
    recorder.write()

    checkpoint = json.loads(path.read_text())["checkpoints"][0]
    assert [item["id"] for item in checkpoint["objects"]] == ["object-1"]
    assert checkpoint["measurement_errors"] == []


def test_recorder_bounds_group_from_only_visible_descendants(tmp_path: Path) -> None:
    path = tmp_path / "spatial_trace.json"
    visible = FakeStyledMobject(
        opacity=1,
        left=-1,
        bottom=-1,
        right=1,
        top=1,
    )
    invisible = FakeStyledMobject(
        opacity=0,
        left=9,
        bottom=-1,
        right=11,
        top=1,
    )
    group = FakeGroup(
        visible,
        invisible,
        left=-1,
        bottom=-1,
        right=11,
        top=1,
    )
    recorder = SpatialTraceRecorder(path, frame_width=10, frame_height=6)

    recorder.record(FakeScene(group), kind="final")
    recorder.write()

    checkpoint = json.loads(path.read_text())["checkpoints"][0]
    assert [item["type"] for item in checkpoint["objects"]] == ["FakeGroup"]
    assert checkpoint["objects"][0]["bounds"] == {
        "bottom": -1.0,
        "left": -1.0,
        "right": 1.0,
        "top": 1.0,
    }
    assert checkpoint["objects"][0]["is_container"] is False
