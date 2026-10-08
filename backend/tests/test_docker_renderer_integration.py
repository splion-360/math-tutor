"""Exercise the pinned Docker renderer against a real local container runtime.
These integration checks verify media output beyond the mocked command boundary."""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient

from math_tutor.api import create_app
from math_tutor.jobs import DispatchingRenderer, LessonService
from math_tutor.rendering.manim import DEFAULT_MANIM_IMAGE, DockerManimRenderer
from math_tutor.validation.models import RenderedAttempt, ValidationStatus
from math_tutor.validation.spatial import SpatialValidator


@pytest.mark.integration
def test_known_scene_job_returns_mp4_from_locked_down_container(tmp_path: Path) -> None:
    scene = (
        Path(__file__).parents[1]
        / "src"
        / "math_tutor"
        / "rendering"
        / "scenes"
        / "pythagorean_theorem.py"
    )
    renderer = DockerManimRenderer(
        artifact_root=tmp_path / "artifacts",
        scene_path=scene,
        timeout_seconds=90,
    )
    service = LessonService(renderer=DispatchingRenderer({"pythagorean-theorem": renderer}))

    with TestClient(create_app(service)) as client:
        submitted = client.post("/lessons", json={"lesson": "pythagorean-theorem"})
        assert submitted.status_code == 202
        assert submitted.json()["status"] == "queued"
        job_id = submitted.json()["id"]

        deadline = monotonic() + 105
        while monotonic() < deadline:
            lesson = client.get(f"/lessons/{job_id}").json()
            if lesson["status"] in {"ready", "failed"}:
                break
            sleep(0.05)

        assert lesson["status"] == "ready", lesson
        assert lesson["diagnostics"]["renderer"] == f"docker:{DEFAULT_MANIM_IMAGE}"
        video = client.get(lesson["video_url"])

    assert video.status_code == 200
    assert len(video.content) > 1_000
    assert b"ftyp" in video.content[:32]
    assert (tmp_path / "artifacts" / job_id / "render.json").is_file()
    trace_path = tmp_path / "artifacts" / job_id / "output" / "spatial_trace.json"
    trace = json.loads(trace_path.read_text())
    assert trace["schema_version"] == "manim-spatial-trace.v1"
    assert trace["checkpoints"]


@pytest.mark.integration
def test_real_trace_ignores_invisible_helpers_and_accepts_axes_plot(
    tmp_path: Path,
) -> None:
    scene = (
        Path(__file__).parents[1]
        / "src"
        / "math_tutor"
        / "rendering"
        / "scenes"
        / "pythagorean_theorem.py"
    )
    renderer = DockerManimRenderer(
        artifact_root=tmp_path / "artifacts",
        scene_path=scene,
        timeout_seconds=90,
    )
    source = """from manim import *

class GeneratedLesson(Scene):
    def construct(self):
        axes = Axes(x_length=6, y_length=4)
        curve = axes.plot(lambda x: x / 2)
        label = Text("A", font_size=24).move_to([1, 1, 0])
        self.add(
            ValueTracker(10),
            VGroup(Dot(), Square().set_opacity(0).shift(RIGHT * 10)),
            axes,
            curve,
            label,
        )
        self.wait(0.1)
        self.wait(0.1)
"""

    outcome = renderer.render_source("spatial-visibility", source, "GeneratedLesson")

    assert outcome.spatial_trace_path is not None
    trace = json.loads(outcome.spatial_trace_path.read_text())
    object_types = {
        item["type"] for checkpoint in trace["checkpoints"] for item in checkpoint["objects"]
    }
    assert "ValueTracker" not in object_types
    assert "Square" not in object_types
    group_bounds = next(
        item["bounds"]
        for checkpoint in trace["checkpoints"]
        for item in checkpoint["objects"]
        if item["type"] == "VGroup"
    )
    assert group_bounds["right"] < 1
    attempt = RenderedAttempt(
        number=0,
        artifact_dir=tmp_path,
        prompt="Plot a quadratic.",
        source=source,
        scene_class="GeneratedLesson",
        outcome=outcome,
        narration_required=False,
        captions_required=False,
    )
    report = SpatialValidator().validate(attempt)
    assert report.status is ValidationStatus.PASS
    assert "persistent_geometric_intersection" in {advisory.code for advisory in report.advisories}
