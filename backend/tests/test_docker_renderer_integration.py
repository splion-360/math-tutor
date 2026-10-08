"""Exercise the pinned Docker renderer against a real local container runtime.
These integration checks verify media output beyond the mocked command boundary."""

from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient

from math_tutor.api import create_app
from math_tutor.jobs import DispatchingRenderer, LessonService
from math_tutor.rendering.manim import DEFAULT_MANIM_IMAGE, DockerManimRenderer


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
