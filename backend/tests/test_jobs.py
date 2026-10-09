"""Verify in-memory lesson job state transitions and retained artifacts.
The tests exercise JobStore behavior without crossing the HTTP boundary.
"""

from pathlib import Path

from math_tutor.domain import LessonStage
from math_tutor.jobs import JobStore, RenderOutcome


def test_job_store_keeps_first_render_and_merges_live_validator_results(tmp_path: Path) -> None:
    first = tmp_path / "first.mp4"
    repaired = tmp_path / "repaired.mp4"
    first.write_bytes(b"first")
    repaired.write_bytes(b"repaired")
    store = JobStore()
    job = store.create("Explain limits.")
    store.mark_running(job.id)

    store.mark_initial_render(job.id, RenderOutcome(first, "renderer", 0.2, "first"))
    store.mark_initial_render(job.id, RenderOutcome(repaired, "renderer", 0.2, "repair"))
    store.mark_validation_axis(
        job.id,
        {
            "validator": "media",
            "status": "pass",
            "finding_count": 0,
            "advisory_count": 0,
            "provenance": {},
        },
    )
    updated = store.mark_validation_axis(
        job.id,
        {
            "validator": "spatial",
            "status": "fail",
            "finding_count": 1,
            "advisory_count": 0,
            "provenance": {},
        },
    )

    assert updated.initial_video_path == str(first)
    assert updated.diagnostics["validation_axes"] == [
        {
            "validator": "media",
            "status": "pass",
            "finding_count": 0,
            "advisory_count": 0,
            "provenance": {},
        },
        {
            "validator": "spatial",
            "status": "fail",
            "finding_count": 1,
            "advisory_count": 0,
            "provenance": {},
        },
    ]

    ready = store.mark_ready(job.id, RenderOutcome(repaired, "renderer", 0.2, "repair"))
    assert ready.initial_video_path == str(first)
    assert ready.video_path == str(repaired)


def test_job_store_tracks_the_single_repair_attempt() -> None:
    store = JobStore()
    job = store.create("Explain limits visually.")
    store.mark_validation_axis(
        job.id,
        {
            "validator": "media",
            "status": "fail",
            "finding_count": 1,
            "advisory_count": 0,
            "provenance": {},
        },
    )

    repaired = store.mark_stage(job.id, LessonStage.REPAIRING)
    generating = store.mark_stage(job.id, LessonStage.GENERATING_CODE)

    assert repaired.attempt == 1
    assert "validation_axes" not in repaired.diagnostics
    assert generating.attempt == 1
