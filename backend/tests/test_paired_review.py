"""Test human-review provenance, coverage, and protection of raw pilot outcomes.
Quality judgments require actual videos and cannot be silently imputed."""

from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from math_tutor.paired_review import apply_reviews, main


def _fixture() -> tuple[dict[str, Any], list[dict[str, str]]]:
    attempts = [
        {
            "id": "one",
            "condition": condition,
            "render_success": True,
            "video_path": f"renders/{condition}/lesson.mp4",
            "human_review_status": "pending",
            "mathematical_correctness": None,
            "prompt_adherence": None,
            "duration_adherence": None,
        }
        for condition in ("base", "shared_adapter")
    ]
    reviews = [
        {
            "id": "one",
            "condition": condition,
            "video_path": f"renders/{condition}/lesson.mp4",
            "reviewer": "reviewer-1",
            "reviewed_at_utc": "2026-10-08T06:00:00+00:00",
            "duration_seconds": "35",
            "mathematical_correctness": "pass",
            "prompt_adherence": "pass",
            "automatic_metrics_sha256": "a" * 64,
            "notes": "Watched the full video; the stated calculation and visual steps match.",
        }
        for condition in ("base", "shared_adapter")
    ]
    return {"attempts": attempts, "conditions": {"base": {}, "shared_adapter": {}}}, reviews


def test_reviewed_report_preserves_automatic_evidence() -> None:
    """Identified judgments produce a derived report without overwriting machine results."""
    metrics, reviews = _fixture()
    result = apply_reviews(metrics, reviews, automatic_metrics_sha256="a" * 64)
    assert result["human_review_status"] == "complete"
    assert result["conditions"]["base"]["confirmed_lesson_successes"] == 1
    assert metrics["attempts"][0]["mathematical_correctness"] is None


def test_partial_reviews_do_not_impute_missing_quality() -> None:
    """Coverage and uncertain judgments remain visible in the derived report."""
    metrics, reviews = _fixture()
    reviews[0].update(mathematical_correctness="uncertain", notes="Visual evidence insufficient")
    reviews[1].update(mathematical_correctness="", prompt_adherence="")
    result = apply_reviews(metrics, reviews, automatic_metrics_sha256="a" * 64)
    assert result["human_review_status"] == "partial_or_pending"
    assert result["conditions"]["base"]["mathematical_correctness"] == {"uncertain": 1}
    assert result["conditions"]["shared_adapter"]["human_reviewed"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"reviewer": ""},
        {"reviewed_at_utc": "2026-10-08"},
        {"duration_seconds": "nan"},
        {"mathematical_correctness": "yes"},
        {"prompt_adherence": "fail", "notes": ""},
        {"notes": ""},
        {"video_path": "renders/another-video.mp4"},
        {"automatic_metrics_sha256": "b" * 64},
    ],
)
def test_invalid_human_judgments_are_rejected(change: dict[str, str]) -> None:
    """Review labels, identities, timestamps, duration, and evidence notes are validated."""
    metrics, reviews = _fixture()
    reviews[0].update(change)
    with pytest.raises(ValueError):
        apply_reviews(metrics, reviews, automatic_metrics_sha256="a" * 64)


def test_no_video_cannot_receive_a_video_judgment() -> None:
    """Code inspection cannot masquerade as review of an unrendered video."""
    metrics, reviews = _fixture()
    metrics["attempts"][0]["render_success"] = False
    with pytest.raises(ValueError, match="rendered video"):
        apply_reviews(metrics, reviews, automatic_metrics_sha256="a" * 64)


@pytest.mark.parametrize("duplicate", [False, True])
def test_incomplete_or_duplicated_review_matrix_is_rejected(duplicate: bool) -> None:
    """Review coverage cannot silently drop or duplicate a condition."""
    metrics, reviews = _fixture()
    reviews = [reviews[0], reviews[0]] if duplicate else reviews[:1]
    with pytest.raises(ValueError, match="complete attempt matrix"):
        apply_reviews(metrics, reviews, automatic_metrics_sha256="a" * 64)


def test_prepare_and_import_preserve_original_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI binds the worksheet to exact report bytes and writes a separate reviewed report."""
    metrics, reviews = _fixture()
    source = tmp_path / "metrics.json"
    original = json.dumps(metrics).encode()
    source.write_bytes(original)
    template = tmp_path / "human_review.csv"
    blank_reviews = [
        {key: value for key, value in row.items() if key != "automatic_metrics_sha256"}
        for row in reviews
    ]
    for row in blank_reviews:
        row.update(mathematical_correctness="", prompt_adherence="")
    with template.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(blank_reviews[0]))
        writer.writeheader()
        writer.writerows(blank_reviews)
    monkeypatch.setattr("sys.argv", ["review", "--evaluation", str(tmp_path), "--prepare"])
    main()
    bound = tmp_path / "human_review_bound.csv"
    with bound.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["automatic_metrics_sha256"] == sha256(original).hexdigest() for row in rows)
    for row in rows:
        row.update(mathematical_correctness="pass", prompt_adherence="pass")
    with bound.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(
        "sys.argv", ["review", "--evaluation", str(tmp_path), "--reviews", str(bound)]
    )
    main()
    assert source.read_bytes() == original
    assert json.loads((tmp_path / "reviewed_metrics.json").read_text())["human_review_status"] == (
        "complete"
    )
    with pytest.raises(FileExistsError):
        main()
