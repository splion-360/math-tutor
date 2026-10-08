"""Attach identified human video judgments to a paired pilot without changing raw metrics.
Incomplete reviews remain explicit; unrenderable attempts cannot receive video-quality scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from math_tutor.evaluation.paired import CONDITIONS

JUDGMENTS = {"pass", "fail", "uncertain"}


def apply_reviews(
    metrics: dict[str, Any],
    reviews: list[dict[str, str]],
    *,
    automatic_metrics_sha256: str,
) -> dict[str, Any]:
    """Validate the full review matrix and report judgments over the declared scope.

    Args:
        metrics: Original automatic evaluation report.
        reviews: CSV rows with identity, timestamp, judgments, duration, and notes.
        automatic_metrics_sha256: Digest of the original automatic report bytes.

    Returns:
        New report retaining original machine outcomes and explicit review coverage.

    Raises:
        ValueError: If reviews are duplicated, missing, unsupported, or score absent videos.
    """
    attempts = {(row["id"], row["condition"]): row for row in metrics["attempts"]}
    keys = [(row.get("id"), row.get("condition")) for row in reviews]
    if len(keys) != len(set(keys)) or set(keys) != set(attempts):
        raise ValueError("review rows must exactly match the complete attempt matrix")
    if any(row.get("automatic_metrics_sha256") != automatic_metrics_sha256 for row in reviews):
        raise ValueError("review worksheet must be bound to this automatic report")
    result: dict[str, Any] = json.loads(json.dumps(metrics))
    result_rows = {(row["id"], row["condition"]): row for row in result["attempts"]}
    for review in reviews:
        key = (review["id"], review["condition"])
        row = result_rows[key]
        correctness = review.get("mathematical_correctness", "").strip()
        adherence = review.get("prompt_adherence", "").strip()
        if not correctness and not adherence:
            continue
        if not row["render_success"]:
            raise ValueError("video judgments require a rendered video")
        if review.get("video_path") != row["video_path"]:
            raise ValueError("review video path must match the recorded rendered video")
        if correctness not in JUDGMENTS or adherence not in JUDGMENTS:
            raise ValueError("both judgments must be pass, fail, or uncertain")
        reviewer = review.get("reviewer", "").strip()
        try:
            timestamp = datetime.fromisoformat(review.get("reviewed_at_utc", ""))
            duration = float(review.get("duration_seconds", ""))
        except ValueError as error:
            raise ValueError("reviews require a valid timestamp and measured duration") from error
        if not reviewer or timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(None):
            raise ValueError("reviews require an identified reviewer and UTC timestamp")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("reviewed video duration must be positive and finite")
        notes = review.get("notes", "").strip()
        if not notes:
            raise ValueError("completed judgments require evidence notes")
        row.update(
            mathematical_correctness=correctness,
            prompt_adherence=adherence,
            duration_seconds=duration,
            duration_adherence=30 <= duration <= 45,
            human_review_status="reviewed",
            human_reviewer=reviewer,
            reviewed_at_utc=timestamp.isoformat(),
            human_review_notes=notes,
        )
    for condition in CONDITIONS:
        subset = [row for row in result["attempts"] if row["condition"] == condition]
        reviewed = [row for row in subset if row["human_review_status"] == "reviewed"]
        summary = result["conditions"][condition]
        summary.update(
            human_reviewed=len(reviewed),
            reviewable_videos=sum(row["render_success"] for row in subset),
            mathematical_correctness=dict(
                Counter(row["mathematical_correctness"] for row in reviewed)
            ),
            prompt_adherence=dict(Counter(row["prompt_adherence"] for row in reviewed)),
            duration_adherence_count=sum(row["duration_adherence"] for row in reviewed),
            confirmed_lesson_successes=sum(
                row["mathematical_correctness"] == row["prompt_adherence"] == "pass"
                and row["duration_adherence"]
                for row in reviewed
            ),
        )
    eligible = [row for row in result["attempts"] if row["render_success"]]
    result["human_review_status"] = (
        "complete"
        if all(row["human_review_status"] == "reviewed" for row in eligible) and eligible
        else "partial_or_pending"
    )
    result["human_review_note"] = (
        "Unblinded, identified reviewer judgments; review coverage is reported separately. "
        "Uncertain judgments are retained; unreviewed videos are not imputed "
        "as passes or failures."
    )
    return result


def main() -> None:
    """Create a derived reviewed report while preserving original automatic evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    source = args.evaluation / "metrics.json"
    metrics = json.loads(source.read_text())
    source_sha256 = sha256(source.read_bytes()).hexdigest()
    if args.prepare:
        with (args.evaluation / "human_review.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = list(reader.fieldnames or []) + ["automatic_metrics_sha256"]
        if any(row.get("mathematical_correctness") or row.get("prompt_adherence") for row in rows):
            parser.error("prepare requires the original worksheet with blank judgments")
        destination = args.evaluation / "human_review_bound.csv"
        with destination.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({**row, "automatic_metrics_sha256": source_sha256} for row in rows)
        print(f"Review worksheet: {destination}")
        return
    if args.reviews is None:
        parser.error("--reviews is required unless preparing a worksheet")
    with args.reviews.open(newline="") as stream:
        reviews = list(csv.DictReader(stream))
    result = apply_reviews(metrics, reviews, automatic_metrics_sha256=source_sha256)
    result["review_lineage"] = {
        "automatic_metrics_sha256": source_sha256,
        "review_csv_sha256": sha256(args.reviews.read_bytes()).hexdigest(),
        "review_tool_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    destination = args.evaluation / "reviewed_metrics.json"
    with destination.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Human review {result['human_review_status']}: {destination}")


if __name__ == "__main__":
    main()
