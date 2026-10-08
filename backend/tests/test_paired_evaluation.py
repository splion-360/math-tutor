"""Test complete pairing, raw first-attempt failures, and pending human judgments.
The evaluation must not substitute repaired outputs or infer mathematical quality."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from math_tutor.paired_evaluation import evaluate_pairs, validate_pairs, wilson_interval


def _fixture() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    example = {
        "id": "example",
        "prompt": "Explain an area",
        "topic": "geometry",
        "difficulty": "foundational",
        "review_checks": ["Area is correct"],
    }
    rows = [
        {
            **example,
            "condition": condition,
            "first_attempt": True,
            "response": "not Python %",
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "completion_token_ids": [1, 2, 3, 4],
            "generation_latency_seconds": 1.0,
            "finish_reason": "eos",
            "output_capped": False,
        }
        for condition in ("base", "shared_adapter")
    ]
    return rows, {
        "examples": [example],
        "decoding": {"max_new_tokens": 4096},
        "scope": "test",
        "renderer": {"image": "test", "timeout_seconds": 90},
    }


def test_pair_matrix_rejects_missing_repeated_or_changed_outputs() -> None:
    """Only one unchanged first-attempt response per condition can be scored."""
    rows, plan = _fixture()
    validate_pairs(rows, plan)
    for invalid in (rows[:1], rows + [rows[0]]):
        with pytest.raises(ValueError, match="matrix"):
            validate_pairs(invalid, plan)
    rows[0]["prompt"] = "Changed prompt"
    with pytest.raises(ValueError, match="identity"):
        validate_pairs(rows, plan)


def test_truncation_must_agree_with_measured_tokens() -> None:
    """Capped outputs remain distinct from EOS-terminated failures."""
    rows, plan = _fixture()
    rows[0].update(
        finish_reason="length",
        completion_tokens=4096,
        completion_token_ids=[1] * 4096,
        output_capped=True,
    )
    validate_pairs(rows, plan)
    rows[0]["output_capped"] = False
    with pytest.raises(ValueError, match="cap status"):
        validate_pairs(rows, plan)


def test_raw_parse_failures_are_preserved_without_rendering(tmp_path: Path) -> None:
    """Unrenderable outputs retain raw evidence and receive no human quality score."""
    rows, plan = _fixture()

    class Renderer:
        def render_source(self, job_id: str, source: str, scene_class: str) -> Any:
            pytest.fail("parse failures must not reach the renderer")

    output = tmp_path / "evaluation"
    report = evaluate_pairs(rows=rows, plan=plan, output=output, renderer=Renderer())
    assert report["conditions"]["base"]["render_successes"] == 0
    assert report["conditions"]["base"]["mathematical_correctness"] is None
    assert report["attempts"][0]["failure_category"] == "syntax"
    assert report["attempts"][0]["human_review_status"] == "unreviewable_no_video"
    assert (output / "responses/base-example.txt").read_text() == rows[0]["response"]
    assert (output / "human_review.csv").is_file()


def test_small_pilot_intervals_show_uncertainty_at_extremes() -> None:
    """Zero or perfect observations do not imply zero population uncertainty."""
    zero, perfect = wilson_interval(0, 15), wilson_interval(15, 15)
    assert zero[0] == 0 and zero[1] > 0.2
    assert perfect[0] < 0.8 and perfect[1] == 1


@pytest.mark.parametrize("count,reason,capped", [(4097, "eos", False), (4095, "length", False)])
def test_impossible_termination_records_are_rejected(count: int, reason: str, capped: bool) -> None:
    """Reported usage cannot exceed the cap or claim premature length termination."""
    rows, plan = _fixture()
    rows[0].update(
        completion_tokens=count,
        completion_token_ids=[1] * count,
        finish_reason=reason,
        output_capped=capped,
    )
    with pytest.raises(ValueError, match="termination"):
        validate_pairs(rows, plan)


def test_token_list_must_match_reported_usage() -> None:
    """Measured usage is checked against retained token IDs."""
    rows, plan = _fixture()
    rows[0]["completion_token_ids"] = [1]
    with pytest.raises(ValueError, match="token IDs"):
        validate_pairs(rows, plan)


@pytest.mark.parametrize("failure", [None, "api", "timeout"])
def test_renderer_outcomes_preserve_quality_review_boundary(
    tmp_path: Path, failure: str | None
) -> None:
    """A rendered video and expected render failures remain separate from human judgment."""
    from math_tutor.jobs import RenderOutcome
    from math_tutor.renderer import RenderFailed, RenderTimedOut

    rows, plan = _fixture()
    for row in rows:
        row["response"] = (
            "from manim import *\nclass Demo(Scene):\n    def construct(self):\n        pass"
        )
    output = tmp_path / "evaluation"

    class Renderer:
        def render_source(self, job_id: str, source: str, scene_class: str) -> RenderOutcome:
            if failure == "api":
                raise RenderFailed("API error", diagnostics={"logs": "AttributeError: absent API"})
            if failure == "timeout":
                raise RenderTimedOut("timeout", diagnostics={"logs": "expired"})
            video = output / "renders" / job_id / "video.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"test video")
            return RenderOutcome(
                video_path=video.resolve(), renderer="test", elapsed_seconds=2.0, logs="ok"
            )

    report = evaluate_pairs(rows=rows, plan=plan, output=output, renderer=Renderer())
    attempt = report["attempts"][0]
    assert attempt["mathematical_correctness"] is None
    assert attempt["render_success"] == (failure is None)
    assert attempt["validation_success"]
    if failure is None:
        assert not Path(attempt["video_path"]).is_absolute()
        assert attempt["human_review_status"] == "pending"
    else:
        assert attempt["failure_category"] == ("invalid_api" if failure == "api" else "timeout")
        assert attempt["diagnostics"]["logs"]
