"""Verify frozen baseline evaluation inputs, metrics, and failure recording.
The tests preserve deterministic first-attempt evaluation semantics."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from math_tutor.evaluation.baseline import (
    AttemptRecord,
    DatasetValidationError,
    aggregate_metrics,
    load_evaluation_slice,
    run_evaluation,
    validate_evaluation_slice,
)
from math_tutor.generation.provider import GenerationConfig, ModelHealth
from math_tutor.settings import Settings


def test_committed_evaluation_slice_has_five_unique_examples_per_difficulty() -> None:
    path = Path("data/evaluation/manim_eval_v1.jsonl")

    examples = load_evaluation_slice(path)

    assert len(examples) == 15
    assert len({example.id for example in examples}) == 15
    assert {
        difficulty: sum(e.difficulty == difficulty for e in examples)
        for difficulty in (
            "foundational",
            "intermediate",
            "advanced",
        )
    } == {"foundational": 5, "intermediate": 5, "advanced": 5}
    assert all(example.split == "evaluation" for example in examples)
    assert all(example.source_name and example.source_reference for example in examples)


def test_evaluation_slice_rejects_training_identifier_overlap() -> None:
    examples = load_evaluation_slice(Path("data/evaluation/manim_eval_v1.jsonl"))

    with pytest.raises(DatasetValidationError, match="overlap"):
        validate_evaluation_slice(examples, training_ids={examples[0].id})


def test_evaluation_slice_rejects_ids_that_are_unsafe_for_artifact_paths() -> None:
    examples = load_evaluation_slice(Path("data/evaluation/manim_eval_v1.jsonl"))
    examples[0] = replace(examples[0], id="../escaped")

    with pytest.raises(DatasetValidationError, match="safe artifact identifiers"):
        validate_evaluation_slice(examples, training_ids=set())


def test_metrics_use_all_first_attempts_as_denominator_and_break_down_difficulty() -> None:
    attempts = [
        _attempt("f-1", "foundational", extraction=True, parse=True, render=True),
        _attempt(
            "i-1",
            "intermediate",
            extraction=True,
            parse=True,
            render=False,
            timed_out=True,
        ),
        _attempt("a-1", "advanced", extraction=False, parse=False, render=False),
        _attempt(
            "a-1-repair",
            "advanced",
            extraction=True,
            parse=True,
            render=True,
            first_attempt=False,
        ),
    ]

    metrics = aggregate_metrics(attempts)

    assert metrics["first_attempt"] == {
        "attempts": 3,
        "extraction_success_rate": pytest.approx(2 / 3),
        "parse_success_rate": pytest.approx(2 / 3),
        "render_pass_at_1": pytest.approx(1 / 3),
        "timeout_rate": pytest.approx(1 / 3),
        "mean_generation_latency_seconds": 0.5,
        "mean_render_latency_seconds": pytest.approx(2 / 3),
        "prompt_tokens": 30,
        "completion_tokens": 60,
        "total_tokens": 90,
    }
    assert metrics["repair_attempts"] == 1
    assert metrics["by_difficulty"]["advanced"]["attempts"] == 1
    assert metrics["by_difficulty"]["advanced"]["render_pass_at_1"] == 0.0


def test_evaluation_records_unexpected_attempt_error_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeClient:
        config = GenerationConfig()

        def health(self) -> ModelHealth:
            return ModelHealth(True, self.config.model, True)

        def close(self) -> None:
            pass

    calls = 0

    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            pass

        def render(self, job_id: str) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("unexpected failure")

    monkeypatch.setattr("math_tutor.evaluation.baseline.ModalVllmClient", lambda **_: FakeClient())
    monkeypatch.setattr("math_tutor.evaluation.baseline.GeneratedLessonPipeline", FakePipeline)

    run_evaluation(
        dataset_path=Path("data/evaluation/manim_eval_v1.jsonl"),
        artifact_root=tmp_path,
        run_id="unexpected-error-run",
        api_key="redacted",
        base_url="https://workspace--qwen.modal.direct/v1",
    )

    attempts = [
        json.loads(line)
        for line in (tmp_path / "unexpected-error-run" / "attempts.jsonl").read_text().splitlines()
    ]
    assert len(attempts) == 15
    assert attempts[0]["failure_stage"] == "error"


def test_unattempted_difficulty_rates_are_unmeasured() -> None:
    metrics = aggregate_metrics(
        [_attempt("f-1", "foundational", extraction=True, parse=True, render=True)]
    )

    assert metrics["by_difficulty"]["intermediate"] == {
        "attempts": 0,
        "extraction_success_rate": None,
        "parse_success_rate": None,
        "render_pass_at_1": None,
        "timeout_rate": None,
        "mean_generation_latency_seconds": None,
        "mean_render_latency_seconds": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_evaluation_cli_uses_shared_modal_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    import math_tutor.evaluation.baseline as evaluation

    observed: dict[str, object] = {}
    settings = Settings(
        _env_file=None,
        modal_vllm_api_key="evaluation-secret",
        modal_vllm_base_url="https://workspace--qwen.modal.direct/v1",
    )

    def fake_run_evaluation(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"first_attempt": {"attempts": 0}}

    monkeypatch.delenv("MODAL_VLLM_API_KEY", raising=False)
    monkeypatch.setattr(evaluation, "get_settings", lambda: settings, raising=False)
    monkeypatch.setattr(evaluation, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "math_tutor.evaluation.baseline",
            "--dataset",
            "data/evaluation/manim_eval_v1.jsonl",
            "--artifact-root",
            str(tmp_path),
            "--run-id",
            "settings-run",
        ],
    )

    evaluation.main()

    assert observed == {
        "dataset_path": Path("data/evaluation/manim_eval_v1.jsonl"),
        "artifact_root": tmp_path,
        "run_id": "settings-run",
        "api_key": "evaluation-secret",
        "base_url": "https://workspace--qwen.modal.direct/v1",
    }


def _attempt(
    example_id: str,
    difficulty: str,
    *,
    extraction: bool,
    parse: bool,
    render: bool,
    timed_out: bool = False,
    first_attempt: bool = True,
) -> AttemptRecord:
    return AttemptRecord(
        example_id=example_id,
        difficulty=difficulty,
        topic="topic",
        prompt="prompt",
        model="Qwen/Qwen3-4B",
        extraction_success=extraction,
        parse_success=parse,
        render_success=render,
        timed_out=timed_out,
        generation_latency_seconds=0.5,
        render_latency_seconds=1.0 if extraction else 0.0,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        first_attempt=first_attempt,
        failure_stage=None if render else "render",
        artifact_directory=f"attempts/{example_id}",
        renderer="docker:test",
    )
