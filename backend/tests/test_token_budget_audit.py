from __future__ import annotations

import json
from pathlib import Path

from math_tutor.token_budget_audit import audit_artifact_root, write_markdown_report


def test_token_budget_audit_measures_specialist_normalization_and_output(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    job_dir = artifact_root / "job-1"
    normalized_dir = artifact_root / "job-1-normalized"
    job_dir.mkdir(parents=True)
    normalized_dir.mkdir(parents=True)
    (job_dir / "prompt.txt").write_text("Explain Taylor polynomials visually.", encoding="utf-8")
    (job_dir / "specialist_response.txt").write_text("Draft visual plan", encoding="utf-8")
    _write_json(
        job_dir / "specialist_generation.json",
        {
            "status": "generated",
            "model": "advanced",
            "finish_reason": "stop",
            "prompt_tokens": 41,
            "completion_tokens": 312,
            "total_tokens": 353,
        },
    )
    (normalized_dir / "prompt.txt").write_text(
        "Create the final lesson.\n<specialist_draft>Draft visual plan</specialist_draft>",
        encoding="utf-8",
    )
    (normalized_dir / "raw_response.txt").write_text("```python\npass\n```", encoding="utf-8")
    _write_json(
        normalized_dir / "generation.json",
        {
            "status": "generated",
            "model": "Qwen/Qwen3-4B",
            "finish_reason": "length",
            "prompt_tokens": 987,
            "completion_tokens": 4096,
            "total_tokens": 5083,
        },
    )

    report = audit_artifact_root(artifact_root)

    assert report["job_count"] == 1
    record = report["records"][0]
    assert record["route"] == "specialist_guided"
    assert record["prompt"]["tokens"] == 41
    assert record["draft"]["tokens"] == 312
    assert record["normalization"]["tokens"] == 987
    assert record["output"]["tokens"] == 4096
    assert record["total_provider_tokens"] == 5436
    assert "Investigate context/output pressure" in report["recommended_limits"][-1]


def test_token_budget_audit_links_failure_examples_to_artifacts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "run" / "attempts"
    job_dir = artifact_root / "failed-job"
    job_dir.mkdir(parents=True)
    (job_dir / "prompt.txt").write_text("Create an invalid scene.", encoding="utf-8")
    (job_dir / "raw_response.txt").write_text("No code fence here.", encoding="utf-8")
    _write_json(
        job_dir / "generation.json",
        {
            "status": "extraction_failed",
            "failure_stage": "extraction",
            "prompt_tokens": 24,
            "completion_tokens": 18,
            "total_tokens": 42,
        },
    )
    (artifact_root.parent / "attempts.jsonl").write_text(
        json.dumps(
            {
                "example_id": "failed-job",
                "failure_stage": "extraction",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_artifact_root(artifact_root)
    markdown = write_markdown_report(report)

    assert report["failure_examples"] == [
        {
            "job_id": "failed-job",
            "failure_stage": "extraction",
            "artifact_directory": str(job_dir),
            "failure_artifacts": [
                str(job_dir / "generation.json"),
                str(job_dir / "raw_response.txt"),
            ],
            "tokens": {
                "prompt": {
                    "phase": "prompt",
                    "tokens": 24,
                    "source": "provider:prompt.txt",
                },
                "draft": {
                    "phase": "draft",
                    "tokens": 0,
                    "source": "not_applicable",
                },
                "normalization": {
                    "phase": "normalization",
                    "tokens": 0,
                    "source": "not_applicable",
                },
                "output": {
                    "phase": "output",
                    "tokens": 18,
                    "source": "provider:raw_response.txt",
                },
                "total_provider_tokens": 42,
            },
        }
    ]
    assert "`failed-job` failed at `extraction`" in markdown
    assert f"Artifacts: `{job_dir}`" in markdown


def test_token_budget_audit_accepts_evaluation_run_root(tmp_path: Path) -> None:
    run_root = tmp_path / "evaluation-run"
    job_dir = run_root / "attempts" / "direct-job"
    job_dir.mkdir(parents=True)
    (job_dir / "prompt.txt").write_text("Explain slope visually.", encoding="utf-8")
    (job_dir / "raw_response.txt").write_text("```python\npass\n```", encoding="utf-8")
    _write_json(
        job_dir / "generation.json",
        {
            "status": "generated",
            "prompt_tokens": 12,
            "completion_tokens": 30,
            "total_tokens": 42,
        },
    )
    (run_root / "attempts.jsonl").write_text(
        json.dumps({"example_id": "direct-job", "failure_stage": None}) + "\n",
        encoding="utf-8",
    )

    report = audit_artifact_root(run_root)

    assert report["job_count"] == 1
    assert report["records"][0]["job_id"] == "direct-job"


def test_token_budget_audit_counts_failed_normalization_and_base_fallback(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    job_dir = artifact_root / "job-2"
    normalized_dir = artifact_root / "job-2-normalized"
    base_dir = artifact_root / "job-2-base"
    job_dir.mkdir(parents=True)
    normalized_dir.mkdir()
    base_dir.mkdir()
    (job_dir / "prompt.txt").write_text("Explain eigenvectors visually.", encoding="utf-8")
    _write_json(
        job_dir / "specialist_generation.json",
        {
            "status": "generated",
            "prompt_tokens": 20,
            "completion_tokens": 100,
            "total_tokens": 120,
        },
    )
    (normalized_dir / "prompt.txt").write_text("normalization prompt", encoding="utf-8")
    _write_json(
        normalized_dir / "generation.json",
        {
            "status": "parse_failed",
            "failure_stage": "parse",
            "prompt_tokens": 40,
            "completion_tokens": 200,
            "total_tokens": 240,
        },
    )
    (base_dir / "prompt.txt").write_text("Explain eigenvectors visually.", encoding="utf-8")
    _write_json(
        base_dir / "generation.json",
        {
            "status": "generated",
            "prompt_tokens": 20,
            "completion_tokens": 250,
            "total_tokens": 270,
        },
    )

    record = audit_artifact_root(artifact_root)["records"][0]

    assert record["output"]["tokens"] == 250
    assert record["total_provider_tokens"] == 630
    assert record["failure_stage"] == "parse"


def test_token_budget_audit_counts_silent_fallback_usage(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    job_dir = artifact_root / "voice-job"
    silent_dir = artifact_root / "voice-job-silent"
    job_dir.mkdir(parents=True)
    silent_dir.mkdir()
    (job_dir / "prompt.txt").write_text("Narrate a derivative lesson.", encoding="utf-8")
    _write_json(
        job_dir / "generation.json",
        {
            "status": "generated",
            "prompt_tokens": 10,
            "completion_tokens": 100,
            "total_tokens": 110,
        },
    )
    _write_json(job_dir / "render.json", {"status": "failed"})
    (silent_dir / "prompt.txt").write_text("Narrate a derivative lesson.", encoding="utf-8")
    _write_json(
        silent_dir / "generation.json",
        {
            "status": "generated",
            "prompt_tokens": 10,
            "completion_tokens": 300,
            "total_tokens": 310,
        },
    )

    record = audit_artifact_root(artifact_root)["records"][0]

    assert record["output"]["tokens"] == 300
    assert record["total_provider_tokens"] == 420
    assert record["failure_stage"] == "render"


def test_token_budget_audit_attributes_output_to_failed_final_fallback(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    job_dir = artifact_root / "voice-job"
    silent_dir = artifact_root / "voice-job-silent"
    job_dir.mkdir(parents=True)
    silent_dir.mkdir()
    (job_dir / "prompt.txt").write_text("Narrate a derivative lesson.", encoding="utf-8")
    _write_json(
        job_dir / "generation.json",
        {
            "status": "generated",
            "prompt_tokens": 10,
            "completion_tokens": 100,
            "total_tokens": 110,
        },
    )
    _write_json(job_dir / "render.json", {"status": "failed"})
    _write_json(
        silent_dir / "generation.json",
        {
            "status": "parse_failed",
            "failure_stage": "parse",
            "prompt_tokens": 10,
            "completion_tokens": 4096,
            "total_tokens": 4106,
        },
    )

    record = audit_artifact_root(artifact_root)["records"][0]

    assert record["output"]["tokens"] == 4096
    assert record["total_provider_tokens"] == 4216
    assert record["failure_stage"] == "parse"


def test_token_budget_audit_attempt_manifest_is_limited_to_selected_run(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "runs" / "selected"
    other = tmp_path / "runs" / "other"
    job_dir = selected / "attempts" / "same-id"
    job_dir.mkdir(parents=True)
    other.mkdir(parents=True)
    (job_dir / "prompt.txt").write_text("Create a scene.", encoding="utf-8")
    _write_json(
        job_dir / "generation.json",
        {
            "status": "generated",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    )
    (selected / "attempts.jsonl").write_text(
        json.dumps({"example_id": "same-id", "failure_stage": "render"}) + "\n",
        encoding="utf-8",
    )
    (other / "attempts.jsonl").write_text(
        json.dumps({"example_id": "same-id", "failure_stage": None}) + "\n",
        encoding="utf-8",
    )

    record = audit_artifact_root(selected)["records"][0]

    assert record["failure_stage"] == "render"


def test_token_budget_audit_detects_render_failure_without_attempt_manifest(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    job_dir = artifact_root / "render-failed"
    job_dir.mkdir(parents=True)
    (job_dir / "prompt.txt").write_text("Create a scene.", encoding="utf-8")
    _write_json(
        job_dir / "generation.json",
        {
            "status": "generated",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    )
    _write_json(job_dir / "render.json", {"status": "timed_out"})

    failure = audit_artifact_root(artifact_root)["failure_examples"][0]

    assert failure["failure_stage"] == "render_timeout"


def test_token_budget_audit_flags_specialist_truncation_pressure(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    job_dir = artifact_root / "job-3"
    normalized_dir = artifact_root / "job-3-normalized"
    job_dir.mkdir(parents=True)
    normalized_dir.mkdir()
    (job_dir / "prompt.txt").write_text("Explain residues visually.", encoding="utf-8")
    _write_json(
        job_dir / "specialist_generation.json",
        {
            "status": "generated",
            "finish_reason": "length",
            "prompt_tokens": 50,
            "completion_tokens": 4096,
            "total_tokens": 4146,
        },
    )
    _write_json(
        normalized_dir / "generation.json",
        {
            "status": "generated",
            "finish_reason": "stop",
            "prompt_tokens": 4200,
            "completion_tokens": 120,
            "total_tokens": 4320,
        },
    )

    report = audit_artifact_root(artifact_root)

    assert report["records"][0]["finish_reasons"] == ("length", "stop")
    assert "Investigate context/output pressure before tuning: job-3" in report[
        "recommended_limits"
    ][-1]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
