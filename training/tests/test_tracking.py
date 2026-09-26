from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from shared_lora_baseline.config import ExperimentTrackingConfig, TrainingConfig
from shared_lora_baseline.dry_run import RunPlan
from shared_lora_baseline.tracking import start_experiment_tracking


def test_wandb_tracking_starts_run_and_groups_train_metrics(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    logged: list[dict[str, int | float | str]] = []
    init_kwargs: dict[str, object] = {}

    class FakeRun:
        url = "https://wandb.ai/demo/math-tutor/runs/abc123"
        summary: dict[str, object] = {}

        def log(self, data: dict[str, int | float | str], *, step: int | None = None) -> None:
            assert step is None
            logged.append(data)

        def finish(self) -> None:
            self.summary["finished"] = True

    fake_run = FakeRun()

    def init(**kwargs: object) -> FakeRun:
        init_kwargs.update(kwargs)
        return fake_run

    monkeypatch.setattr(
        "shared_lora_baseline.tracking.import_module",
        lambda name: SimpleNamespace(init=init) if name == "wandb" else None,
    )
    monkeypatch.setenv("WANDB_API_KEY", "unit-test-key")

    config = TrainingConfig(
        train_path=tmp_path / "train.jsonl",
        holdout_path=tmp_path / "holdout.jsonl",
        output_dir=tmp_path / "adapter",
        metadata_path=tmp_path / "run.json",
        tracking=ExperimentTrackingConfig(
            mode="online",
            project="math-tutor-dynamic-lora",
            run_name="unit-run",
            tags=("dynamic-lora", "unit"),
            modal_artifact_path="modal://dream-ai-training/runs/unit-run",
        ),
    )
    plan = RunPlan(
        metadata={
            "condition": "shared_lora_static_control",
            "adapter_id": "shared-lora-qwen3-4b-manim-v1",
            "adapter_kind": "static_shared_lora",
            "model_id": "Qwen/Qwen3-4B-Instruct-2507",
            "model_revision": "1b4199c4f36b0cef378bfb12390c18780c18af4c",
            "dynamic_adapter_spawning": False,
            "learned_router": False,
            "dataset": {"training_content_sha256": "abc"},
            "training": {"max_steps": 1},
            "lora": {"r": 16},
            "artifacts": {"output_dir": "<workspace>/adapter"},
        },
        redacted_text="",
    )

    tracking_run = start_experiment_tracking(config, plan)
    tracking_run.log_train_metrics(
        {
            "train_loss": 1.25,
            "train_runtime": 3.5,
            "ignored": object(),
        }
    )
    tracking_run.finish()

    assert tracking_run.report_to == ["wandb"]
    assert tracking_run.metadata["active"] is True
    assert tracking_run.metadata["run_url"] == "https://wandb.ai/demo/math-tutor/runs/abc123"
    assert init_kwargs["project"] == "math-tutor-dynamic-lora"
    assert init_kwargs["name"] == "unit-run"
    assert init_kwargs["tags"] == ["dynamic-lora", "unit"]
    assert fake_run.summary["modal_artifact_path"] == "modal://dream-ai-training/runs/unit-run"
    assert fake_run.summary["finished"] is True
    assert logged == [{"train/loss": 1.25, "train/runtime_seconds": 3.5}]


def test_auto_tracking_stays_inactive_without_api_key(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    config = TrainingConfig(
        train_path=tmp_path / "train.jsonl",
        holdout_path=tmp_path / "holdout.jsonl",
        output_dir=tmp_path / "adapter",
        metadata_path=tmp_path / "run.json",
    )
    plan = RunPlan(metadata={}, redacted_text="")

    tracking_run = start_experiment_tracking(config, plan)

    assert tracking_run.report_to == []
    assert tracking_run.metadata["active"] is False
    assert tracking_run.metadata["reason"] == "missing_wandb_api_key"


def test_online_tracking_requires_modal_artifact_path(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "unit-test-key")
    config = TrainingConfig(
        train_path=tmp_path / "train.jsonl",
        holdout_path=tmp_path / "holdout.jsonl",
        output_dir=tmp_path / "adapter",
        metadata_path=tmp_path / "run.json",
        tracking=ExperimentTrackingConfig(mode="online"),
    )
    plan = RunPlan(metadata={}, redacted_text="")

    with pytest.raises(RuntimeError, match="modal_artifact_path"):
        start_experiment_tracking(config, plan)
