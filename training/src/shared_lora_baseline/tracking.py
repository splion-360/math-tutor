from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from shared_lora_baseline.config import TrainingConfig, tracking_metadata
from shared_lora_baseline.dry_run import RunPlan


class WandbRun(Protocol):
    url: str | None
    summary: dict[str, Any]

    def log(self, data: dict[str, int | float | str], *, step: int | None = None) -> None: ...

    def finish(self) -> None: ...


@dataclass(frozen=True)
class ExperimentTrackingRun:
    report_to: list[str]
    metadata: dict[str, Any]
    _wandb_run: WandbRun | None = None

    def log_metrics(self, metrics: Mapping[str, object]) -> None:
        if self._wandb_run is None:
            return
        grouped = _group_metrics(metrics)
        if grouped:
            self._wandb_run.log(grouped)

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()


def start_experiment_tracking(
    config: TrainingConfig,
    plan: RunPlan,
) -> ExperimentTrackingRun:
    tracking = config.tracking
    git_revision = _git_revision(config)

    if tracking.mode == "disabled":
        return ExperimentTrackingRun(
            report_to=[],
            metadata=tracking_metadata(
                tracking,
                active=False,
                reason="disabled",
                git_revision=git_revision,
            ),
        )

    if "WANDB_API_KEY" not in os.environ:
        if tracking.mode == "online":
            raise RuntimeError("WANDB_API_KEY is required when tracking.mode is online")
        return ExperimentTrackingRun(
            report_to=[],
            metadata=tracking_metadata(
                tracking,
                active=False,
                reason="missing_wandb_api_key",
                git_revision=git_revision,
            ),
        )

    missing_provenance = []
    if git_revision is None:
        missing_provenance.append("git_revision")
    if tracking.modal_artifact_path is None:
        missing_provenance.append("modal_artifact_path")
    if missing_provenance:
        reason = "missing_" + "_and_".join(missing_provenance)
        if tracking.mode == "online":
            joined = ", ".join(missing_provenance)
            raise RuntimeError(f"tracking.mode online requires {joined}")
        return ExperimentTrackingRun(
            report_to=[],
            metadata=tracking_metadata(
                tracking,
                active=False,
                reason=reason,
                git_revision=git_revision,
            ),
        )

    try:
        wandb = import_module("wandb")
    except ModuleNotFoundError:
        if tracking.mode == "online":
            raise RuntimeError("wandb must be installed when tracking.mode is online") from None
        return ExperimentTrackingRun(
            report_to=[],
            metadata=tracking_metadata(
                tracking,
                active=False,
                reason="wandb_not_installed",
                git_revision=git_revision,
            ),
        )

    wandb_run = wandb.init(
        project=tracking.project,
        name=tracking.run_name,
        tags=list(tracking.tags),
        job_type="training",
        config=_wandb_config(config, plan, git_revision),
    )
    if tracking.modal_artifact_path is not None:
        wandb_run.summary["modal_artifact_path"] = tracking.modal_artifact_path

    return ExperimentTrackingRun(
        report_to=["wandb"],
        metadata=tracking_metadata(
            tracking,
            active=True,
            git_revision=git_revision,
            run_url=getattr(wandb_run, "url", None),
        ),
        _wandb_run=wandb_run,
    )


def _wandb_config(
    config: TrainingConfig,
    plan: RunPlan,
    git_revision: str | None,
) -> dict[str, Any]:
    return {
        "condition": plan.metadata["condition"],
        "adapter_id": plan.metadata["adapter_id"],
        "adapter_kind": plan.metadata["adapter_kind"],
        "model_id": plan.metadata["model_id"],
        "model_revision": plan.metadata["model_revision"],
        "dynamic_adapter_spawning": plan.metadata["dynamic_adapter_spawning"],
        "learned_router": plan.metadata["learned_router"],
        "dataset": plan.metadata["dataset"],
        "training": plan.metadata["training"],
        "lora": plan.metadata["lora"],
        "artifacts": {
            **plan.metadata["artifacts"],
            "modal_artifact_path": config.tracking.modal_artifact_path,
        },
        "git_revision": git_revision,
    }


def _group_metrics(metrics: Mapping[str, object]) -> dict[str, int | float | str]:
    grouped: dict[str, int | float | str] = {}
    for key, value in metrics.items():
        if not isinstance(value, int | float | str):
            continue
        grouped[_metric_name(key)] = value
    return grouped


def _metric_name(key: str) -> str:
    replacements = {
        "eval_loss": "eval/loss",
        "eval_runtime": "eval/runtime_seconds",
        "eval_samples_per_second": "eval/samples_per_second",
        "eval_steps_per_second": "eval/steps_per_second",
        "train_loss": "train/loss",
        "train_runtime": "train/runtime_seconds",
        "train_samples_per_second": "train/samples_per_second",
        "train_steps_per_second": "train/steps_per_second",
        "total_flos": "train/total_flos",
    }
    return replacements.get(key, key)


def _git_revision(config: TrainingConfig) -> str | None:
    for key in ("GIT_REVISION", "GITHUB_SHA", "SOURCE_VERSION"):
        value = os.environ.get(key)
        if value:
            return value
    workspace = _workspace_root(config.metadata_path)
    if workspace is None:
        workspace = _workspace_root(Path.cwd())
    if workspace is None:
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _workspace_root(path: Path) -> Path | None:
    resolved = path.resolve()
    start = resolved if resolved.is_dir() else resolved.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None
