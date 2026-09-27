from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shared_lora_baseline.config import TrainingConfig, tracking_metadata
from shared_lora_baseline.constants import (
    ADAPTER_ID,
    ADAPTER_KIND,
    CONDITION,
    FROZEN_MODEL_ID,
    FROZEN_MODEL_REVISION,
    TRAIN_DEPENDENCY_CONSTRAINTS,
)
from shared_lora_baseline.validation import DatasetValidationReport, validate_training_dataset


@dataclass(frozen=True)
class RunPlan:
    metadata: dict[str, Any]
    redacted_text: str


def build_run_plan(config: TrainingConfig) -> RunPlan:
    report = validate_training_dataset(config.train_path, config.holdout_path)
    metadata = _metadata(config, report)
    redacted_text = _render_plan(metadata)
    return RunPlan(metadata=metadata, redacted_text=redacted_text)


def add_runtime_versions(plan: RunPlan, versions: dict[str, str]) -> RunPlan:
    metadata = {
        **plan.metadata,
        "weights_loaded": True,
        "runtime_versions": dict(sorted(versions.items())),
    }
    return RunPlan(metadata=metadata, redacted_text=_render_plan(metadata))


def add_tracking_metadata(plan: RunPlan, tracking: dict[str, Any]) -> RunPlan:
    metadata = {**plan.metadata, "tracking": tracking}
    return RunPlan(metadata=metadata, redacted_text=_render_plan(metadata))


def write_run_metadata(plan: RunPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metadata(config: TrainingConfig, report: DatasetValidationReport) -> dict[str, Any]:
    return {
        "condition": CONDITION,
        "adapter_id": ADAPTER_ID,
        "adapter_kind": ADAPTER_KIND,
        "model_id": FROZEN_MODEL_ID,
        "model_revision": FROZEN_MODEL_REVISION,
        "dynamic_adapter_spawning": False,
        "learned_router": False,
        "dependency_constraints": dict(sorted(TRAIN_DEPENDENCY_CONSTRAINTS.items())),
        "weights_loaded": False,
        "runtime_versions": {},
        "tracking": tracking_metadata(
            config.tracking,
            active=False,
            reason="not_started",
            git_revision=None,
        ),
        "dataset": {
            "train_path": _redact_path(config.train_path),
            "holdout_path": _redact_path(config.holdout_path),
            "record_count": report.record_count,
            "difficulty_counts": dict(sorted(report.difficulty_counts.items())),
            "training_ids_sha256": report.training_ids_sha256,
            "training_content_sha256": report.training_content_sha256,
            "holdout_content_sha256": report.holdout_content_sha256,
        },
        "training": {
            "seed": config.seed,
            "max_steps": config.max_steps,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "max_seq_length": config.max_seq_length,
            "load_in_4bit": config.load_in_4bit,
            "run_smoke_eval": config.run_smoke_eval,
        },
        "lora": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": list(config.target_modules),
        },
        "artifacts": {
            "output_dir": _redact_path(config.output_dir),
            "metadata_path": _redact_path(config.metadata_path),
        },
    }


def _render_plan(metadata: dict[str, Any]) -> str:
    lines = [
        "Shared-LoRA static control dry run",
        f"condition: {metadata['condition']}",
        f"model_id: {metadata['model_id']}",
        f"model_revision: {metadata['model_revision']}",
        f"adapter_id: {metadata['adapter_id']}",
        "dynamic_adapter_spawning: false",
        "learned_router: false",
        "dataset:",
        f"  train_path: {metadata['dataset']['train_path']}",
        f"  holdout_path: {metadata['dataset']['holdout_path']}",
        f"  record_count: {metadata['dataset']['record_count']}",
        f"  difficulty_counts: {metadata['dataset']['difficulty_counts']}",
        f"  training_ids_sha256: {metadata['dataset']['training_ids_sha256']}",
        f"  training_content_sha256: {metadata['dataset']['training_content_sha256']}",
        f"  holdout_content_sha256: {metadata['dataset']['holdout_content_sha256']}",
        f"weights_loaded: {str(metadata['weights_loaded']).lower()}",
        f"runtime_versions: {metadata['runtime_versions']}",
        f"tracking: {metadata['tracking']}",
        "training:",
    ]
    training = metadata["training"]
    for key in sorted(training):
        lines.append(f"  {key}: {training[key]}")
    lora = metadata["lora"]
    lines.extend(
        [
            "lora:",
            f"  r: {lora['r']}",
            f"  alpha: {lora['alpha']}",
            f"  dropout: {lora['dropout']}",
            f"  target_modules: {lora['target_modules']}",
            "artifacts:",
            f"  output_dir: {metadata['artifacts']['output_dir']}",
            f"  metadata_path: {metadata['artifacts']['metadata_path']}",
        ]
    )
    return "\n".join(lines)


def _redact_path(path: Path) -> str:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return "<workspace>/" + resolved.relative_to(candidate).as_posix()
    return "<external>/" + resolved.name


def config_asdict(config: TrainingConfig) -> dict[str, Any]:
    value: dict[str, Any] = asdict(config)
    value["target_modules"] = list(config.target_modules)
    for key in ("train_path", "holdout_path", "output_dir", "metadata_path"):
        value[key] = str(value[key])
    return value
